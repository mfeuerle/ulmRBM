r""" Collection of some standard problems created with FEniCSx.

Problems
----------------
.. autosummary::
   :toctree: generated/
   
    weak_problem
    thermal_block
    
Utility Functions
-----------------
.. autosummary::
    :toctree: generated/
    
    assemble_matrix
    assemble_vector
    apply_dirichletbc
    assemble_system
"""

__all__ = [
    'apply_dirichletbc',
    'assemble_matrix',
    'assemble_vector',
    'assemble_system',
    'weak_problem',
    'thermal_block',
    ]

import numpy as np
from scipy.sparse import csr_array, sparray
import pyvista as pv
from itertools import product

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from ulmRBM.core import Mu
from ulmRBM.affine import AffineList, AffineObject, AffineLinear, wrap_affinelinear
from ulmRBM.fenicsx import utils, AffineDirichletBC, FEniCSxSpaceWithDirichletBCs


def assemble_matrix(B: ufl.form.Form | AffineList[ufl.form.Form]) -> csr_array | AffineLinear[Mu, csr_array]:
    r"""Assemble the matrix of a (parametric) bilinear form.
    
    Args:
        B:
            (Parametric) bilinear form.
            
    Returns:
        Assembled matrix representation as a (parametric) sparse array.
    """
    
    assemble = lambda B: csr_array(fem.assemble_matrix(fem.form(B)).to_scipy())
    if isinstance(B, AffineList):
        return AffineLinear(B.compress().apply2data(assemble))
    else:
        return assemble(B)


def assemble_vector(l: ufl.form.Form | AffineList[ufl.form.Form]) -> np.ndarray | AffineLinear[Mu,np.ndarray]:
    r"""Assemble the vector of a (parametric) linear form.
    
    Args:
        l:
            (Parametric) linear form.
            
    Returns:
        Assembled vector representation as a (parametric) numpy array.
    """
    
    assemble = lambda l: fem.assemble_vector(fem.form(l)).array
    if isinstance(l, AffineList):
        return AffineLinear(l.compress().apply2data(assemble))
    else:
        return assemble(l)
    
    
def apply_dirichletbc(B: np.ndarray | sparray | AffineList[np.ndarray | sparray], 
                      f: np.ndarray |AffineList[np.ndarray], 
                      U: FEniCSxSpaceWithDirichletBCs, 
                      V: FEniCSxSpaceWithDirichletBCs) -> tuple[AffineLinear[Mu, np.ndarray | sparray], AffineLinear[Mu, np.ndarray]]:
    r"""Apply Dirichlet boundary conditions to a linear system.
    
    Starting from the system assembled on the full trial/test spaces,

    .. math::
        Bu = f,\qquad u[D_U] = g[D_U],
        
    and denoting the free and dirichlet degrees of freedom of trial and test space by :math:`F_U, D_U` and :math:`F_V, D_V`, respectively, with :math:`g` denoting the dirchlet boundary conditions stored in ``U.bcs``, the system has the block structure
    
    .. math::
        B = \begin{bmatrix} B[F_V, F_U] & B[F_V, D_U] \\
        B[D_V, F_U] & B[D_V, D_U] \end{bmatrix},\qquad
        f = \begin{bmatrix} f[F_V] \\ f[D_V] \end{bmatrix},\qquad
        u = \begin{bmatrix} u[F_U] \\ u[D_U] \end{bmatrix}.
        
    To enforce the dirchlet boundary condition, this function reduces the system to the free degrees of freedom of trial and test space, removing all dirichlet dofs from the test space and moving the dirichlet boundary condition on the trial space to the right-hand side:
    
    .. math::
        \tilde{B} u[F_U] = \tilde{f},\qquad
        \tilde{B} = B[F_V, F_U],\qquad
        \tilde{f} = f[F_V] - B[F_V, D_U]g[D_U].

    Args:
        B :
            System matrix assembled on the full spaces, i.e. ``B.shape = (V.dim,U.dim)``.
        f :
            Right-hand side vector assembled on the full test space, i.e. ``f.shape = (V.dim,)``.
        U :
            Trial space including the Dirichlet boundary data :math:`g` and the and the dof split :mat:`F_U,D_U`.
        V :
            Test space including the dof split :mat:`F_V,D_V`. The dirichlet dofs of the test space are removed in the final system, enforcing homogeneous Dirichlet constraints on the test space.
    
    Returns:
        B :
            Reduced system matrix :math:`\tilde{B}` on the free trial/test dofs, i.e. ``B.shape = (sum(V.dofs), sum(U.dofs))``.
        f :
            Reduced right-hand side :math:`\tilde{f}` on the free test dofs, i.e. ``f.shape = (sum(V.dofs),)``.
    """    
    
    B = wrap_affinelinear(B)
    f = wrap_affinelinear(f)
    
    f = f.apply2data(       lambda fq: fq[V.dofs]             ) \
        - sum( B.apply2data(lambda Bq: Bq[V.dofs,:][:,bc.dofs]) @ bc for bc in U.bcs )
    B = B.apply2data(       lambda Bq: Bq[V.dofs,:][:,U.dofs] )
    
    return B.compress(), f.compress()


def assemble_system(B: ufl.form.Form | AffineObject[Mu,ufl.form.Form],
                    f: ufl.form.Form | AffineObject[Mu,ufl.form.Form],
                    U: FEniCSxSpaceWithDirichletBCs, 
                    V: FEniCSxSpaceWithDirichletBCs) -> tuple[AffineLinear[Mu,csr_array], AffineLinear[Mu,np.ndarray]]:
    r"""Assemble a (parametric) linear system and applying Dirichlet boundary conditions.
    
    Just a wrapper around ``assemble_matrix``, ``assemble_vector`` and ``apply_dirichletbc`` for convenience.
    """
    
    return apply_dirichletbc(assemble_matrix(B), assemble_vector(f), U, V)


def weak_problem(msh: mesh.Mesh, 
                 operator: list[AffineObject, AffineObject, AffineObject], 
                 data: list[AffineList, list[AffineList], list[AffineList]] | list[any,any], 
                 dbdry_U: list = [], 
                 nbdry_U: list = [], 
                 dbdry_V: list = None, 
                 U: fem.FunctionSpace | None = None, 
                 V: fem.FunctionSpace | None = None,
                 A_Space: fem.FunctionSpace | None = None, 
                 B_space: fem.FunctionSpace | None = None, 
                 C_space: fem.FunctionSpace | None = None, 
                 F_space: fem.FunctionSpace | None = None, 
                 H_spaces: list[fem.FunctionSpace] | None = None) -> tuple[AffineLinear, AffineLinear, FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Build the FEM weak formulation of a general second-order operator with
    inhomogeneous Dirichlet and Neumann boundary data.

    Consider the second order PDE

    .. math::
        \nabla \cdot \left(A\nabla u\right) + b \cdot \nabla u + c\,u = f
        \quad \text{in } \Omega,

    .. math::
        u = g_i \quad \text{on } \Gamma_D^i, \quad i=1,...,N_D,
        
    .. math::
        (A\nabla u)\cdot n = h_i \quad \text{on } \Gamma_N^i, \quad i=1,...,N_N.

    using the weak formulation to find u in a trial space U such that
    for all v in a test space V it holds
    
    .. math::
        b(u, v) = f(v)
        
    with 
    
    .. math::
        b(u, v) = -\left(A\nabla u, \nabla v\right)_{\Omega}
        + \left(b\cdot\nabla u, v\right)_{\Omega}
        + \left(cu, v\right)_{\Omega},
        
    and
    
    .. math::
        f(v) = \left(f, v\right)_{\Omega} - \left(h, v\right)_{\Gamma_N}.
        
    The discrete system is assembled first on the full spaces, resulting in a linear system
    
    .. math::
        Bu = f,
        
    with 
    
    .. math:: 
        B = \begin{bmatrix} B[F_V, F_U] & B[F_V, D_U] \\ B[D_V, F_U] & B[D_V, D_U] \end{bmatrix},\qquad
        f = \begin{bmatrix} f[F_V] \\ f[D_V] \end{bmatrix},\qquad
        u = \begin{bmatrix} u[F_U] \\ u[D_U] \end{bmatrix}.
    
    Thereby :math:`F_U` and :math:`F_V` denote the free degrees of freedom of trial and test space, respectively, while :math:`D_U` and :math:`D_V` denote the dirichlet degrees of freedom. The system is then reduced to the free degrees of freedom according to Dirichlet constraints, removing all dirichlet dofs from the test space and moving the dirichlet boundary condition on the trial space to the right-hand side, resulting in
    
    .. math::
        \tilde{B}\tilde{u} = \tilde{f},\qquad
        \tilde{B} = B[F_V, F_U],\quad
        \tilde{f} = f[F_V] - B[F_V, D_U]g[D_U].

    The final solution then reads :math:`u[F_U] = \tilde{u}` and :math:`u[D_U] = g[D_U]`.

    Parameters
    ----------
    msh :
        Mesh of the domain :math:`\Omega \in \mathbb{R}^{gdim}`.
    operator :
        Tuple (A, b, c) of affine coefficient objects, where A is the :math:`\mathbb{R}^{gdim \times gdim}` diffusion matrix, b  is the :math:`\mathbb{R}^{gdim}` convection vector, and c is the :math:`\mathbb{R}` reaction coefficient. All have to be compatible with `utils.interpolate_function`.
    data :
        Either:
        1) (u_exact, mu_exact): 
        :math:`f,g,h` are calculated from an exact solution :math:`u_{exact}` at a given parameter value :math:`\mu_{exact}`. Has to be compatible with `utils.interpolate_function`.
        2) (f, g, h):
        User-provided affine right-hand side, Dirichlet data list, and
        Neumann data list, have to be compatible with `utils.interpolate_function`.
    dbdry_U :
        List of boundary locator callables for Dirichlet boundaries of trial space U compateble with `dolfinx.mesh.locate_entities_boundary`.
    nbdry_U :
        List of boundary locator callables for Neumann boundaries compateble with `dolfinx.mesh.locate_entities_boundary`.
    dbdry_V :
        List of boundary locator callables for homogeneous Dirichlet boundaries
        of test space V compateble with `dolfinx.mesh.locate_entities_boundary`. If None, it is set to :math:`\partial\Omega \setminus \bigcup_i \Gamma_N^i`, i.e. all boundaries that are not Neumann boundaries for U are treated as homogeneous Dirichlet boundaries for V.
    U, V :
        Optional trial/test spaces. Defaults are H1, i.e. first-order Lagrange spaces.
    A_Space, B_space, C_space, F_space, H_spaces :
        Optional interpolation spaces for :math:`A,b,c,f,h`. Defaults are L2, i.e. piecewise constant discontinuous Galerkin spaces.

    Returns
    -------
    B :
        AffineLinear of the system matrix :math:`\tilde{B}` reduced to the free degrees of freedom.
    f :
        AffineLinear of the reduced right-hand side :math:`\tilde{f}`.
    U :
        Trial space wrapper with Dirichlet metadata.
    V :
        Test space wrapper with Dirichlet metadata.
    """
    
    tdim = msh.topology.dim
    gdim = msh.geometry.dim
    msh.topology.create_connectivity(tdim-1, tdim)
    
    ########################################
    # DEFAULT SPACES
    ########################################
    
    if U is None: 
        U = fem.functionspace(msh, ("Lagrange", 1)) # trial space
    if V is None: 
        V = fem.functionspace(msh, ("Lagrange", 1)) # test space
    if A_Space is None: 
        A_Space = fem.functionspace(msh, ("DG", 0, (gdim, gdim)))     # space for the diffusion matrix aa
    if B_space is None: 
        B_space = fem.functionspace(msh, ("DG", 0, (gdim,)))          # space for the convection vector bb
    if C_space is None: 
        C_space = fem.functionspace(msh, ("DG", 0))                   # space for the reaction coefficient cc
    if F_space is None: 
        F_space = fem.functionspace(msh, ("DG", 0))                   # space for rhs
    if H_spaces is None: 
        H_spaces = [fem.functionspace(msh,("DG", 0))] * len(nbdry_U) # spaces for the neumann boundary conditions
    
    ########################################
    # DISCRETIZ OPERATOR AND DATA
    ########################################
    
    A, b, c = operator
    
    if len(data) == 2:
        u_exact, mu_exact = data
        
        g = [AffineObject([1.0], [u_exact])] * len(dbdry_U)
    
        _u = [utils.interpolate_function(utils.change_element(H_space, add_degree=1), u_exact) for H_space in H_spaces]
        _A = utils.interpolate_function(utils.change_element(F_space, shape=(gdim, gdim)), A(mu_exact))
        h  = [AffineObject([1.0], [ufl.inner(_A * ufl.grad(_u), ufl.FacetNormal(msh))]) for _u in _u]
        
        _u = utils.interpolate_function(utils.change_element(F_space, add_degree=2), u_exact)
        _A = utils.interpolate_function(utils.change_element(F_space, add_degree=1, shape=(gdim, gdim)), A(mu_exact))
        _b = utils.interpolate_function(utils.change_element(F_space, shape=(gdim,)), b(mu_exact))
        _c = utils.interpolate_function(F_space, c(mu_exact))
        f = AffineObject([1.0], [ufl.div(_A*ufl.grad(_u)) + ufl.inner(_b,ufl.grad(_u)) + _c*_u])
    
    elif len(data) == 3:
        f, g, h = data
        
        f = f.apply2data(lambda fq: utils.interpolate_function(F_space, fq))
        g = [g.apply2data(lambda gq: utils.interpolate_function(U, gq)) for g in g]
        h = [h.apply2data(lambda hq: utils.interpolate_function(H_space, hq)) for h, H_space in zip(h, H_spaces)]
        
    else:
        raise ValueError("data has to be either (u_exact, mu_exact) or (f, g, h)")
    
    A = A.apply2data(lambda Aq: utils.interpolate_function(A_Space, Aq))
    b = b.apply2data(lambda bq: utils.interpolate_function(B_space, bq))
    c = c.apply2data(lambda cq: utils.interpolate_function(C_space, cq))
    
    ########################################
    # BOUNDARY CONDITIONS
    ########################################
    
    dbdry_U = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  dbdry_U]
    nbdry_U = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  nbdry_U]
    if dbdry_V is None:
        _all_bdry = mesh.exterior_facet_indices(msh.topology)
        dbdry_V = [np.setdiff1d(_all_bdry, np.concatenate(nbdry_U))] if len(nbdry_U) > 0 else [_all_bdry]
    else:
        dbdry_V = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  dbdry_V]

    # create dirichlet boundary conditions
    dbcs_U = [AffineDirichletBC(U, g, bdry) for g, bdry in zip(g, dbdry_U)]
    dbcs_V = [AffineDirichletBC(V, 0.0, bdry) for bdry in dbdry_V]
    
    U = FEniCSxSpaceWithDirichletBCs(U, dbcs_U)
    V = FEniCSxSpaceWithDirichletBCs(V, dbcs_V, warn=False)

    # boundary measure for the neumann boundary parts
    ds = utils.create_measure("ds", msh, tdim-1, nbdry_U)

    ########################################
    # VARIATIONAL FORMULATION
    ########################################

    u = ufl.TrialFunction(U.space)
    v = ufl.TestFunction(V.space)

    B_ufl = - A.apply2data(lambda Aq: ufl.inner(Aq * ufl.grad(u), ufl.grad(v)) * ufl.dx) \
            + b.apply2data(lambda bq: ufl.inner(bq, ufl.grad(u)) * v * ufl.dx) \
            + c.apply2data(lambda cq: cq * u * v * ufl.dx)
        
    f_ufl = f.apply2data(lambda fq: fq * v * ufl.dx) \
            - sum(h.apply2data(lambda hq: hq * v * ds(i)) for i,h in enumerate(h))

    ########################################
    # ASSEMBLE SYSTEM
    ########################################
    
    B, f = assemble_system(B_ufl, f_ufl, U, V)
    
    return B, f, U, V


def thermal_block(nh: list[int,int], nblocks: list[int,int], plot: bool = False) -> tuple[AffineLinear, AffineLinear, FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Create the parametric thermal block problem.

    The domain is the unit square :math:`\Omega = (0,1)^2`, partitioned into
    :math:`n_{\mathrm{blocks},1} \times n_{\mathrm{blocks},2}` rectangular blocks.
    The model uses

    .. math::
        b(x) = 0, \qquad c(x) = 0, \qquad f(x) = 1,

    and a blockwise affine-parametric diffusion tensor

    .. math::
        A_\mu(x) = -\sum_{q=1}^{Q} \theta_q(\mu)\,\chi_q(x)\,I,

    where :math:`\chi_q` are indicator functions of the blocks and
    :math:`\theta_q(\mu) = \mu_q`.

    All exterior facets are treated as Dirichlet boundaries for 
    trial and test spaces, i.e. :math:`\Gamma_D = \partial\Omega`
    and :math:`\Gamma_N = \emptyset` with dirichlet data :math:`g=0`.

    Parameters
    ----------
    nh : 
        Number of mesh cells in each spatial direction.
    nblocks : 
        Number of thermal blocks in each spatial direction.
    plot :
        If ``True``, visualize all block indicator functions :math:`\chi_q`.

    Returns
    -------
    See `weak_problem` for details on the return values.
    """
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [1, 1]], nh)
    gdim = msh.geometry.dim
    
    dbdry = [lambda x: np.ones(x.shape[1], dtype=bool)]
    
    f = AffineObject([1.0], [1.0])
    g = [AffineObject([0.0], [1.0])]
    h = []
    
    blocks = [np.linspace(0, 1, nblocks[i]+1) for i in range(2)]
    
    def chi(x, block_id):
        value = np.ones(x.shape[1], dtype=bool)
        for i,id in enumerate(block_id):
            if id != 0:
                value &= blocks[i][id] <= x[i]
            if id != nblocks[i] - 1:
                value &= x[i] < blocks[i][id+1]
        return value

    A = AffineObject()
    for idx in product(*[range(n) for n in nblocks]):
        A += [(lambda mu, idx=idx: mu[idx], lambda x, idx=idx: -np.eye(gdim).reshape(-1,1) * chi(x, idx) )]
    b = AffineObject([0.0], [np.zeros(gdim)])
    c = AffineObject([0.0], [0.0])
        
    if plot:
        plotter = pv.Plotter(shape=(nblocks))
        L2 = fem.functionspace(msh, ("DG", 0))
        for idx in product(range(nblocks[0]), range(nblocks[1])):
            plotter.subplot(*idx)
            tmp = fem.Function(utils.change_element(L2, shape=()))
            tmp.interpolate(lambda x: chi(x, idx))
            utils.plot_pyvista(tmp.x.array, utils.change_element(L2, shape=()), f"chi {idx}", plotter)
        plotter.show(interactive_update=True)
    
    return weak_problem(msh, (A,b,c), (f,g,h), dbdry)