import numpy as np
import pyvista as pv
from itertools import product
from collections.abc import Callable

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from ulmRBM.core import Mu, Matrix, Number, Vector
from ulmRBM.affine import AffineObject, AffineFunction, AffineLinear
from ulmRBM.fenicsx import utils, AffineDirichletBC, FEniCSxSpaceWithDirichletBCs, assemble_system, apply_dirichletbc

__all__ = [
    'weak_problem',
    'thermal_block',
    'simple_elliptic',
]


def weak_problem(msh: mesh.Mesh, 
                 operator: list[AffineObject, AffineObject, AffineObject], 
                 data: list[AffineObject, list[AffineObject], list[AffineObject]] | list[any,any], 
                 dbdry_U: list = [], 
                 nbdry_U: list = [], 
                 dbdry_V: list = None, 
                 U: fem.FunctionSpace | None = None, 
                 V: fem.FunctionSpace | None = None,
                 A_Space: fem.FunctionSpace | None = None, 
                 B_space: fem.FunctionSpace | None = None, 
                 C_space: fem.FunctionSpace | None = None, 
                 F_space: fem.FunctionSpace | None = None, 
                 H_spaces: list[fem.FunctionSpace] | None = None) -> tuple[AffineObject[Mu,ufl.Form], AffineObject[Mu, ufl.Form], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Weak formulation of a general 2nd-order operator with
    inhomogeneous Dirichlet and Neumann boundary data. Suitable e.g. for elliptic problems or the wave equation.

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
        b(u, v) = -\left(A\nabla u, \nabla v\right)_{L^2(\Omega)}
        + \left(b\cdot\nabla u, v\right)_{L^2(\Omega)}
        + \left(cu, v\right)_{L^2(\Omega)},
        
    and
    
    .. math::
        f(v) = \left(f, v\right)_{L^2(\Omega)} - \sum_{i=1}^{N_N} \left(h_i, v\right)_{L^2(\Gamma_N^i)}.,
        
    where :math:`u` is restricted to the dirichlet boundary conditions :math:`g_i` on :math:`\Gamma_D^i` and the test functions :math:`v` are restricted to be zero on the dirichlet boundaries of the test space as given in ``dbdry_V``.
    
    To assemble the discrete system, follow up with `assemble_system`.

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
        Bilinear form :math:`b`, see also `utils.assemble_matrix` and `assemble_system`.
    f :
        Linear form :math:`f`, see also `utils.assemble_vector` and `assemble_system`.
    U :
        Trial space with dirichlet boundary conditions.
    V :
        Test space with dirichlet boundary conditions.
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
    # DISCRETIZE OPERATOR AND DATA
    ########################################
    
    A, b, c = operator
    
    if len(data) == 2:
        u_exact, mu_exact = data
        
        g = [AffineObject([1.0], [u_exact])] * len(dbdry_U)
    
        _u = [utils.interpolate_function(utils.change_element(H_space, add_degree=1), u_exact) for H_space in H_spaces]
        _A = [utils.interpolate_function(utils.change_element(H_space, shape=(gdim, gdim)), A(mu_exact)) for H_space in H_spaces]
        h  = [AffineObject([1.0], [ufl.inner(__A * ufl.grad(__u), ufl.FacetNormal(msh))]) for __u, __A in zip(_u, _A)]
        
        _u = utils.interpolate_function(utils.change_element(F_space, add_degree=2), u_exact)
        _A = utils.interpolate_function(utils.change_element(F_space, add_degree=1, shape=(gdim, gdim)), A(mu_exact))
        _b = utils.interpolate_function(utils.change_element(F_space, shape=(gdim,)), b(mu_exact))
        _c = utils.interpolate_function(F_space, c(mu_exact))
        f = AffineObject([1.0], [ufl.div(_A*ufl.grad(_u)) + ufl.inner(_b,ufl.grad(_u)) + _c*_u])
    
    elif len(data) == 3:
        f, g, h = data
        
        if not isinstance(f, AffineObject): f = AffineObject([1.0], [f])  # right-hand side
        g = [AffineObject([1.0], [g_]) if not isinstance(g_, AffineObject) else g_ for g_ in g]  # dirichlet boundary condition list
        h = [AffineObject([1.0], [h_]) if not isinstance(h_, AffineObject) else h_ for h_ in h]  # neumann boundary condition list
        
        f = f.apply2data(lambda fq: utils.interpolate_function(F_space, fq))
        g = [g_.apply2data(lambda gq: utils.interpolate_function(U, gq)) for g_ in g]
        h = [h_.apply2data(lambda hq: utils.interpolate_function(H_space, hq)) for h_, H_space in zip(h, H_spaces)]
        
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
    dbcs_U = [AffineDirichletBC(U, g_, bdry) for g_, bdry in zip(g, dbdry_U)]
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
            - sum(h_.apply2data(lambda hq: hq * v * ds(i)) for i,h_ in enumerate(h))

    return B_ufl, f_ufl, U, V


def simple_elliptic(n: int | list[int] = 10, 
                    f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1, 
                    g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Parametric elliptic problem on the unit square.

    The model uses `weak_problem` with

    .. math::
        A_\mu(x) = -\begin{pmatrix}1 & & & \\ & 0 &\ddots & \\ & & & 0\end{pmatrix} -\mu\begin{pmatrix}0 & & & \\ & 1 &\ddots & \\ & & & 1\end{pmatrix}, \qquad b(x) = 0, \qquad c(x) = 0,

    where :math:`\mu` is the parameter. All exterior facets are treated as
    Dirichlet boundaries for trial and test spaces, i.e. :math:`\Gamma_D =
    \partial\Omega`.

    Args:
        n : 
            Number of mesh cells in each spatial direction, where ``len(n)`` is the spatial dimension.
        f :
            (Parametric) right-hand side. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        g :
            (Parametric) Dirichlet boundary condition on :math:`\partial\Omega`.
            Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
    """
    
    if np.isscalar(n) or len(n) == 1:
        if not np.isscalar(n): n = n[0]
        msh = mesh.create_interval(MPI.COMM_WORLD, n, [0,1])
    elif len(n) == 2:
        msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [1,1]], n)
    elif len(n) == 3:
        msh = mesh.create_box(MPI.COMM_WORLD, [[0,0,0], [1,1,1]], n)
    else:
        raise ValueError("Only 1D, 2D and 3D in space are supported.")

    gdim = msh.geometry.dim

    A1 = np.diag([1.0] + (gdim-1)*[0.0]) 
    A2 = np.diag([0.0] + (gdim-1)*[1.0])
    A = -AffineObject([1.0, lambda mu: mu], [A1, A2])
    b = AffineObject([0.0], [np.ones(gdim)])
    c = AffineObject([0.0], [1.0])
    
    dbdry = [lambda x: np.ones(x.shape[1], dtype=bool)]
    
    B, f, U, V = weak_problem(msh, (A,b,c), (f,[g],[]), dbdry)
    B, f = assemble_system(B, f, U, V)
    return B, f, U, V


def thermal_block(nh: list[int,int],
                  nblocks: list[int,int],
                  output_mode: int = 0,
                  plot: bool = False) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs, AffineLinear[Mu,Vector], AffineLinear[Mu,Number],]:
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
       
    For ``outputmode = 1`` the output is defined as the flux over the domain boundary, i.e.

    .. math::
        s_\mu(u_\mu(x)) = \int_{\partial\Omega} -A_\mu(x) \nabla_x u_\mu(x) n \, \text{d} x

    For ``outputmode = 2`` the output is defined as the difference in the solution between 
    the coordinates [0.25, 0.25] and [0.75, 0.75], i.e.

    .. math::
        s_\mu(u_\mu(x)) = u_\mu([0.25, 0.25]) - u_\mu([0.75, 0.75])

    Parameters
    ----------
    nh : 
        Number of mesh cells in each spatial direction.
    nblocks : 
        Number of thermal blocks in each spatial direction.
    output_mode:
        Choose output. 0 for no output. 1 for flux over boundary. 2 for temperature difference beteen coordinates [0.25, 0.25] and [0.75, 0.75]
    plot :
        If ``True``, visualize all block indicator functions :math:`\chi_q`.

    Returns
    -------
    See `weak_problem` for details on the return values.
    l :
        Output functional for :py:attr:`ulmRBM.fom.FOM.l`.
    s0 :
        Contribution of the Dirichlet boundaries to the output.
    """
    msh = mesh.create_unit_square(MPI.COMM_WORLD, *nh)
    gdim = msh.geometry.dim
    
    dbdry = [lambda x: np.ones(x.shape[1], dtype=bool)]
    
    f = AffineObject([1.0], [1.0])
    g = [AffineObject([0.0], [0.0])]
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
    
    
    B, f, U, V = weak_problem(msh, (A,b,c), (f,g,h), dbdry)
    
    if output_mode==0: # No output
        B, f = assemble_system(B, f, U, V)
        return B, f, U, V
    
    # Output computation
    
    tdim = msh.topology.dim
    gdim = msh.geometry.dim

    u = ufl.TrialFunction(U.space)
    v = ufl.TrialFunction(V.space)

    if output_mode == 1: # Flux over Dirichlet boundary
        A = A.apply2data(lambda Aq: utils.interpolate_function(fem.functionspace(msh, ("DG", 0, (gdim, gdim))), Aq))
        A.apply2data(lambda Aq: ufl.inner(Aq * ufl.grad(u), ufl.grad(v)) * ufl.dx)
        
        all_bdry = [mesh.locate_entities_boundary(msh, tdim-1, lambda x: np.full(x.shape[1], True, dtype=bool))]
        ds = utils.create_measure("ds", msh, tdim-1, all_bdry)
        l = A.apply2data(lambda Aq: ufl.dot(Aq * ufl.grad(u), ufl.FacetNormal(msh)) * ds)

        B, f, l, s0 = assemble_system(B, f, U, V, l)
    elif output_mode == 2: # Temperature difference between coordinates [0.25, 0.25] and [0.75, 0.75] 
        l_vecs = utils.point_functional(U.space, np.array([[0.25, 0.25, 0],[0.75, 0.75, 0]]))
        l = l_vecs[0,:]-l_vecs[1,:]
        B, f, l, s0 = apply_dirichletbc(utils.assemble_matrix(B), utils.assemble_vector(f), U, V, l)
    else:
        raise ValueError('Unknown output case for thermalblock')
    
    return B, f, U, V, l, s0