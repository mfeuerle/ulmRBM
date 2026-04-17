r""" Collection of some standard problems created with FEniCSx.

Classes
-------
.. autosummary::
   :toctree: generated/
   
    FEniCSxSpaceWithDirichletBCs

Functions
----------------
.. autosummary::
   :toctree: generated/
   
    weak_problem
    thermal_block
"""

__all__ = [
    'FEniCSxSpaceWithDirichletBCs',
    'weak_problem',
    'thermal_block',
    ]

import numpy as np
from scipy.sparse import csr_array
import pyvista as pv
from itertools import product

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from ulmRBM.core import NO_MU
from ulmRBM.affine import AffineList, AffineObject, AffineLinear
from ulmRBM.fenicsx import utils, AffineDirichletBC, free_dofs


class FEniCSxSpaceWithDirichletBCs:
    r""" Combination of a FEniCSx FunctionSpace and boundary conditions.
    """
    def __init__(self, space: fem.FunctionSpace, bcs: list[AffineDirichletBC], warn: bool = True):
        r"""
        Args:
            space:
                Function space to be wrapped.
            bcs:
                List of dirichlet boundary conditions.
            warn:
                Whether to warn, if there are dofs set by multiple boundary conditions.
        """
        
        self.space: fem.FunctionSpace = space
        "Function space"
        self.dim: int = space.dofmap.index_map.size_global
        "Dimension of the space (number of free + restricted dofs)."
        self.bcs: list[AffineDirichletBC] = bcs
        "List of all dirichlet boundary conditions of this space."
        self.dofs: np.ndarray[bool] = free_dofs(bcs, warn=warn) if len(bcs) > 0 else np.ones(self.dim, dtype=bool)
        "Boolean array of the free dofs (True if free, False if restricted by a dirichlet boundary condition)."
        self._ndofs = sum(self.dofs)
        "Total number of free dofs."
        
    def __repr__(self):
        return f"<{self.__class__.__name__} of dim={self.dim} and {self._ndofs} free dofs>"
        
    def set_dirichletbcs(self, mu, u: np.ndarray | fem.Function):
        r""" Insert the dirichlet boundary conditions into a given vector.
        
        Args:
            mu:
                Parameter value, at which the boundary conditions should be evaluated.
            u:
                Vector (or FEniCS function) containing the values at the free dofs, to which the dirichlet boundary conditions should be added. Can either have length equal to the total number of dofs (free + restricted) or only the number of free dofs.
        """ 
        
        is_fenics = isinstance(u, fem.Function)
        
        if is_fenics:
            u_fenics = u
            u = u.x.array

        if len(u) == self.dim:
            _u = u
        elif len(u) == self._ndofs:
            _u = np.zeros(self.dim)
            _u[self.dofs] = u
        else:
            raise ValueError(f"Length of u has to be either {self.dim} (full vector) or {self._ndofs} (only unrestricted dofs), but is {len(u)}")
            
        for bc in self.bcs: bc.set(mu, _u)
        
        if is_fenics:
            u_fenics.x.array[:] = _u
            return u_fenics
        else:
            return _u


def weak_problem(msh: mesh.Mesh, 
                 operator: list[AffineObject, AffineObject, AffineObject], 
                 data: list[AffineList, list[AffineList], list[AffineList]] | list[any,any], 
                 dbdry_U: list = [], 
                 dbdry_V: list = [], 
                 nbdry_U: list = [], 
                 U: fem.FunctionSpace | None = None, 
                 V: fem.FunctionSpace | None = None,
                 A_Space: fem.FunctionSpace | None = None, 
                 B_space: fem.FunctionSpace | None = None, 
                 C_space: fem.FunctionSpace | None = None, 
                 F_space: fem.FunctionSpace | None = None, 
                 H_spaces: list[fem.FunctionSpace] | None = None) -> tuple[AffineLinear, AffineLinear, FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r""" Create a weak variational formulation of an abstract 2nd order operator.
        
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
        l = AffineObject([1.0], [ufl.div(_A*ufl.grad(_u)) + ufl.inner(_b,ufl.grad(_u)) + _c*_u])
    
    elif len(data) == 3:
        l, g, h = data
        
        l = l.apply2data(lambda fq: utils.interpolate_function(F_space, fq))
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
    dbdry_V = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  dbdry_V]

    # create dirichlet boundary conditions
    bcs_U_D = [AffineDirichletBC(U, g, bdry) for g, bdry in zip(g, dbdry_U)]
    bcs_V_D = [AffineDirichletBC(V, 0.0, bdry) for bdry in dbdry_V]
    
    U = FEniCSxSpaceWithDirichletBCs(U, bcs_U_D)
    V = FEniCSxSpaceWithDirichletBCs(V, bcs_V_D, warn=False)

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
        
    l_ufl = l.apply2data(lambda fq: fq * v * ufl.dx) \
            - sum(h.apply2data(lambda hq: hq * v * ds(i)) for i,h in enumerate(h))

    B_ufl = B_ufl.compress()
    l_ufl = l_ufl.compress()

    ########################################
    # extract discrete system

    B_full = AffineLinear(B_ufl.apply2data(lambda Bq: csr_array(fem.assemble_matrix(fem.form(Bq)).to_scipy())))
    l_full = AffineLinear(l_ufl.apply2data(lambda lq:           fem.assemble_vector(fem.form(lq)).array))

    ########################################
    # apply dirichlet boundary conditions

    B = B_full.apply2data(lambda Bq: Bq[V.dofs,:][:,U.dofs])
    l = l_full.apply2data(lambda lq: lq[V.dofs]) \
        - sum(B_full.apply2data(lambda Bq: Bq[V.dofs,:][:,bc.dofs]) @ bc for bc in bcs_U_D)
    
    return B, l, U, V


def thermal_block(Omega: np.ndarray, nh: list[int], nblocks: list[int], data: list[AffineList, AffineList, AffineList]  | list[any,any] = None, plot: bool = False) -> tuple[AffineLinear, AffineLinear, FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r""" Create a weak variational formulation of the thermal block problem.
        
    """
    msh = mesh.create_rectangle(MPI.COMM_WORLD, Omega, nh)
    gdim = msh.geometry.dim
    
    dbdry_U = [lambda x: utils.isclose(x[0:gdim], Omega)]
    dbdry_V = dbdry_U
    
    blocks = [np.linspace(Omega[0][i], Omega[1][i], nblocks[i]+1) for i in range(Omega.shape[1])]
    
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
    b = AffineObject([1.0], [np.zeros(gdim)])
    c = AffineObject([1.0], [0.0])
    
    if data is None:
        data = (AffineObject([1.0], [1.0]), # right-hand side
                AffineObject([1.0], [0.0]), # dirichlet boundary condition
                AffineObject([1.0], [0.0])) # neumann boundary condition
    
    if len(data) == 3:
        data = (data[0], [data[1]], [data[2]])
        
    if plot:
        plotter = pv.Plotter(shape=(nblocks))
        L2 = fem.functionspace(msh, ("DG", 0))
        for idx in product(range(nblocks[0]), range(nblocks[1])):
            plotter.subplot(*idx)
            tmp = fem.Function(utils.change_element(L2, shape=()))
            tmp.interpolate(lambda x: chi(x, idx))
            utils.plot_pyvista(tmp.x.array, utils.change_element(L2, shape=()), f"chi {idx}", plotter)
        plotter.show(interactive_update=True)
    
    return weak_problem(msh, (A,b,c), data, dbdry_U, dbdry_V)