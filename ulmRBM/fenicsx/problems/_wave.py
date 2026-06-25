import numpy as np

from mpi4py import MPI
from dolfinx import mesh
import ufl

from ulmRBM.core import Mu
from ulmRBM.affine import AffineObject
from ulmRBM.fenicsx import utils, FEniCSxSpaceWithDirichletBCs, assemble_system
from ulmRBM.fenicsx.problems import weak_problem


__all__ = [
    'simple_wave',
    'simple_wave_with_output'
]

def simple_wave(K:int=10, 
                nx:list[int]=[10], 
                f: float | AffineObject = 1, 
                g: float | AffineObject = 0, 
                u0: float | AffineObject = 0, 
                u1: float | AffineObject = 0, 
                exact_sol = None, exact_mu: float = 1.0) -> tuple[AffineObject[Mu, ufl.Form], AffineObject[Mu, ufl.Form], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Parametric wave problem.
    
    Let :math:`\mu>0` be the wave speed. For :math:`I=(0,T)` and :math:`\Omega=\subset \mathbb{R}^d`, :math:`d\in\{1,2\}` consider the wave equation
    
    .. math::
        u_{tt} - \mu \Delta_x u = f_\mu \quad\text{in } I\times\Omega, \\
        u(0) = u0_\mu\quad\text{in } \Omega,\\
        u_t(0) = u1_\mu \quad\text{in } \Omega, \\
        u = g_\mu \quad\text{on } I\times\partial\Omega,
    
    and its resulting weak formulation to find :math:`u_\mu \in U := \{u\in H^1(I\times\Omega) : u(0)=u0_\mu,\, u\vert_{I\times\partial\Omega}=g_\mu\}` such that 
    
    .. math:: 
        b_\mu(u_\mu, v) = f_\mu(v) \quad\text{for all } v \in V := \{v\in H^1(I\times\Omega) : v(T)=0,\, v\vert_{I\times\partial\Omega}=0\},
        
    with the bilinear form :math:`b_\mu:U\times V \to \mathbb{R}` and linear form :math:`f_\mu:V \to \mathbb{R}` given by
    
    .. math::
        b_\mu(u,v) := -(\partial_t u, \partial_t v)_{L^2(I\times\Omega)}
        + \mu (\nabla_x u, \nabla_x v)_{L^2(I\times\Omega)},\\
        f_\mu(v) := (f_\mu, v)_{L^2(I\times\Omega)} + (u1_\mu, v(0))_{L^2(\Omega)}

    Args:
        K: 
            Number of mesh cells in the time direction.
        nx: 
            Number of mesh cells in each spatial direction, where ``len(nx)`` is the spatial dimension :math:`d`.
        f:
            (Parametric) right-hand side. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        g:
            (Parametric) Dirichlet boundary condition on :math:`I\times\partial\Omega`. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        u0:
            (Parametric) Initial condition for :math:`u(0)`. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        u1:
            (Parametric) Initial velocity for :math:`u_t(0)`. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        exact_sol:
            Exact solution of the problem, which can be parameter-dependent. If given, :math:`f,u0,u1,g` are ignored and calculated from the exact solution instead. Compatible with `utils.interpolate_function`.
        exact_mu:
            Parameter value at which the exact solution is given.
    """
    
    if np.isscalar(nx): nx = [nx]
    if len(nx) == 1:
        Omega = [0,1]
    elif len(nx) == 2:
        Omega = [[0,0], [1,1]]
    else:
        raise ValueError("Only 1D, 2D in space are supported.")
    
    I = np.asarray([0,1]).reshape(-1,1)
    Omega = np.asarray(Omega)
    if Omega.ndim == 1:
        Omega = Omega.reshape(-1,1)
        
    min_nt = np.inf
    for i in range(Omega.shape[1]):
        min_nt = min(min_nt, np.ceil((I[1]-I[0])/((Omega[1][i]-Omega[0][i])/(nx[i-1]+1))))
    print(f"To ensure the CFL condition, K should be at least ``mu * {min_nt[0]:.2f}``.")
    
    n = [K] + nx
    Q = np.hstack([I, Omega])
    
    if len(n) == 2:
        msh = mesh.create_rectangle(MPI.COMM_WORLD, Q, n)
    elif len(n) == 3:
        msh = mesh.create_box(MPI.COMM_WORLD, Q, n)
    else:
        raise ValueError("Only 1D, 2D in space are supported.")
    
    gdim = msh.geometry.dim
    
    # dirichlet boundary conditions
    dbdry = [lambda tx: utils.isclose(tx[0], I[0]),                       # {0} x Omega
             lambda tx: utils.isclose(tx[1:gdim], Omega) & ~dbdry[0](tx)] # I x Gamma
    # neumann boundary conditions               
    nbdry = [dbdry[0]]                                                    # {0} x Omega
    
    if exact_sol is not None:
        data = (exact_sol, exact_mu)
    else:
        data = (f, [u0, g], [-u1])
    
    A = AffineObject([1.0], [np.diag([1.0] + (gdim-1)*[0.0])])  # u_tt
    A += [(lambda mu: -mu,   np.diag([0.0] + (gdim-1)*[1.0]))]  # - mu * Delta_x u
    b = AffineObject([0.0], [np.ones(gdim)])
    c = AffineObject([0.0], [1.0])
    
    B, f, U, V = weak_problem(msh, (A,b,c), data, dbdry, nbdry)
    B, f = assemble_system(B, f, U, V)
    
    return B, f, U, V

def simple_wave_with_output(K:int=10, 
                            nx:list[int]=[10], 
                            f: float | AffineObject = 1, 
                            g: float | AffineObject = 0, 
                            u0: float | AffineObject = 0, 
                            u1: float | AffineObject = 0, 
                            exact_sol = None, exact_mu: float = 1.0) -> tuple[AffineObject[Mu, ufl.Form], AffineObject[Mu, ufl.Form], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Parametric wave problem with output.
    
    Let :math:`\mu>0` be the wave speed. For :math:`I=(0,T)` and :math:`\Omega=\subset \mathbb{R}^d`, :math:`d\in\{1,2\}` consider the wave equation
    
    .. math::
        u_{tt} - \mu \Delta_x u = f_\mu \quad\text{in } I\times\Omega, \\
        u(0) = u0_\mu\quad\text{in } \Omega,\\
        u_t(0) = u1_\mu \quad\text{in } \Omega, \\
        u = g_\mu \quad\text{on } I\times\partial\Omega,
    
    and its resulting weak formulation to find :math:`u_\mu \in U := \{u\in H^1(I\times\Omega) : u(0)=u0_\mu,\, u\vert_{I\times\partial\Omega}=g_\mu\}` such that 
    
    .. math:: 
        b_\mu(u_\mu, v) = f_\mu(v) \quad\text{for all } v \in V := \{v\in H^1(I\times\Omega) : v(T)=0,\, v\vert_{I\times\partial\Omega}=0\},
        
    with the bilinear form :math:`b_\mu:U\times V \to \mathbb{R}` and linear form :math:`f_\mu:V \to \mathbb{R}` given by
    
    .. math::
        b_\mu(u,v) := -(\partial_t u, \partial_t v)_{L^2(I\times\Omega)}
        + \mu (\nabla_x u, \nabla_x v)_{L^2(I\times\Omega)},\\
        f_\mu(v) := (f_\mu, v)_{L^2(I\times\Omega)} + (u1_\mu, v(0))_{L^2(\Omega)}

    The output vector l is defined such that the output s is the integral of the state at terminal time T

    .. math::
        s(mu) := lu := \int_\Omega u(T) dx

    Args:
        K: 
            Number of mesh cells in the time direction.
        nx: 
            Number of mesh cells in each spatial direction, where ``len(nx)`` is the spatial dimension :math:`d`.
        f:
            (Parametric) right-hand side. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        g:
            (Parametric) Dirichlet boundary condition on :math:`I\times\partial\Omega`. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        u0:
            (Parametric) Initial condition for :math:`u(0)`. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        u1:
            (Parametric) Initial velocity for :math:`u_t(0)`. Any object compatible with `utils.interpolate_function`, or an affine decomposition compatible with `utils.interpolate_function`.
        exact_sol:
            Exact solution of the problem, which can be parameter-dependent. If given, :math:`f,u0,u1,g` are ignored and calculated from the exact solution instead. Compatible with `utils.interpolate_function`.
        exact_mu:
            Parameter value at which the exact solution is given.
    """
    
    if np.isscalar(nx): nx = [nx]
    if len(nx) == 1:
        Omega = [0,1]
    elif len(nx) == 2:
        Omega = [[0,0], [1,1]]
    else:
        raise ValueError("Only 1D, 2D in space are supported.")
    
    I = np.asarray([0,1]).reshape(-1,1)
    Omega = np.asarray(Omega)
    if Omega.ndim == 1:
        Omega = Omega.reshape(-1,1)
        
    min_nt = np.inf
    for i in range(Omega.shape[1]):
        min_nt = min(min_nt, np.ceil((I[1]-I[0])/((Omega[1][i]-Omega[0][i])/(nx[i-1]+1))))
    print(f"To ensure the CFL condition, K should be at least ``mu * {min_nt[0]:.2f}``.")
    
    n = [K] + nx
    Q = np.hstack([I, Omega])
    
    if len(n) == 2:
        msh = mesh.create_rectangle(MPI.COMM_WORLD, Q, n)
    elif len(n) == 3:
        msh = mesh.create_box(MPI.COMM_WORLD, Q, n)
    else:
        raise ValueError("Only 1D, 2D in space are supported.")
    
    gdim = msh.geometry.dim
    
    # dirichlet boundary conditions
    dbdry = [lambda tx: utils.isclose(tx[0], I[0]),                       # {0} x Omega
             lambda tx: utils.isclose(tx[1:gdim], Omega) & ~dbdry[0](tx)] # I x Gamma
    # neumann boundary conditions               
    nbdry = [dbdry[0]]                                                    # {0} x Omega
    
    if exact_sol is not None:
        data = (exact_sol, exact_mu)
    else:
        data = (f, [u0, g], [-u1])
    
    A = AffineObject([1.0], [np.diag([1.0] + (gdim-1)*[0.0])])  # u_tt
    A += [(lambda mu: -mu,   np.diag([0.0] + (gdim-1)*[1.0]))]  # - mu * Delta_x u
    b = AffineObject([0.0], [np.ones(gdim)])
    c = AffineObject([0.0], [1.0])
    
    B, f, U, V = weak_problem(msh, (A,b,c), data, dbdry, nbdry)

    # Output
    
    tdim = msh.topology.dim
    terminal_bdry = [lambda tx: utils.isclose(tx[0], I[1])]
    terminal_bdry = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  terminal_bdry]
    ds = utils.create_measure("ds", msh, tdim-1, terminal_bdry)
    u = ufl.TrialFunction(U.space)
    l = u * ds

    B, f, l, s0 = assemble_system(B, f, U, V, l)
    
    return B, f, U, V, l, s0