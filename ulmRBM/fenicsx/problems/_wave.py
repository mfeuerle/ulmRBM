import numpy as np

from mpi4py import MPI
from dolfinx import mesh
import ufl

from ulmRBM.affine import AffineList, AffineObject
from ulmRBM.fenicsx import utils, FEniCSxSpaceWithDirichletBCs
from ulmRBM.fenicsx.problems import weak_problem


__all__ = [
    'simple_wave',
]

def simple_wave(I:list[float]=[0,1], Omega:list[float]=[0,1], 
                nt:int=10, nx:list[int]=[10], 
                f: float | AffineList = 1, 
                u0: float | AffineList = 0, 
                u1: float | AffineList = 0, 
                g: float | AffineList = 0, 
                exact_sol = None, exact_mu: float = 1.0) -> tuple[AffineList[ufl.Form], AffineList[ufl.Form], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
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
        I:
            Time interval of the problem.
        Omega:
            Spatial domain of the problem, which is assumed to be a box domain with ``Omega[0]`` being the lower left corner and ``Omega[1]`` being the upper right corner, where ``len(Omega[0])`` is the spatial dimension :math:`d`.
        nt: 
            Number of mesh cells in the time direction.
        nx: 
            Number of mesh cells in each spatial direction, where ``len(nx)`` is the spatial dimension :math:`d`.
        f:
            (Parametric) right-hand side. Either a scalar, or an affine decomposition compatible with `utils.interpolate_function`.
        u0:
            (Parametric) Initial condition for :math:`u(0)`. Either a scalar, or an affine decomposition compatible with `utils.interpolate_function`.
        u1:
            (Parametric) Initial velocity for :math:`u_t(0)`. Either a scalar, or an affine decomposition compatible with `utils.interpolate_function`.
        g:
            (Parametric) Dirichlet boundary condition on :math:`I\times\partial\Omega`. Either a scalar, or an affine decomposition compatible with `utils.interpolate_function`.
        exact_sol:
            Exact solution of the problem, which can be parameter-dependent. If given, :math:`f,u0,u1,g` are ignored and calculated from the exact solution instead. Compatible with `utils.interpolate_function`.
        exact_mu:
            Parameter value at which the exact solution is given.
            
    Returns
    -------
    See `weak_problem` for details on the return values.
    """
    
    I = np.asarray(I).reshape(-1,1)
    Omega = np.asarray(Omega)
    if Omega.ndim == 1:
        Omega = Omega.reshape(-1,1)
        
    min_nt = np.inf
    for i in range(Omega.shape[1]):
        min_nt = min(min_nt, np.ceil((I[1]-I[0])/((Omega[1][i]-Omega[0][i])/(nx[i-1]+1))))
    print(f"To ensure the CFL condition, nt should be at least ``mu * {min_nt[0]:.2f}``.")
    
    n = [nt] + nx
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
        if not isinstance(f, AffineList): 
            f = AffineObject([1.0], [f])  # right-hand side
        if not isinstance(u0, AffineList): 
            u0 = AffineObject([1.0], [u0])  # initial condition u(0)
        if not isinstance(u1, AffineList): 
            u1 = AffineObject([1.0], [u1])  # initial velocity u_t(0)
        if not isinstance(g, AffineList): 
            g = AffineObject([1.0], [g])  # boundary condition on IxGamma
        data = (f, [u0, g], [-u1])
    
    A = AffineObject([1.0], [np.diag([1.0] + (gdim-1)*[0.0])])  # u_tt
    A += [(lambda mu: -mu,   np.diag([0.0] + (gdim-1)*[1.0]))]  # - mu * Delta_x u
    b = AffineObject([0.0], [np.ones(gdim)])
    c = AffineObject([0.0], [1.0])
    
    return weak_problem(msh, (A,b,c), data, dbdry, nbdry)


# def wave_kron()