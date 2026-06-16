from collections.abc import Callable

import numpy as np

from mpi4py import MPI
from dolfinx import mesh
import ufl

from ulmRBM.core import Mu, unwrap, Matrix, Vector
from ulmRBM.affine import AffineObject, AffineLinear, AffineFunction
from ulmRBM.fenicsx import utils, norms, FEniCSxSpaceWithDirichletBCs
from ulmRBM.fenicsx.problems import assemble_matrix, assemble_vector, weak_problem


__all__ = [
    'simple_timestepping_heat',
]

def simple_timestepping_heat(Omega:list[float]=[0,1],
                             nx:list[int]=[10],
                             f: float | Callable[[float], any] | AffineFunction[Mu] = 1,
                             u0: float | AffineObject = 0) -> tuple[AffineLinear[Mu, Matrix], AffineLinear[Mu, Matrix], AffineFunction[Mu], AffineLinear[Mu, Vector], FEniCSxSpaceWithDirichletBCs]:
    r"""Simple parametric heat problem operators for time-stepping.
    
    For some time interval :math:`I` and a spatial domain :math:`\Omega \subset\mathbb{R}^d`, consider the simple parametric heat equation
    
    .. math::
        \begin{aligned}
        u_t(t,x) - \mu \Delta_x u(t,x) &= f_\mu(t,x) \quad\text{for all $(t,x)\in I\times\Omega$}, \\
        u(0,x) &= u0_\mu(x) \quad\text{for all $x\in\Omega$}, \\
        u(t,x) &= 0 \quad\text{for all $(t,x)\in I\times\partial\Omega$}.
        \end{aligned}
    
    Using a semi variational formulation in space, this leads to the linear time-invariant problem of finding :math:`u \in C^1(I;W)` such that
    
    .. math::
        \begin{aligned}
        (u_t(t), w)_{L^2(\Omega)} + \mu (\nabla_x u(t), \nabla_x w)_{L^2(\Omega)} &= (f_\mu(t), w)_{L^2(\Omega)} \quad\text{for all $w \in W$ and all $t\in I$},\\
        u(0) &= u0_\mu,
        \end{aligned}
    
    with :math:`W := H^1_0(\Omega)`.
    
    Args:
        Omega:
            Spatial domain of the problem, compatible with `dolfinx.mesh.create_interval`, `dolfinx.mesh.create_rectangle`, or `dolfinx.mesh.create_box`, depending on the length of ``nx``.
        nx:
            Number of mesh cells in each spatial direction, where ``len(nx)`` is the spatial dimension :math:`d`.
        f:
            (Parametric) right-hand side. Either a scalar, a callable such that ``f(t)`` is compatible with `utils.interpolate_function`, or a parametric `AffineFunction` such that ``f(mu)(t)`` is compatible with `utils.interpolate_function`.
        u0:
            Initial condition. Either a scalar, or a parametric `AffineObject` compatible with `utils.interpolate_function`.
            
    Returns
    -------
    A :
        Stiffness matrix corresponding to the bilinear form :math:`(\nabla_x u, \nabla_x w)_{L^2(\Omega)}`.
    M :
        Mass matrix corresponding to the bilinear form :math:`(u, w)_{L^2(\Omega)}`.
    f :
        Parametric right-hand side vector corresponding to the linear form :math:`(f_\mu(t), w)_{L^2(\Omega)}`. Signature ``f(mu)(t)``.
    u0 :
        Initial condition vector corresponding to :math:`u0_\mu`.
    W :
        Function space :math:`W := H^1_0(\Omega)`.
    """
        
    if not isinstance(f, AffineFunction):
        if np.isscalar(f):
            f_val = f
            f = lambda t: f_val
        f = AffineFunction([1.0], [f])
        
    if not isinstance(u0, AffineObject):
        u0 = AffineObject([1.0], [u0])
       
    if np.isscalar(nx) or len(nx) == 1:
        if not np.isscalar(nx): nx = nx[0]
        msh = mesh.create_interval(MPI.COMM_WORLD, nx, Omega)
    elif len(nx) == 2:
        msh = mesh.create_rectangle(MPI.COMM_WORLD, Omega, nx)
    elif len(nx) == 3:
        msh = mesh.create_box(MPI.COMM_WORLD, Omega, nx)
    else:
        raise ValueError("Only 1D, 2D and 3D in space are supported.")
    
    gdim = msh.geometry.dim
    
    dbdry = [lambda x: np.ones(x.shape[1], dtype=bool)]
    
    f_dummy = AffineObject([1.0], [1.0])
    g = [AffineObject([0.0], [0.0])]
    h = []
    
    A = AffineObject([lambda mu: mu], [np.diag(gdim*[1.0])])
    b = AffineObject([0.0], [np.ones(gdim)])
    c = AffineObject([0.0], [1.0])
    
    A, _, U, _ = weak_problem(msh, (A,b,c), (f_dummy, g, h), dbdry)
    A = assemble_matrix(A)
    A = A.apply2data(lambda Aq: Aq[U.dofs,:][:,U.dofs])
    M = unwrap(norms.l2(U)._M)
    
    v = ufl.TestFunction(U.space)
    def eval_rhs(f):
        def at_time(t):
            ft = utils.interpolate_function(U.space, f(t))
            ft = ft * v * ufl.dx
            return assemble_vector(ft)[U.dofs]
        return at_time
    f = f.apply2data(eval_rhs)
    
    u0 = u0.apply2data(lambda u0q: utils.interpolate_function(U.space, u0q).x.array[U.dofs])
    
    return A, M, f, u0, U
    