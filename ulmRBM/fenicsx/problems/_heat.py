from collections.abc import Callable

import numpy as np
import scipy as sp

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from ulmRBM.core import Mu, unwrap, Matrix, Vector
from ulmRBM.affine import AffineObject, AffineLinear, AffineFunction, wrap_affinelinear
from ulmRBM.fenicsx import utils, norms, FEniCSxSpaceWithDirichletBCs, SpaceTimeKey, SpaceTimeAffineDirichletBC, SpaceTimeFEniCSxSpaceWithDirichletBCs, interpolate_space_time, apply_dirichletbc_space_time
from ulmRBM.fenicsx.problems import weak_problem

__all__ = [
    'heat_equation',
    'simple_heat',
    'simple_heat_timestepping',
]

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME
    
def heat_equation(msh: dict[SpaceTimeKey, mesh.Mesh],
                A: list[AffineObject, AffineObject, AffineObject],
                f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1,
                g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0,
                u0: float | AffineObject = 0,
                dbdry: Callable[[np.ndarray[float]], np.ndarray[bool]] = None) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], SpaceTimeFEniCSxSpaceWithDirichletBCs, SpaceTimeFEniCSxSpaceWithDirichletBCs]:
    r"""Parametric heat equation.
    
    For some time interval :math:`I` and a spatial domain :math:`\Omega \subset\mathbb{R}^d`, consider the parametric heat equation
    
    .. math::
        \begin{aligned}
        u_t(t,x) + A_\mu(x) u(t,x) &= f_\mu(t,x) \quad\text{for all $(t,x)\in I\times\Omega$}, \\
        u(0,x) &= u0_\mu(x) \quad\text{for all $x\in\Omega$}, \\
        u(t,x) &= g_\mu(t,x) \quad\text{for all $(t,x)\in I\times\partial\Omega$},
        \end{aligned}
        
    with
    
    .. math:: 
        A_\mu(x) u(x) := \nabla_x \cdot (\uline{A}_\mu(x) \nabla_x u(x)) + \uline{b}_\mu(x) \cdot \nabla_x u(x) + \uline{c}_\mu(x) u(x).
        
    Then, after some homogenization, for :math:`U := H^1_{0,}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))` and :math:`V := L^2(I;H^1_0(\Omega))`, the weak formulation of the heat equation reads: Find :math:`u_\mu \in U` such that
    
    .. math::
        b_\mu(u_\mu, v) = f_\mu(v) \quad\text{for all } v \in V,
        
    with the bilinear form :math:`b_\mu:U\times V \to \mathbb{R}` and linear form :math:`f_\mu:V \to \mathbb{R}` given by
    
    .. math::
        b_\mu(u,v) := (\partial_t u, v)_{L^2(I\times\Omega)} - (\uline{A}_\mu \nabla_x u, \nabla_x v)_{L^2(I\times\Omega)} + (\uline{b}_\mu \cdot \nabla_x u, v)_{L^2(I\times\Omega)} + (\uline{c}_\mu u, v)_{L^2(I\times\Omega)},\\
        f_\mu(v) := (f_\mu, v)_{L^2(I\times\Omega)}.
    
    Args:
        msh:
            Dictionary containing the spatial and temporal mesh.
        A:
            List of three affine objects ``(A_,b_,c_)`` corresponding to :math:`\uline{A}_\mu,\uline{b}_\mu,\uline{c}_\mu`. Each of these affine objects should be compatible with `utils.interpolate_function`.
        f:
            (Parametric) right-hand side. A scalar or an (affine) function with signature ``f(t,x)`` or ``f(mu)(t,x)``.
        g:
            (Parametric) Dirichlet boundary condition. A scalar or an (affine) function with signature ``g(t,x)`` or ``g(mu)(t,x)``.
        u0:
            (Parametric) Initial condition. A scalar or an (affine) function with signature ``u0(x)`` or ``u0(mu)(x)``.
        dbdry:
            Function that takes as input points on the spatial domain boundary and returns a boolean array indicating which of these points are on the spatial part of the Dirichlet boundary, on which the Dirichlet boundary condition ``g`` is applied. If None, the Dirichlet boundary condition will be applied on the entire spatial boundary :math:`\partial\Omega`.
            
    Returns
    -------
    B :
        Discretization of the bilinear form :math:`b_\mu` as an affine matrix.
    f :
        Discretization of the linear form :math:`f_\mu` as an affine vector.
    U :
        Function space :math:`U := H^1_{0,}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))` containing the initial condition :math:`u0_\mu` and Dirichlet boundary condition :math:`g_\mu`.
    V :
        Function space :math:`V := L^2(I;H^1_0(\Omega))`.
    """
    
    gdim = {KEY: msh[KEY].geometry.dim for KEY in SpaceTimeKey}
    tdim = {KEY: msh[KEY].topology.dim for KEY in SpaceTimeKey}
    
    for KEY in SpaceTimeKey: 
        msh[KEY].topology.create_connectivity(tdim[KEY]-1, tdim[KEY])
        
    ########################################
    # SPACES
    ########################################
    
    L2 = {KEY: fem.functionspace(msh[KEY], ("DG", 0))       for KEY in SpaceTimeKey}
    H1 = {KEY: fem.functionspace(msh[KEY], ("Lagrange", 1)) for KEY in SpaceTimeKey}
    
    U = {SPACE: H1[SPACE], TIME: H1[TIME]}
    V = {SPACE: H1[SPACE], TIME: L2[TIME]}
    
    F_space = L2
    A_space = utils.change_element(L2[SPACE], shape=(gdim[SPACE], gdim[SPACE]))
    b_space = utils.change_element(L2[SPACE], shape=(gdim[SPACE],))
    c_space = L2[SPACE]
    
    ########################################
    # DISCRETIZ OPERATOR AND DATA
    ########################################
    
    if not isinstance(f, AffineObject): f = AffineObject([1.0], [f])
    if not isinstance(g, AffineObject): g = AffineObject([1.0], [g])
    if not isinstance(u0, AffineObject): u0 = AffineObject([1.0], [u0])
    
    f  = AffineLinear( f.apply2data(lambda fq : interpolate_space_time(F_space, fq)))
    g  = AffineLinear( g.apply2data(lambda gq : interpolate_space_time(U,  gq)))
    u0 = AffineLinear(u0.apply2data(lambda u0q: interpolate_space_time(U,  u0q if np.isscalar(u0q) else lambda t, x: u0q(x))))
    
    A_, b_, c_ = A
    A_ = A_.apply2data(lambda Aq: utils.interpolate_function(A_space, Aq))
    b_ = b_.apply2data(lambda bq: utils.interpolate_function(b_space, bq))
    c_ = c_.apply2data(lambda cq: utils.interpolate_function(c_space, cq))
    
    ########################################
    # BOUNDARY CONDITIONS
    ########################################
    
    if dbdry is None:
        dbdry = lambda x: np.ones(x.shape[1], dtype=bool)
    
    def get_entities(KEY, dbdry):
        return mesh.locate_entities_boundary(msh[KEY], tdim[KEY]-1, dbdry)
    
    dbcs_U = [(TIME,  u0,  lambda t: np.isclose(t[0], msh[TIME].geometry.x[:,0].min())),
              (SPACE, g,   dbdry)]
    dbcs_V = [(SPACE, 0.0, lambda x: np.ones(x.shape[1], dtype=bool))]
    
    dbcs_U = [SpaceTimeAffineDirichletBC(KEY, U, g, get_entities(KEY, dbdry)) for KEY, g, dbdry in dbcs_U]
    dbcs_V = [SpaceTimeAffineDirichletBC(KEY, V, g, get_entities(KEY, dbdry)) for KEY, g, dbdry in dbcs_V]
    
    U = SpaceTimeFEniCSxSpaceWithDirichletBCs(U, dbcs_U)
    V = SpaceTimeFEniCSxSpaceWithDirichletBCs(V, dbcs_V, warn=False)
    
    ########################################
    # VARIATIONAL FORMULATION
    ########################################

    u = {KEY: ufl.TrialFunction(U.space[KEY]) for KEY in SpaceTimeKey}
    v = {KEY: ufl.TestFunction(V.space[KEY])  for KEY in SpaceTimeKey}
    
    M = {KEY:  ufl.inner(u[KEY], v[KEY]) * ufl.dx for KEY in SpaceTimeKey}
    N = {TIME: ufl.inner(ufl.Dx(u[TIME], 0), v[TIME]) * ufl.dx}
    A = {SPACE: - A_.apply2data(lambda Aq: ufl.inner(Aq * ufl.grad(u[SPACE]), ufl.grad(v[SPACE])) * ufl.dx) \
                + b_.apply2data(lambda bq: ufl.inner(bq, ufl.grad(u[SPACE])) * v[SPACE] * ufl.dx) \
                + c_.apply2data(lambda cq: cq * u[SPACE] * v[SPACE] * ufl.dx)}
    
    B = [{SPACE: M[SPACE], TIME: N[TIME]}, 
         {SPACE: A[SPACE], TIME: M[TIME]}]
    B = [{KEY: wrap_affinelinear(utils.assemble_matrix(Bi[KEY])) for KEY in SpaceTimeKey} for Bi in B]
    
    F = {KEY: ufl.inner(ufl.TrialFunction(F_space[KEY]), ufl.TestFunction(V.space[KEY])) * ufl.dx for KEY in SpaceTimeKey}
    F = {KEY: utils.assemble_matrix(F[KEY]) for KEY in SpaceTimeKey}
    f  = (F[SPACE] @ f @ F[TIME].T).apply2data(lambda fq: fq.reshape(-1,))
    
    B, f = apply_dirichletbc_space_time(B, f, U, V)
    
    return B, f, U, V


def simple_heat(K: int = 10,
                nx: int | list[int] = 10,
                f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1,
                g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0,
                u0: float | AffineObject = 0) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], SpaceTimeFEniCSxSpaceWithDirichletBCs, SpaceTimeFEniCSxSpaceWithDirichletBCs]:
    r"""Simple parametric heat problem operators.
    
    See `heat_equation`, with :math:`I=(0,1)`, :math:`\Omega=(0,1)^d`, :math:`A_\mu(x) = -\mu \Delta_x` and :math:`\uline{b}_\mu(x) = 0`, :math:`\uline{c}_\mu(x) = 0`
    """
    
    if np.isscalar(nx) or len(nx) == 1:
        if not np.isscalar(nx): nx = nx[0]
        msh = {SPACE: mesh.create_interval(MPI.COMM_WORLD, nx, [0,1]),
               TIME:  mesh.create_interval(MPI.COMM_WORLD, K, [0,1])}
    elif len(nx) == 2:
        msh = {SPACE: mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [1,1]], nx),
               TIME:  mesh.create_interval(MPI.COMM_WORLD, K, [0,1])}
    elif len(nx) == 3:
        msh = {SPACE: mesh.create_box(MPI.COMM_WORLD, [[0,0,0], [1,1,1]], nx),
               TIME:  mesh.create_interval(MPI.COMM_WORLD, K, [0,1])}
    else:
        raise ValueError("Only 1D, 2D and 3D in space are supported.")

    gdim = {SPACE: msh[SPACE].geometry.dim}

    # for x dependent diffusion, e.g.: AffineFunction([lambda mu: mu], [lambda x: -np.eye(gdim[SPACE]).reshape(-1,1) * np.ones(x.shape[1])])
    A = AffineObject([lambda mu: mu], [-np.eye(gdim[SPACE])])
    b = AffineObject([0.0], [np.ones(gdim[SPACE])])
    c = AffineObject([0.0], [1.0])

    return heat_equation(msh, (A, b, c), f, g, u0)


def simple_heat_timestepping(nx:list[int]=[10],
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
        msh = mesh.create_interval(MPI.COMM_WORLD, nx, [0,1])
    elif len(nx) == 2:
        msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [1,1]], nx)
    elif len(nx) == 3:
        msh = mesh.create_box(MPI.COMM_WORLD, [[0,0,0], [1,1,1]], nx)
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
    A = utils.assemble_matrix(A)
    A = A.apply2data(lambda Aq: Aq[U.dofs,:][:,U.dofs])
    M = unwrap(norms.l2(U)._M)
    
    v = ufl.TestFunction(U.space)
    def eval_rhs(f):
        def at_time(t):
            ft = utils.interpolate_function(U.space, f(t))
            ft = ft * v * ufl.dx
            return utils.assemble_vector(ft)[U.dofs]
        return at_time
    f = f.apply2data(eval_rhs)
    
    u0 = u0.apply2data(lambda u0q: utils.interpolate_function(U.space, u0q))
    u0 = u0.apply2data(lambda u0q: u0q.value * np.ones(sum(U.dofs)) if isinstance(u0q, fem.Constant) else u0q.x.array[U.dofs])
    # u0 = u0.apply2data(lambda u0q: utils.interpolate_function(U.space, u0q).x.array[U.dofs])
    
    return A, M, f, u0, U