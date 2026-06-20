from collections.abc import Callable

import numpy as np
import scipy as sp

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from ulmRBM.core import Mu, unwrap, Matrix, Vector
from ulmRBM.affine import AffineObject, AffineLinear, AffineFunction, wrap_affinelinear
from ulmRBM.fenicsx import utils, norms, FEniCSxSpaceWithDirichletBCs, SpaceTimeKey, SpaceTimeAffineDirichletBC, SpaceTimeFEniCSxSpaceWithDirichletBCs
from ulmRBM.fenicsx.problems import assemble_matrix, assemble_vector, weak_problem

__all__ = [
    'simple_timestepping_heat',
    'simple_heat',
]

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME

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
    
    
    
 ##########################
 # Muss noch wo anders hin
 ##########################   

def kron(A: Matrix | AffineLinear[Mu, Matrix], B: Matrix | AffineLinear[Mu, Matrix]) -> AffineLinear[Mu, Matrix]:
    A = wrap_affinelinear(A)
    B = wrap_affinelinear(B)
    from ulmRBM.core import KRON_AVAILABLE
    if KRON_AVAILABLE:
        import kron
        _kron = kron.kron
    elif all([sp.sparse.issparse(Aq) for Aq in A.data]) or all([sp.sparse.issparse(Bq) for Bq in B.data]):
        _kron = sp.sparse.kron
    else:
        _kron = np.kron
        
    from ulmRBM.affine import multiply_theta
    
    theta = []
    data  = []
    for (Aq_theta, Aq) in A:
        for (Bq_theta, Bq) in B:
            theta.append(multiply_theta(Aq_theta, Bq_theta))
            data.append(_kron(Aq, Bq))

    return AffineLinear(theta, data)
        
    
    
def _get_points(U: fem.FunctionSpace):
    y = [None]
    def __get_points(x):
        y[0] = x.copy()
        dummy = np.zeros(U.value_shape).reshape(-1,1)
        return np.zeros((dummy.shape[0],x.shape[1]))
    fem.Function(U).interpolate(__get_points)
    return y[0]

def fixe_time(f,t):
    return lambda x: f(t,x)

def interpolate_space_time(U: dict[SpaceTimeKey, fem.FunctionSpace], u):
    
    T = _get_points(U[TIME])[0]
    UT = np.zeros((U[SPACE].dofmap.index_map.size_global, len(T)))
    u_tmp = fem.Function(U[SPACE])
    for k,t in enumerate(T):
        u_tmp.interpolate(fixe_time(u, t))
        UT[:,k] = u_tmp.x.array.copy()
        
    UST = np.zeros((U[SPACE].dofmap.index_map.size_global, U[TIME].dofmap.index_map.size_global))
    u_tmp = fem.Function(U[TIME])
    for j in range(U[SPACE].dofmap.index_map.size_global):
        u_tmp.interpolate(lambda t: UT[j,:])
        UST[j,:] = u_tmp.x.array.copy()
        
    return UST

    # UST = np.zeros((U[SPACE].dofmap.index_map.size_global, U[TIME].dofmap.index_map.size_global))
    # U_fenicsx_time = U[SPACE].dofmap.index_map.size_global * [None]
    # for j in range(U[SPACE].dofmap.index_map.size_global):
    #     U_fenicsx_time[j] = fem.Function(U[TIME])
    #     U_fenicsx_time[j].interpolate(lambda t: UT[j,:])
    #     UST[j,:] = U_fenicsx_time[j].x.array.copy()
    
    # U_fenicsx_space = [fem.Function(U[SPACE]) for _ in range(U[TIME].dofmap.index_map.size_global)]
    # for k in range(U[TIME].dofmap.index_map.size_global):
    #     U_fenicsx_space[k].x.array[:] = UST[:,k]
        
    # return UST, U_fenicsx_space, U_fenicsx_time

###########################
    
def simple_heat(I:list[float],
                Omega:list[float],
                K: int,
                nx:list[int],
                A: list[AffineObject, AffineObject, AffineObject],
                f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1,
                g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0,
                u0: float | AffineObject = 0,
                dbdry: Callable[[np.ndarray[float]], np.ndarray[bool]] = None):
    
    ########################################
    # MESH
    ########################################
    if np.isscalar(nx) or len(nx) == 1:
        if not np.isscalar(nx): nx = nx[0]
        msh = {SPACE: mesh.create_interval(MPI.COMM_WORLD, nx, Omega),
               TIME:  mesh.create_interval(MPI.COMM_WORLD, K, I)}
    elif len(nx) == 2:
        msh = {SPACE: mesh.create_rectangle(MPI.COMM_WORLD, Omega, nx),
               TIME:  mesh.create_interval(MPI.COMM_WORLD, K, I)}
    elif len(nx) == 3:
        msh = {SPACE: mesh.create_box(MPI.COMM_WORLD, Omega, nx),
               TIME:  mesh.create_interval(MPI.COMM_WORLD, K, I)}
    else:
        raise ValueError("Only 1D, 2D and 3D in space are supported.")
    
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
    
    f  = AffineLinear( f.apply2data(lambda fq : interpolate_space_time(F_space, fq)))
    g  = AffineLinear( g.apply2data(lambda gq : interpolate_space_time(U,  gq)))
    u0 = AffineLinear(u0.apply2data(lambda u0q: interpolate_space_time(U,  lambda t, x: u0q(x))))
    
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
    
    dbcs_U = [(TIME,  u0,  lambda t: np.isclose(t[0], I[0])),
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
    B = [{KEY: wrap_affinelinear(assemble_matrix(Bi[KEY])) for KEY in SpaceTimeKey} for Bi in B]
    
    F = {KEY: ufl.inner(ufl.TrialFunction(F_space[KEY]), ufl.TestFunction(V.space[KEY])) * ufl.dx for KEY in SpaceTimeKey}
    F = {KEY: assemble_matrix(F[KEY]) for KEY in SpaceTimeKey}
    f  = (F[SPACE] @ f @ F[TIME].T).apply2data(lambda fq: fq.reshape(-1,))
    
    B, f = apply_dirichletbc(B, f, U, V)
    
    return B, f, U, V


def apply_dirichletbc(B: list[dict[SpaceTimeKey, AffineLinear[Mu, Matrix]]],
                    f: AffineLinear[Mu, Vector],
                    U: FEniCSxSpaceWithDirichletBCs, 
                    V: FEniCSxSpaceWithDirichletBCs) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector]]:
    
    B_F =  [{KEY: Bi[KEY].apply2data(lambda Biq: Biq[V.dofs[KEY]])    for KEY in SpaceTimeKey} for Bi in B]
    B_FF = [{KEY: Bi[KEY].apply2data(lambda Biq: Biq[:, U.dofs[KEY]]) for KEY in SpaceTimeKey} for Bi in B_F]
    
    def apply_bc(bc):        
        B_FD = [{KEY: Bi[KEY].apply2data(lambda Biq: Biq[:, bc.key_dofs]) if bc.key == KEY else Bi[KEY] for KEY in SpaceTimeKey} for Bi in B_F]
        B_FD = sum([kron(Bi[SPACE], Bi[TIME]) for Bi in B_FD])
        return B_FD @ bc
    
    B_FF = sum([kron(Bi[SPACE], Bi[TIME]) for Bi in B_FF])
    B_FD_bcs = sum([apply_bc(bc) for bc in U.bcs])
    f_F  = f.apply2data(lambda fq: fq[V.full_dofs]) - B_FD_bcs
    
    return B_FF.compress(), f_F.compress()