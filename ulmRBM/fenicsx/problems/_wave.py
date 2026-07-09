from collections.abc import Callable

import numpy as np

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from ulmRBM.core import Mu, Matrix, Vector, Number
from ulmRBM.affine import AffineObject, AffineLinear, AffineFunction, affine_kron
from ulmRBM.fenicsx import utils, FEniCSxSpaceWithDirichletBCs, SpaceTimeKey, SpaceTimeAffineDirichletBC, SpaceTimeFEniCSxSpaceWithDirichletBCs, interpolate_space_time, apply_dirichletbc_space_time, assemble_system
from ulmRBM.fenicsx.problems import weak_problem, hilbert_transform


__all__ = [
    'simple_wave',
    'wave_equation_structured',
    'simple_wave_structured',
    'wave_equation_hilbert',
    'simple_wave_hilbert'
]

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME


def simple_wave(K:int=10, 
                nx:list[int]=[10], 
                f: float | AffineObject = 1, 
                g: float | AffineObject = 0, 
                u0: float | AffineObject = 0, 
                u1: float | AffineObject = 0, 
                exact_sol = None, exact_mu: float = 1.0,
                output_mode: int = 0) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], SpaceTimeFEniCSxSpaceWithDirichletBCs, SpaceTimeFEniCSxSpaceWithDirichletBCs, AffineLinear[Mu,Vector], AffineLinear[Mu,Number],]:
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

    For ``outputmode = 1`` the output is defined as the integral of the state at terminal time T, i.e.

    .. math::
        s_\mu(u_\mu(t, x)) = \int_\Omega u(T, x) \, \text{d}x

    Parameters
    -----------
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
    output_mode:
        Choose output. 0 for no output. 1 for integral over the state at the terminal time point.
    
    Returns
    -------
    See `weak_problem` for details on the return values.
    l :
        Output functional for :py:attr:`ulmRBM.fom.FOM.l`.
    s0 :
        Contribution of the Dirichlet boundaries to the output.
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

    if output_mode==0:
        B, f = assemble_system(B, f, U, V)
        return B, f, U, V

    # Output computation

    if output_mode==1:
        tdim = msh.topology.dim
        terminal_bdry = [lambda tx: utils.isclose(tx[0], I[1])]
        terminal_bdry = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  terminal_bdry]
        ds = utils.create_measure("ds", msh, tdim-1, terminal_bdry)
        u = ufl.TrialFunction(U.space)
        l = u * ds
    else:
        raise ValueError('Unknown output case for simple_wave')

    B, f, l, s0 = assemble_system(B, f, U, V, l)
    
    return B, f, U, V, l, s0



def wave_equation_structured(msh: dict[SpaceTimeKey, mesh.Mesh],
                A: list[AffineObject, AffineObject, AffineObject],
                f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1,
                g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0,
                u0: float | AffineFunction[Mu] = 0,
                u1: float | AffineFunction[Mu] = 0) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], SpaceTimeFEniCSxSpaceWithDirichletBCs, SpaceTimeFEniCSxSpaceWithDirichletBCs]:
    r"""Parametric wave equation with a structured tensor product discretization.
    
    For some time interval :math:`I` and a spatial domain :math:`\Omega \subset\mathbb{R}^d`, consider the parametric wave equation
    
    .. math::
        \begin{aligned}
        u_{tt}(t,x) + A_\mu(x) u(t,x) &= f_\mu(t,x) \quad\text{for all $(t,x)\in I\times\Omega$}, \\
        u(0,x) &= u0_\mu(x) \quad\text{for all $x\in\Omega$}, \\
        u_t(0,x) &= u1_\mu(x) \quad\text{for all $x\in\Omega$}, \\
        u(t,x) &= g_\mu(t,x) \quad\text{for all $(t,x)\in I\times\partial\Omega$},
        \end{aligned}
        
    with
    
    .. math:: 
        A_\mu(x) u(x) := \nabla_x \cdot (\underline{A}_\mu(x) \nabla_x u(x)) + \underline{b}_\mu(x) \cdot \nabla_x u(x) + \underline{c}_\mu(x) u(x).
        
    Then, after some homogenization, for :math:`U := H^1_{0,}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))` and :math:`V := H^1_{,0}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))`, the weak formulation of the wave equation reads: Find :math:`u_\mu \in U` such that
    
    .. math::
        b_\mu(u_\mu, v) = f_\mu(v) \quad\text{for all } v \in V,
        
    with the bilinear form :math:`b_\mu:U\times V \to \mathbb{R}` and linear form :math:`f_\mu:V \to \mathbb{R}` given by
    
    .. math::
        b_\mu(u,v) := -(\partial_{t} u, \partial_t v)_{L^2(I\times\Omega)} - (\underline{A}_\mu \nabla_x u, \nabla_x v)_{L^2(I\times\Omega)} + (\underline{b}_\mu \cdot \nabla_x u, v)_{L^2(I\times\Omega)} + (\underline{c}_\mu u, v)_{L^2(I\times\Omega)},\\
        f_\mu(v) := (f_\mu, v)_{L^2(I\times\Omega)} + \left(u_1, v(0)\right)_{L^2(\Gamma_N^i)}.
        
    .. note::
        For a unstructured discretization, one can use `weak_problem`, see e.g. `simple_wave`.
        
    .. note::
        This discretization needs to satisfy a CFL type condition, i.e. the linear system might not be solvable for arbitrary space-time meshes.
    
    Args:
        msh:
            Dictionary containing the spatial and temporal mesh.
        A:
            List of three affine objects ``(A_,b_,c_)`` corresponding to :math:`\underline{A}_\mu,\underline{b}_\mu,\underline{c}_\mu`. Each of these affine objects should be compatible with `utils.interpolate_function`.
        f:
            (Parametric) right-hand side. A scalar or an (affine) function with signature ``f(t,x)`` or ``f(mu)(t,x)``.
        g:
            (Parametric) Dirichlet boundary condition. A scalar or an (affine) function with signature ``g(t,x)`` or ``g(mu)(t,x)``.
        u0:
            (Parametric) Initial condition. A scalar or an (affine) function with signature ``u0(x)`` or ``u0(mu)(x)``.
        u1:
            (Parametric) Initial velocity. A scalar or an (affine) function with signature ``u1(x)`` or ``u1(mu)(x)``.
            
    Returns
    -------
    B :
        Discretization of the bilinear form :math:`b_\mu` as an affine matrix.
    f :
        Discretization of the linear form :math:`f_\mu` as an affine vector.
    U :
        Function space :math:`U := H^1_{0,}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))` containing the initial condition :math:`u0_\mu` and Dirichlet boundary condition :math:`g_\mu`.
    V :
        Function space :math:`V := H^1_{,0}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))`.
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
    V = {SPACE: H1[SPACE], TIME: H1[TIME]}
    
    F_space = L2
    U1_space = {SPACE: L2[SPACE], TIME: H1[TIME]}
    A_space = utils.change_element(L2[SPACE], shape=(gdim[SPACE], gdim[SPACE]))
    b_space = utils.change_element(L2[SPACE], shape=(gdim[SPACE],))
    c_space = L2[SPACE]
    
    ########################################
    # DISCRETIZ OPERATOR AND DATA
    ########################################
    
    if not isinstance(f, AffineObject): f = AffineObject([1.0], [f])
    if not isinstance(g, AffineObject): g = AffineObject([1.0], [g])
    if not isinstance(u0, AffineObject): u0 = AffineObject([1.0], [u0])
    if not isinstance(u1, AffineObject): u1 = AffineObject([1.0], [u1])
    
    f  = AffineLinear( f.apply2data(lambda fq : interpolate_space_time(F_space, fq)))
    g  = AffineLinear( g.apply2data(lambda gq : interpolate_space_time(U,  gq)))
    u0 = AffineLinear(u0.apply2data(lambda u0q: interpolate_space_time(U,  u0q if np.isscalar(u0q) else lambda t, x: u0q(x))))
    u1 = AffineLinear(u1.apply2data(lambda u1q: interpolate_space_time(U1_space,  u1q if np.isscalar(u1q) else lambda t, x: u1q(x))))
    
    A_, b_, c_ = A
    A_ = A_.apply2data(lambda Aq: utils.interpolate_function(A_space, Aq))
    b_ = b_.apply2data(lambda bq: utils.interpolate_function(b_space, bq))
    c_ = c_.apply2data(lambda cq: utils.interpolate_function(c_space, cq))
    
    ########################################
    # BOUNDARY CONDITIONS
    ########################################
    
    def get_entities(KEY, dbdry):
        return mesh.locate_entities_boundary(msh[KEY], tdim[KEY]-1, dbdry)
    
    bdry_t0 = get_entities(TIME, lambda t: np.isclose(t[0], msh[TIME].geometry.x[:,0].min()))
    bdry_t1 = get_entities(TIME, lambda t: np.isclose(t[0], msh[TIME].geometry.x[:,0].max()))
    bdry_x  = get_entities(SPACE, lambda x: np.ones(x.shape[1], dtype=bool))
    
    #################
    # Dirichlet    
    dbcs_U = [SpaceTimeAffineDirichletBC(TIME,  U, u0,  bdry_t0),
              SpaceTimeAffineDirichletBC(SPACE, U, g,   bdry_x)]
    dbcs_V = [SpaceTimeAffineDirichletBC(TIME,  V, 0.0, bdry_t1),
              SpaceTimeAffineDirichletBC(SPACE, V, 0.0, bdry_x)]
    
    U = SpaceTimeFEniCSxSpaceWithDirichletBCs(U, dbcs_U)
    V = SpaceTimeFEniCSxSpaceWithDirichletBCs(V, dbcs_V, warn=False)
    
    ##################
    # Neumann
    u1 = SpaceTimeAffineDirichletBC(TIME, U1_space, u1, bdry_t0)
    
    ########################################
    # VARIATIONAL FORMULATION
    ########################################

    u = {KEY: ufl.TrialFunction(U.space[KEY]) for KEY in SpaceTimeKey}
    v = {KEY: ufl.TestFunction(V.space[KEY])  for KEY in SpaceTimeKey}
    
    # Mass & stiffness matrix
    M = {KEY:  ufl.inner(u[KEY], v[KEY]) * ufl.dx for KEY in SpaceTimeKey}
    S = {TIME: - ufl.inner(ufl.Dx(u[TIME], 0), ufl.Dx(v[TIME], 0)) * ufl.dx,
         SPACE: - A_.apply2data(lambda Aq: ufl.inner(Aq * ufl.grad(u[SPACE]), ufl.grad(v[SPACE])) * ufl.dx) \
                + b_.apply2data(lambda bq: ufl.inner(bq, ufl.grad(u[SPACE])) * v[SPACE] * ufl.dx) \
                + c_.apply2data(lambda cq: cq * u[SPACE] * v[SPACE] * ufl.dx)}
    
    M = {KEY: utils.assemble_matrix(M[KEY]) for KEY in SpaceTimeKey}
    S = {KEY: utils.assemble_matrix(S[KEY]) for KEY in SpaceTimeKey}
    
    # bilinear form matrix
    B = [{SPACE: M[SPACE], TIME: S[TIME]}, 
         {SPACE: S[SPACE], TIME: M[TIME]}]
    
    # right-hand side vector
    F = {KEY: ufl.inner(ufl.TrialFunction(F_space[KEY]), v[KEY]) * ufl.dx for KEY in SpaceTimeKey}
    F = {KEY: utils.assemble_matrix(F[KEY]) for KEY in SpaceTimeKey}
    f  = (F[SPACE] @ f @ F[TIME].T).apply2data(lambda fq: fq.reshape(-1,))
    
    # initial velocity vector
    def assemble_neumann_boundary(M, bc):
        M_FD = {KEY: M[KEY][:, bc.key_dofs] if bc.key == KEY else M[KEY] for KEY in SpaceTimeKey}
        return affine_kron(M_FD[SPACE], M_FD[TIME]) @ bc
    
    U1 = {SPACE: ufl.inner(ufl.TrialFunction(U1_space[SPACE]), v[SPACE]) * ufl.dx,
          TIME: M[TIME]}
    U1[SPACE] = utils.assemble_matrix(U1[SPACE])
    u1 = assemble_neumann_boundary(U1, u1)
    
    # assemble everything
    B, f = apply_dirichletbc_space_time(B, f + u1, U, V)
    return B, f, U, V


def simple_wave_structured(K: int = 10,
                nx: int | list[int] = 10,
                f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1,
                g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0,
                u0: float | AffineObject = 0,
                u1: float | AffineObject = 0) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], SpaceTimeFEniCSxSpaceWithDirichletBCs, SpaceTimeFEniCSxSpaceWithDirichletBCs]:
    r"""Simple parametric wave equation with a structured tensor product discretization.
    
    See `wave_equation_structured`, with :math:`I=(0,1)`, :math:`\Omega=(0,1)^d`, :math:`A_\mu(x) = -\mu \Delta_x` and :math:`\underline{b}_\mu(x) = 0`, :math:`\underline{c}_\mu(x) = 0`
    """
    
    if np.isscalar(nx) or len(nx) == 1:
        if np.isscalar(nx): nx = [nx]
        msh = {SPACE: mesh.create_interval(MPI.COMM_WORLD, nx[0], [0,1]),
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
    
    print(f"To ensure the CFL condition, K should be at least ``mu * {np.min(nx):.0f}``.")

    A = AffineObject([lambda mu: mu], [-np.eye(gdim[SPACE])])
    b = AffineObject([0.0], [np.ones(gdim[SPACE])])
    c = AffineObject([0.0], [1.0])

    return wave_equation_structured(msh, (A, b, c), f, g, u0, u1)



def wave_equation_hilbert(msh: dict[SpaceTimeKey, mesh.Mesh],
                A: list[AffineObject, AffineObject, AffineObject],
                f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1,
                g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0,
                u0: float | AffineFunction[Mu] = 0,
                u1: float | AffineFunction[Mu] = 0,
                dbdry: Callable[[np.ndarray[float]], np.ndarray[bool]] = None) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], SpaceTimeFEniCSxSpaceWithDirichletBCs, SpaceTimeFEniCSxSpaceWithDirichletBCs]:
    r"""Parametric wave equation with a structured tensor product discretization using the modified hilbert transformation in time.
    
    For some time interval :math:`I` and a spatial domain :math:`\Omega \subset\mathbb{R}^d`, consider the parametric wave equation
    
    .. math::
        \begin{aligned}
        u_{tt}(t,x) + A_\mu(x) u(t,x) &= f_\mu(t,x) \quad\text{for all $(t,x)\in I\times\Omega$}, \\
        u(0,x) &= u0_\mu(x) \quad\text{for all $x\in\Omega$}, \\
        u_t(0,x) &= u1_\mu(x) \quad\text{for all $x\in\Omega$}, \\
        u(t,x) &= g_\mu(t,x) \quad\text{for all $(t,x)\in I\times\partial\Omega$},
        \end{aligned}
        
    with
    
    .. math:: 
        A_\mu(x) u(x) := \nabla_x \cdot (\underline{A}_\mu(x) \nabla_x u(x)) + \underline{b}_\mu(x) \cdot \nabla_x u(x) + \underline{c}_\mu(x) u(x).
        
    Then, after some homogenization, for :math:`U := H^1_{0,}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))` and :math:`V := H^1_{,0}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega)) = \mathcal{H}_T U`, the weak formulation of the wave equation reads: Find :math:`u_\mu \in U` such that
    
    .. math::
        b_\mu(u_\mu, \mathcal{H}_T w) = f_\mu(w) \quad\text{for all } w \in U,
        
    with the bilinear form :math:`b_\mu:U\times U \to \mathbb{R}` and linear form :math:`f_\mu:U \to \mathbb{R}` given by
    
    .. math::
        b_\mu(u,\mathcal{H}_T w) := -(\partial_{t} u,  \partial_t \mathcal{H}_T w)_{L^2(I\times\Omega)} - (\underline{A}_\mu \nabla_x u,  \nabla_x \mathcal{H}_T w)_{L^2(I\times\Omega)} + (\underline{b}_\mu \cdot \nabla_x u, \mathcal{H}_T w)_{L^2(I\times\Omega)} + (\underline{c}_\mu u, \mathcal{H}_T w)_{L^2(I\times\Omega)},\\
        f_\mu(w) := (f_\mu, \mathcal{H}_T w)_{L^2(I\times\Omega)} + \left(u_1, [\mathcal{H}_T w](T)\right)_{L^2(\Gamma_N^i)}.
        
    .. note::
        In contrast to `wave_equation_structured` or a unstructured discretization based on `weak_problem`, using the modified hilbert transformation leads to a unconditionally stable discretization, i.e. the resulting linear system is always solvable, independent of the space-time mesh (no CFL condition needed!).
    
    Args:
        msh:
            Dictionary containing the spatial and temporal mesh.
        A:
            List of three affine objects ``(A_,b_,c_)`` corresponding to :math:`\underline{A}_\mu,\underline{b}_\mu,\underline{c}_\mu`. Each of these affine objects should be compatible with `utils.interpolate_function`.
        f:
            (Parametric) right-hand side. A scalar or an (affine) function with signature ``f(t,x)`` or ``f(mu)(t,x)``.
        g:
            (Parametric) Dirichlet boundary condition. A scalar or an (affine) function with signature ``g(t,x)`` or ``g(mu)(t,x)``.
        u0:
            (Parametric) Initial condition. A scalar or an (affine) function with signature ``u0(x)`` or ``u0(mu)(x)``.
        u1:
            (Parametric) Initial velocity. A scalar or an (affine) function with signature ``u1(x)`` or ``u1(mu)(x)``.
            
    Returns
    -------
    B :
        Discretization of the bilinear form :math:`b_\mu` as an affine matrix.
    f :
        Discretization of the linear form :math:`f_\mu` as an affine vector.
    U :
        Function space :math:`U := H^1_{0,}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))` containing the initial condition :math:`u0_\mu` and Dirichlet boundary condition :math:`g_\mu`.
    V :
        Function space :math:`\tilde{V} := H^1_{0,}(I;L^2(\Omega)) \cap L^2(I;H^1_0(\Omega))`, such that :math:`V = \mathcal{H}_T \tilde{V}`.
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
    V = {SPACE: H1[SPACE], TIME: H1[TIME]}
    
    F_space = {SPACE: L2[SPACE], TIME: H1[TIME]}
    U1_space = {SPACE: L2[SPACE], TIME: H1[TIME]}
    A_space = utils.change_element(L2[SPACE], shape=(gdim[SPACE], gdim[SPACE]))
    b_space = utils.change_element(L2[SPACE], shape=(gdim[SPACE],))
    c_space = L2[SPACE]
    
    ########################################
    # DISCRETIZ OPERATOR AND DATA
    ########################################
    
    if not isinstance(f, AffineObject): f = AffineObject([1.0], [f])
    if not isinstance(g, AffineObject): g = AffineObject([1.0], [g])
    if not isinstance(u0, AffineObject): u0 = AffineObject([1.0], [u0])
    if not isinstance(u1, AffineObject): u1 = AffineObject([1.0], [u1])
    
    f  = AffineLinear( f.apply2data(lambda fq : interpolate_space_time(F_space, fq)))
    g  = AffineLinear( g.apply2data(lambda gq : interpolate_space_time(U,  gq)))
    u0 = AffineLinear(u0.apply2data(lambda u0q: interpolate_space_time(U,  u0q if np.isscalar(u0q) else lambda t, x: u0q(x))))
    u1 = AffineLinear(u1.apply2data(lambda u1q: interpolate_space_time(U1_space,  u1q if np.isscalar(u1q) else lambda t, x: u1q(x))))
    
    A_, b_, c_ = A
    A_ = A_.apply2data(lambda Aq: utils.interpolate_function(A_space, Aq))
    b_ = b_.apply2data(lambda bq: utils.interpolate_function(b_space, bq))
    c_ = c_.apply2data(lambda cq: utils.interpolate_function(c_space, cq))
    
    ########################################
    # BOUNDARY CONDITIONS
    ########################################
    
    def get_entities(KEY, dbdry):
        return mesh.locate_entities_boundary(msh[KEY], tdim[KEY]-1, dbdry)
    
    bdry_t0 = get_entities(TIME, lambda t: np.isclose(t[0], msh[TIME].geometry.x[:,0].min()))
    bdry_x  = get_entities(SPACE, lambda x: np.ones(x.shape[1], dtype=bool))
    
    #################
    # Dirichlet    
    dbcs_U = [SpaceTimeAffineDirichletBC(TIME,  U, u0,  bdry_t0), 
              SpaceTimeAffineDirichletBC(SPACE, U, g,   bdry_x)]
    dbcs_V = [SpaceTimeAffineDirichletBC(TIME,  V, 0.0, bdry_t0),
              SpaceTimeAffineDirichletBC(SPACE, V, 0.0, bdry_x)]
    
    U = SpaceTimeFEniCSxSpaceWithDirichletBCs(U, dbcs_U)
    V = SpaceTimeFEniCSxSpaceWithDirichletBCs(V, dbcs_V, warn=False)
    
    ##################
    # Neumann
    u1 = SpaceTimeAffineDirichletBC(TIME, U1_space, u1, bdry_t0)
    
    ########################################
    # VARIATIONAL FORMULATION
    ########################################

    u = {SPACE: ufl.TrialFunction(U.space[SPACE])}
    v = {SPACE: ufl.TestFunction(V.space[SPACE])}
    
    # Mass & stiffness matrix
    if V.space[TIME] is not H1[TIME] or U.space[TIME] is not H1[TIME]:
        raise ValueError("The Hilbert transformation in time is only implemented for H1 elements.")
    
    M = {SPACE:  ufl.inner(u[SPACE], v[SPACE]) * ufl.dx,
         TIME: hilbert_transform.mass_matrix(U.space[TIME])}
    S = {TIME: -hilbert_transform.stiffness_matrix(U.space[TIME]),
         SPACE: - A_.apply2data(lambda Aq: ufl.inner(Aq * ufl.grad(u[SPACE]), ufl.grad(v[SPACE])) * ufl.dx) \
                + b_.apply2data(lambda bq: ufl.inner(bq, ufl.grad(u[SPACE])) * v[SPACE] * ufl.dx) \
                + c_.apply2data(lambda cq: cq * u[SPACE] * v[SPACE] * ufl.dx)}
    
    M[SPACE] = utils.assemble_matrix(M[SPACE])
    S[SPACE] = utils.assemble_matrix(S[SPACE])
    
    # bilinear form matrix
    B = [{SPACE: M[SPACE], TIME: S[TIME]}, 
         {SPACE: S[SPACE], TIME: M[TIME]}]
    
    # right-hand side vector
    if F_space[TIME] is not H1[TIME]:
        raise ValueError("The Hilbert transformation in time is only implemented for H1 elements.")
    
    F = {SPACE: utils.assemble_matrix(ufl.inner(ufl.TrialFunction(F_space[SPACE]), v[SPACE]) * ufl.dx),
         TIME: M[TIME]}
    f  = (F[SPACE] @ f @ F[TIME].T).apply2data(lambda fq: fq.reshape(-1,))
    
    # initial velocity vector
    def assemble_neumann_boundary(M, bc):
        M_FD = {KEY: M[KEY][:, bc.key_dofs] if bc.key == KEY else M[KEY] for KEY in SpaceTimeKey}
        return affine_kron(M_FD[SPACE], M_FD[TIME]) @ bc
    
    if U1_space[TIME] is not H1[TIME]:
        raise ValueError("The Hilbert transformation in time is only implemented for H1 elements.")
    
    U1 = {SPACE: utils.assemble_matrix(ufl.inner(ufl.TrialFunction(U1_space[SPACE]), v[SPACE]) * ufl.dx),
          TIME: M[TIME]}
    u1 = assemble_neumann_boundary(U1, u1)
    
    # assemble everything
    B, f = apply_dirichletbc_space_time(B, f + u1, U, V)
    return B, f, U, V


def simple_wave_hilbert(K: int = 10,
                nx: int | list[int] = 10,
                f: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 1,
                g: float | Callable[[float, np.ndarray], float] | AffineFunction[Mu] = 0,
                u0: float | AffineObject = 0,
                u1: float | AffineObject = 0) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector], SpaceTimeFEniCSxSpaceWithDirichletBCs, SpaceTimeFEniCSxSpaceWithDirichletBCs]:
    r"""Simple parametric wave equation with structured tensor product discretization using the modified hilbert transformation in time.
    
    See `wave_equation_hilbert`, with :math:`I=(0,1)`, :math:`\Omega=(0,1)^d`, :math:`A_\mu(x) = -\mu \Delta_x` and :math:`\underline{b}_\mu(x) = 0`, :math:`\underline{c}_\mu(x) = 0`
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

    A = AffineObject([lambda mu: mu], [-np.eye(gdim[SPACE])])
    b = AffineObject([0.0], [np.ones(gdim[SPACE])])
    c = AffineObject([0.0], [1.0])

    return wave_equation_hilbert(msh, (A, b, c), f, g, u0, u1)