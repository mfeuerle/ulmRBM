from numbers import Number
from collections.abc import Callable
from enum import IntEnum

import numpy as np

from dolfinx import fem

from ulmRBM.fenicsx import utils
from ulmRBM.core import Mu, Matrix, Vector
from ulmRBM.affine import AffineObject, AffineLinear, wrap_affinelinear, affine_kron

from ulmRBM.affine._affine import _ConstructNew
from ._basic import free_dofs

__all__ = [
    'SpaceTimeKey',
    'interpolate_space_time',
    'SpaceTimeAffineDirichletBC',
    'SpaceTimeFEniCSxSpaceWithDirichletBCs',
    'apply_dirichletbc_space_time',
    'zero_overlapping_space_time_bcs'
]

class SpaceTimeKey(IntEnum):
    r"""Readability key for space-time problems, to distinguish between spatial and temporal objects.
    
    Can be used as a key in a dictionary, e.g. ``object = {SpaceTimeKey.SPACE: space_object, SpaceTimeKey.TIME: time_object}`` or as a list ``object = [space_object, time_object]``. Then, accessing the object can be done by ``object[SpaceTimeKey.SPACE]`` or ``object[SpaceTimeKey.TIME]``.
    """
    
    SPACE = 0
    """Spatial identifier"""
    TIME = 1
    """Temporal identifier"""
    
SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME


def interpolate_space_time(U: dict[SpaceTimeKey, fem.FunctionSpace], u: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    r"""Interpolate a space-time function into a tensor-product of two FEniCSx function spaces.
    
    Consider the tensor-product of two FEniCSx function spaces :math:`U = U_x \otimes U_t` with the spatial function space :math:`U_x` and the temporal function space :math:`U_t`, and a function :math:`u(t,x)` defined on the space-time domain. This function interpolates :math:`u` into the tensor-product function space :math:`U`.
    
    Args:
        U:
            Tensor-product of two FEniCSx function spaces :math:`U = U_x \otimes U_t` given by its parts :math:`U_x` and :math:`U_t`.
        u:
            Function :math:`u(t,x)` to interpolate into the tensor-product function space :math:`U`. Thereby, for fixed :math:`t`, the function :math:`u(t,\cdot)` should be compatible with `dolfinx.fem.Function.interpolate`.
    
    Returns:
        Matrix representation of the interpolated function :math:`u`, with ``out[i,k]`` being the value of the interpolated function at the :math:`i`-th spatial dof and the :math:`k`-th temporal dof.
    """
    
    if np.isscalar(u):
        u_val = u
        u = lambda t,x: np.ones(x.shape[1]) * u_val
    
    def fixe_time(f,t):
        return lambda x: f(t,x)
    
    T = utils.get_interpolation_points(U[TIME])[0]
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


        
class SpaceTimeAffineDirichletBC(AffineLinear[Mu, np.ndarray]):
    r"""Purely temporal or purely spatial affine Dirichlet boundary condition on the Kroneckker product of two FEniCSx function spaces.
    
    Consider a temporal function space :math:`U_t` and a spatial function space :math:`U_x`. Then, the Kronecker product of these two spaces is given by :math:`U = U_x \otimes U_t`. A purely temporal Dirichlet boundary condition is then a Dirichlet boundary condition on :math:`U_t` that is applied to all dofs in :math:`U_x`, while a purely spatial Dirichlet boundary condition is a Dirichlet boundary condition on :math:`U_x` that is applied to all dofs in :math:`U_t`.
    """
    
    space: dict[SpaceTimeKey, fem.FunctionSpace]
    """Underlying function spaces of the boundary condition, with keys `SpaceTimeKey`."""
    dim: dict[SpaceTimeKey, int]
    """Dimensions of the function spaces, i.e. total number of dofs (free and restricted) in each space."""
    key: SpaceTimeKey
    """Key indicating whether the boundary condition is applied in space or time."""
    key_dofs: np.ndarray[int]
    """List of dofs restricted by this boundary condition in the space given by `key`."""
    dofs: np.ndarray[int]
    """List of dofs restricted by this boundary condition in the full space-time function space."""
    
    def __init__(self, key: SpaceTimeKey, 
                 space: dict[SpaceTimeKey, fem.FunctionSpace], 
                 g: Number | np.ndarray | AffineLinear[Mu, np.ndarray], 
                 entities: np.ndarray[int] = None, 
                 dofs: np.ndarray[int] = None):
        r"""
        Args:
            key:
                Key indicating whether the boundary condition is applied in space or time.
            space:
                Tensor-product space :math:`U: U_x\otimes U_t` given by its parts :math:`U_x` and :math:`U_t`.
            g:
                Affine representation of the boundary condition values of shape ``(dim[SPACE], dim[TIME])`` or a scalar. Alternatively, for a spatial boundary condition of shape ``(len(dofs), dim[TIME])`` or for a temporal boundary condition of shape ``(dim[SPACE], len(dofs))``.
            entities:
                Mesh entities (e.g. facets) on the temporal or spatial domain (as defined by `key`) on which the boundary condition is applied. If entities are given, do not provide dofs. The dofs are determined from the entities using the function space's dofmap.
            dofs:
                List of dofs on the temporal or spatial domain (as defined by `key`) on which the boundary condition is applied. If dofs are given, do not provide entities.
        """
        
        if entities is None and dofs is None:
            raise ValueError("Either entities or dofs must be given to define the boundary condition.")
        if entities is not None and dofs is not None:
            raise ValueError("Only one of entities or dofs can be given to define the boundary condition.")
        
        if key not in SpaceTimeKey:
            raise ValueError(f"Invalid key, must be one of {list(SpaceTimeKey)}.")
        
        if dofs is None:
            dofs = fem.locate_dofs_topological(space[key], space[key].mesh.topology.dim-1, entities)
        if dofs.dtype == bool:
            dofs = np.where(dofs)[0]
        dofs = dofs.reshape(-1)
        
        self.space = space
        self.key = key
        self.dim = {KEY: space[KEY].dofmap.index_map.size_global for KEY in SpaceTimeKey}
        self.key_dofs = dofs
        
        if key == SPACE:
            dofs = dofs.reshape(-1,1)*self.dim[TIME] + np.arange(self.dim[TIME]).reshape(1,-1)
        else: # key == TIME
            dofs = np.arange(self.dim[SPACE]).reshape(-1,1)*self.dim[TIME] + dofs.reshape(1,-1)
            
        self.dofs = dofs.reshape(-1,)
        
        if np.isscalar(g) or g.shape == ():
            g = np.full(dofs.shape, g)
        
        g = wrap_affinelinear(g)
        
        if len(g.shape) != 2:
            raise ValueError("Boundary condition values must be a 2D array with first dimension space and second dimension time.")
        
        if key == SPACE:
            if g.shape[TIME] != self.dim[TIME]:
                raise ValueError(f"Boundary condition values must have second dimension equal to the number of dofs in the temporal function space ({self.dim[TIME]}).")
            
            if g.shape[SPACE] == self.dim[SPACE]:
                g = g.apply2data(lambda gq: gq[self.key_dofs,:])
            elif g.shape[SPACE] != len(self.key_dofs):
                raise ValueError(f"Boundary condition values must have first dimension equal to the number of dofs ({self.dim[SPACE]}) in the spatial function space or the number of dofs restricted by the boundary condition ({len(self.key_dofs)}).")
            
        else: # key == TIME
            if g.shape[SPACE] != self.dim[SPACE]:
                raise ValueError(f"Boundary condition values must have first dimension equal to the number of dofs in the spatial function space ({self.dim[SPACE]}).")
            
            if g.shape[TIME] == self.dim[TIME]:
                g = g.apply2data(lambda gq: gq[:,self.key_dofs])
            elif g.shape[TIME] != len(self.key_dofs):
                raise ValueError(f"Boundary condition values must have second dimension equal to the number of dofs ({self.dim[TIME]}) in the temporal function space or the number of dofs restricted by the boundary condition ({len(self.key_dofs)}).")
        
        g = g.apply2data(lambda gq: gq.reshape(-1,))
        super().__init__(g)
        
    def _construct_new(self, theta, data, type: _ConstructNew = _ConstructNew.SAME) -> AffineLinear[Mu, np.ndarray]:
        g = AffineLinear(theta, data)
        dim = (len(self.key_dofs), self.dim[TIME]) if self.key == SPACE else (self.dim[SPACE], len(self.key_dofs))
        
        if type == _ConstructNew.SAME:
            return self.__class__(self.key, self.space, g.apply2data(lambda gq: gq.reshape(dim)), dofs=self.key_dofs)
        try:
            return self.__class__(self.key, self.space, g.apply2data(lambda gq: gq.reshape(dim)), dofs=self.key_dofs)
        except:
            return g

    def set(self, mu: Mu, x: np.ndarray = None):
        r"""Set the boundary condition.
        
        Args:
            mu:
                Parameter value at which to evaluate the boundary condition.
            x:
                Vector on which the boundary condition values will be set. If ``None``, a zero vector is used. The vector must have length equal to the total number of dofs in the space-time function space, i.e. ``dim[SPACE]*dim[TIME]``, such that, after ``x.reshape(dim[SPACE], dim[TIME])``, the the first dimension corresponds to the spatial dofs and the second dimension to the temporal dofs. Alternatively, a vector of length equal to the number of free / unrestricted dofs.
                
        Returns:
            Vector with set boundary condition values of shape ``(dim[SPACE]*dim[TIME],)``.
        """
        
        full_dim = self.dim[SPACE]*self.dim[TIME]
        if x is None:
            x = np.zeros(full_dim)
        
        if x.shape[0] == len(self.dofs):
            _x = np.zeros(full_dim)
            free_dofs = np.ones(full_dim, dtype=bool)
            free_dofs[self.dofs] = False
            _x[free_dofs] = x
            x = _x
            
        if x.shape[0] == full_dim:
            x[self.dofs] = self(mu)
        else:
            raise ValueError(f"Input vector has incompatible shape, should be either dimension of the space ({full_dim}) or number of free dofs ({full_dim-len(self.dofs)}).")

        return x
    
    
def zero_overlapping_space_time_bcs(bcs: list[SpaceTimeAffineDirichletBC], priority: SpaceTimeKey = TIME) -> list[SpaceTimeAffineDirichletBC]:
    r"""Zero out values of boundary conditions at dofs, which are set by another boundary condition with higher priority.
    
    `SpaceTimeAffineDirichletBC` represents a boundary condition either exclusively in space or in time, e.g. it specifies either ``u[:,bc1.key_dofs].flatten() = bc1`` or ``u[bc2.key_dofs,:].flatten() = bc2`. Now, if there are at least one boundary condition in space and time, there will be an overlap ``u[bc2.key_dofs,bc1.key_dofs]``, wich is defined by both boundary conditions. Compleatly removing these dofs from one boundary condition gets quite complex. So, this is a simple work-around by setting the values of one of the boundary conditions at these overlapping dofs to zero, i.e. they are still part of both boundary conditions, but one of them has zero values at the overlapping dofs.
    
    The list of boundary conditions is returned with the priority boundary conditions (with non-zero values at the overlapping dofs) at the end of the list, i.e. if boundary conditions are set in order, the one with ``priority`` will be applied last and thus overriding any set values of the others.
    """
    bcs = [bc for bc in bcs if bc.key != priority] + [bc for bc in bcs if bc.key == priority]
    
    assert all([bc.dim == bcs[0].dim for bc in bcs]), "Boundary conditions have different dimensions."

    priority_dofs = np.zeros(bcs[0].dim[SPACE]*bcs[0].dim[TIME], dtype=bool)
    for bc in bcs:
        if bc.key == priority:
            priority_dofs[bc.dofs] = True
            
    for bc in bcs:
        if bc.key != priority:
            double_defined_dofs = priority_dofs[bc.dofs]
            if np.any(double_defined_dofs):
                for bcq in bc.data: bcq[double_defined_dofs] = 0.0
                
    return bcs
    
    
    
    
class SpaceTimeFEniCSxSpaceWithDirichletBCs:
    r""" Combination of a Tensor-product product of two FEniCSx function spaces, together with space-time boundary conditions.
    
    Conditions in time have priority over conditions in space, i.e. if there are conflicting boundary conditions at a few dofs, the temporal boundary conditions will override the spatial ones.
    """
    
    space: dict[SpaceTimeKey, fem.FunctionSpace]
    bcs: list[SpaceTimeAffineDirichletBC]
    full_dofs: np.ndarray[bool]
    _ndofs: int
    
    def __init__(self, space: dict[SpaceTimeKey, fem.FunctionSpace], bcs: list[SpaceTimeAffineDirichletBC], warn: bool = True, priority: SpaceTimeKey = TIME):
        r"""
        Args:
            space:
                Tensor-product space :math:`U: U_x\otimes U_t` given by its parts :math:`U_x` and :math:`U_t`.
            bcs:
                List of dirichlet boundary conditions.
            warn:
                Whether to warn, if there are spatial dofs set by multiple spatial boundary conditions or temporal dofs set by multiple temporal boundary conditions.
            priority:
                Key indicating whether the spatial or temporal boundary conditions have priority, i.e. if there are conflicting boundary conditions at a few dofs, the boundary conditions with this key will override the others.
        """
        
        if any([bc.space[KEY] != space[KEY] for KEY in SpaceTimeKey for bc in bcs]):
            raise ValueError("Function space does not match the space of some boundary conditions.")

        self.dim = {KEY: space[KEY].dofmap.index_map.size_global for KEY in SpaceTimeKey}
        self.space = space
        self.bcs = zero_overlapping_space_time_bcs(bcs, priority=priority)
        
        self.dofs = {KEY: free_dofs(self.dim[KEY], [bc.key_dofs for bc in self.bcs if bc.key == KEY], warn=warn) for KEY in SpaceTimeKey}
        self.full_dofs = np.outer(self.dofs[SPACE], self.dofs[TIME]).flatten()
        self._ndofs = {KEY: sum(self.dofs[KEY]) for KEY in SpaceTimeKey}
        
    def __repr__(self):
        return f"<{self.__class__.__name__} of dim={self.dim[SPACE]}x{self.dim[TIME]}={self.dim[SPACE]*self.dim[TIME]} and {self._ndofs[SPACE]}x{self._ndofs[TIME]}={self._ndofs[SPACE]*self._ndofs[TIME]} free dofs>"
        
    def set_dirichletbcs(self, mu, u: np.ndarray | list[fem.Function], as_matrix: bool | None = None) -> np.ndarray | list[fem.Function]:
        r""" Insert the dirichlet boundary conditions into a given vector.
        
        Args:
            mu:
                Parameter value, at which the boundary conditions should be evaluated.
            u:
                Either a 1D vector of shape ``(dim[SPACE]*dim[TIME],)`` or ``(free_dofs[SPACE]*free_dofs[TIME])``,
                a 2D matrix of shape ``(dim[SPACE], dim[TIME])``, ``(free_dofs[SPACE], dim[TIME])``, ``(dim[SPACE], free_dofs[TIME])`` or ``(free_dofs[SPACE], free_dofs[TIME])``,
                or a list of spatial FEniCS functions of length ``dim[TIME]``. Thereby, ``free_dofs[SPACE/TIME]`` denote the dofs, which are not restricted by any of the spatial/temporal boundary conditions.
            as_matrix:
                If True, the output will be a 2D matrix of shape ``(dim[SPACE], dim[TIME])``. If False, the output will be a 1D vector of shape ``(dim[SPACE]*dim[TIME],)``. If None, the output will have the same shape as the input u, i.e. matrix, vector or list of FEniCS functions.
                
        Returns:
            Vector of shape ``(dim[SPACE]*dim[TIME],)``, matrix of shape ``(dim[SPACE], dim[TIME])`` or list of FEniCS functions of length ``dim[TIME]`` with set boundary condition values of shape, depending on the shape of ``u`` and the value of ``as_matrix``.        
        """ 
        
        is_fenics = isinstance(u[0], fem.Function)
        
        if is_fenics:    
            u_fenics = [uk for uk in u]
            assert all([isinstance(uk, fem.Function) for uk in u_fenics]), "If u is a list of FEniCS functions, all entries must be FEniCS functions."
            u = np.array([uk.x.array for uk in u_fenics]).T
        else:
            u = np.asarray(u)
            
        is_1d = u.ndim == 1
            
        if is_1d:
            if u.shape[0] == self._ndofs[SPACE]*self._ndofs[TIME]:
                u = u.reshape(self._ndofs[SPACE], self._ndofs[TIME])
            elif u.shape[0] == self.dim[SPACE]*self.dim[TIME]:
                u = u.reshape(self.dim[SPACE], self.dim[TIME])
            else:
                raise ValueError(f"Input vector has incompatible shape, for 1D vectors, should be either dimension of the space ({self.dim[SPACE]*self.dim[TIME]}) or number of free dofs ({self._ndofs[SPACE]*self._ndofs[TIME]}) but has {u.shape[0]} elements.")

        if u.shape[TIME] == self._ndofs[TIME]:
            u_new = np.zeros((u.shape[SPACE], self.dim[TIME]))
            u_new[:,self.dofs[TIME]] = u
            u = u_new
            if is_fenics:
                u_fenics_new = [fem.Function(self.space[SPACE]) for _ in range(self.dim[TIME])]
                u_fenics_new[self.dofs[TIME]] = u_fenics
                u_fenics = u_fenics_new
        
        if u.shape[SPACE] == self._ndofs[SPACE]:
            u_new = np.zeros((self.dim[SPACE], self.dim[TIME]))
            u_new[self.dofs[SPACE],:] = u
            u = u_new
                
        u = u.reshape(-1)
        for bc in self.bcs: u = bc.set(mu, u)
        
        if is_fenics or not is_1d:
            u = u.reshape(self.dim[SPACE], self.dim[TIME])
        
        if is_fenics and as_matrix is None:
            for k in range(self.dim[TIME]):
                u_fenics[k].x.array[:] = u[:,k]
            u = u_fenics
        
        if as_matrix is not None:
            if as_matrix:
                u = u.reshape(self.dim[SPACE], self.dim[TIME])
            else:
                u = u.reshape(-1)
                
        return u



def apply_dirichletbc_space_time(B: list[dict[SpaceTimeKey, AffineLinear[Mu, Matrix]]],
                    f: AffineLinear[Mu, Vector],
                    U: SpaceTimeFEniCSxSpaceWithDirichletBCs, 
                    V: SpaceTimeFEniCSxSpaceWithDirichletBCs) -> tuple[AffineLinear[Mu,Matrix], AffineLinear[Mu,Vector]]:
    r"""Apply Space-Time Dirichlet boundary condtions to the right-hand side.
    
    Args:
        B:
            System matrix given by ``B = B_i[SPACE] \otimes B_i[TIME]`` for each ``i`` in ``len(B)``.
        f:
            Right-hand side vector.
        U:
            Function space containing the Dirichlet boundary conditions.
        V:
            Function space for the solution.
            
    Returns
    -------
    B:
        System matrix with test and trial Dirichlet boundary dofs removed.
    f:
        Right-hand side vector with test Dirichlet boundary dofs removed and trial Dirichlet boundary conditions applied.
    """
    B = [{KEY: wrap_affinelinear(Bi[KEY]) for KEY in SpaceTimeKey} for Bi in B]
    f = wrap_affinelinear(f)
    
    B_F =  [{KEY: Bi[KEY].apply2data(lambda Biq: Biq[V.dofs[KEY]])    for KEY in SpaceTimeKey} for Bi in B]
    B_FF = [{KEY: Bi[KEY].apply2data(lambda Biq: Biq[:, U.dofs[KEY]]) for KEY in SpaceTimeKey} for Bi in B_F]
    
    def apply_bc(bc):        
        B_FD = [{KEY: Bi[KEY].apply2data(lambda Biq: Biq[:, bc.key_dofs]) if bc.key == KEY else Bi[KEY] for KEY in SpaceTimeKey} for Bi in B_F]
        B_FD = sum([affine_kron(Bi[SPACE], Bi[TIME]) for Bi in B_FD])
        return B_FD @ bc
    
    B_FF = sum([affine_kron(Bi[SPACE], Bi[TIME]) for Bi in B_FF])
    B_FD_bcs = sum([apply_bc(bc) for bc in U.bcs])
    f_F  = f.apply2data(lambda fq: fq[V.full_dofs]) - B_FD_bcs
    
    return B_FF.compress(), f_F.compress()