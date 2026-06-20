
from numbers import Number
import numpy as np
from scipy.sparse import sparray, csr_array
from enum import IntEnum

from dolfinx import fem
import ufl

from ulmRBM.fenicsx import utils

from ulmRBM.core import Mu
from ulmRBM.affine import AffineObject, AffineLinear, wrap_affinelinear

from ulmRBM.affine._affine import _ConstructNew

__all__ = [
    'utils',
    'AffineDirichletBC',
    'FEniCSxSpaceWithDirichletBCs',
    'SpaceTimeAffineDirichletBC',
    'SpaceTimeFEniCSxSpaceWithDirichletBCs',
    'free_dofs',
    'SpaceTimeKey',
    'zero_overlapping_space_time_bcs'
]

class SpaceTimeKey(IntEnum):
    SPACE = 0
    TIME = 1
    
SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME

class AffineDirichletBC(AffineLinear[Mu, np.ndarray]):
    def __init__(self, space: fem.FunctionSpace, g: AffineObject[Mu, np.ndarray] | any, entities = None, dofs = None):
        """ Affine Dirichlet boundary condition on a FEniCSx function space.
        
        Args:
            space:
                FEniCSx function space on which the boundary condition is defined.
            g:
                AffineObject representing the boundary values of the Dirichlet boundary condition.
                If g is not an `AffineObject`, it is converted to one using AffineObject([1.0], [g]).
                Then, if g is (convertible to) an `AffineLinear`, i.e. a (affine) vector, it has to be of length equal to the number of dofs in the function space or equal to the number of dofs restricted by the boundary condition.
                If it is not (convertible to) an AffineLinear, it is assumed to be a  fenicsx object representing a boundary condition (e.g. a Function a Constant, or a ufl Expression), which has to be supported by :func:`utils.dirichletbc`.
            entities:
                Mesh entities (e.g. facets) on which the boundary condition is applied. If entities are given, do not provide dofs. The dofs are determined from the entities using the function space's dofmap.
            dofs:
                List of dofs on which the boundary condition is applied. If dofs are given, do not provide entities.
        """
        if entities is None and dofs is None:
            raise ValueError("Either entities or dofs must be given to define the boundary condition.")
        if entities is not None and dofs is not None:
            raise ValueError("Only one of entities or dofs can be given to define the boundary condition.")
        
        if dofs is None:
            dofs = fem.locate_dofs_topological(space, space.mesh.topology.dim-1, entities)
        
        
        if not isinstance(g, AffineObject):
            g = AffineObject([1.0], [g])
            try:
                g = AffineLinear(g)
            except: pass
            
        
        def as_vector(g):
            # messy, weil fenics gefühlt 100 wege hat die scheiß boundary condition zu speichern
            try:
                g = g.x.array
            except AttributeError:
                pass
            try: 
                g = g.value
            except AttributeError:
                pass

            if isinstance(g, Number) or g.shape == ():
                return np.full(len(dofs), g)
            else:
                if g.ndim != 1 and g.shape[1] != 1:
                    raise ValueError("Boundary condition values must be a scalar or a 1D array.")
                g = g.reshape(-1)
                if len(g) == len(dofs):
                    return g
                elif len(g) == space.dofmap.index_map.size_global:
                    return g[dofs]
                else:
                    raise ValueError("Boundary condition values must have length equal to the number of dofs in the function space or the number of dofs restricted by the boundary condition.")
            
        
        if isinstance(g, AffineLinear):
            g = g.apply2data(lambda gq: as_vector(gq))
        else:
            g = g.apply2data(lambda gq: utils.dirichletbc(space, gq, dofs))
            g = g.apply2data(lambda gq: as_vector(gq.g))
        g = g.compress()
                    
        super().__init__(g)
        self.space: fem.FunctionSpace = space
        """Underlying function space of the boundary condition"""
        self.dofs: np.ndarray[int] = dofs
        """List of dofs restricted by this boundary condition"""
        self.dim: int = space.dofmap.index_map.size_global
        """Dimension of the function space, i.e. total number of dofs (free and restricted)"""
        
    def _construct_new(self, theta, data, type: _ConstructNew = _ConstructNew.SAME) -> AffineLinear[Mu, np.ndarray]:
        g = AffineLinear(theta, data)
        if type == _ConstructNew.SAME:
            return self.__class__(self.space, g, dofs=self.dofs)
        try:
            return self.__class__(self.space, g, dofs=self.dofs)
        except:
            return g
        
        
    def set(self, mu: Mu, x: np.ndarray = None):
        """Set the boundary condition.

        Args:
            mu:
                Parameter value at which to evaluate the boundary condition.
            x:
                If given, the boundary condition values will be written to the corresponding dofs in x.
                Otherwise, a zero vector is created and the boundary condition values are written to the corresponding dofs in this vector.
                
        Returns:
            If x is given, the boundary condition values will be set inplace. Otherwise, a new vector of length equal to the number of dofs in the function space, with boundary condition values at the corresponding dofs and zeros elsewhere.
        """
        if x is None:
            x = np.zeros(self.dim)
        
        if x.shape[0] == len(self.dofs):
            _x = np.zeros(self.dim)
            free_dofs = np.ones(self.dim, dtype=bool)
            free_dofs[self.dofs] = False
            _x[free_dofs] = x
            x = _x
            
        if x.shape[0] == self.dim:
            x[self.dofs] = self(mu)
        else:
            raise ValueError(f"Input vector has incompatible shape, should be either dimension of the space ({self.dim}) or number of free dofs ({self.dim-len(self.dofs)}).")

        return x
        
    def tofenicsx(self, mu) -> fem.DirichletBC:
        """
        Convert the affine Dirichlet boundary condition to a FEniCSx DirichletBC object.
        
        Args:
            mu:
                Parameter value at which to evaluate the boundary condition.
        Returns:
            A FEniCSx DirichletBC object representing the boundary condition at the given parameter value.
        """
        g = fem.Function(self.space)
        g.x.array[self.dofs] = self(mu)
        return fem.dirichletbc(g, self.dofs)          
            
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
        
        assert all(bc.space == space for bc in bcs)
        
        self.space: fem.FunctionSpace = space
        "Function space"
        self.dim: int = space.dofmap.index_map.size_global
        "Dimension of the space (number of free + restricted dofs)."
        self.bcs: list[AffineDirichletBC] = bcs
        "List of all dirichlet boundary conditions of this space."
        self.dofs: np.ndarray[bool] = free_dofs(self.dim, [bc.dofs for bc in bcs], warn=warn)
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
        
        
class SpaceTimeAffineDirichletBC(AffineLinear[Mu, np.ndarray]):
    
    dim: dict[SpaceTimeKey, int]
    space: dict[SpaceTimeKey, fem.FunctionSpace]
    key: SpaceTimeKey
    key_dofs: dict[SpaceTimeKey, np.ndarray[int]]
    dofs: np.ndarray[int]
    
    def __init__(self, key: SpaceTimeKey, space: dict[SpaceTimeKey, fem.FunctionSpace], g: Number | np.ndarray | AffineLinear[Mu, np.ndarray], entities: np.ndarray[int] = None, dofs: np.ndarray[int] = None):
        
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
    
    `SpaceTimeAffineDirichletBC` represents a boundary condition either in exclusively in space or in time, e.g. specify the either ``u[:,bc1.key_dofs].flatten() = bc1`` or ``u[bc2.key_dofs,:].flatten() = bc2`. Now, if there are at least one boundary condition in space and time, there will be an overlap ``u[bc2.key_dofs,bc1.key_dofs]``, wich is defined by both boundary conditions. Compleatly removing these dofs from one boundary condition gets quite complex. So, this is a simple work-around by setting the values of one of the boundary conditions at these overlapping dofs to zero, i.e. they are still part of both boundary conditions, but one of them has zero values at the overlapping dofs.
    
    The list of boundary conditions is returned with the priority boundary conditions (with non-zero values at the overlapping dofs) at the end of the list, i.e. if boundary conditions are applied in order, the one with `priority` will be applied last and thus overriding the others.
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
    r""" Combination of a FEniCSx FunctionSpace and boundary conditions.
    
    Conditions in time have priority over conditions in space, i.e. if there are conflicting boundary conditions, the one with key TIME is applied.
    """
    
    space: dict[SpaceTimeKey, fem.FunctionSpace]
    bcs: list[SpaceTimeAffineDirichletBC]
    full_dofs: np.ndarray[bool]
    _ndofs: int
    
    def __init__(self, space: dict[SpaceTimeKey, fem.FunctionSpace], bcs: list[SpaceTimeAffineDirichletBC], warn: bool = True, priority: SpaceTimeKey = TIME):
        r"""
        Args:
            space:
                Function space to be wrapped.
            bcs:
                List of dirichlet boundary conditions.
            warn:
                Whether to warn, if there are dofs set by multiple boundary conditions.
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
        
    def set_dirichletbcs(self, mu, u: np.ndarray | list[fem.Function], as_matrix: bool | None = None):
        r""" Insert the dirichlet boundary conditions into a given vector.
        
        Args:
            mu:
                Parameter value, at which the boundary conditions should be evaluated.
            u:
                Vector (or FEniCS function) containing the values at the free dofs, to which the dirichlet boundary conditions should be added. Can either have length equal to the total number of dofs (free + restricted) or only the number of free dofs.
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
        
        if is_fenics:
            for k in range(self.dim[TIME]):
                u_fenics[k].x.array[:] = u[:,k]
            u = u_fenics
        
        if as_matrix is not None:
            if as_matrix:
                u = u.reshape(self.dim[SPACE], self.dim[TIME])
            else:
                u = u.reshape(-1)
                
        return u
    
        

def free_dofs(dim: int, dofs: list[np.ndarray], warn: bool = True) -> np.ndarray[bool]:
    """Determine free dofs from a list of affine Dirichlet boundary conditions.
    Args:
        dim:
            The total number of dofs in the function space.
        dofs:
            List of arrays containing the indices of the restricted dofs.
        warn:
            If True, a warning is issued if there are conflicting boundary conditions (i.e. multiple boundary conditions restricting the same dof). Note that in this case, the behavior is undefined, i.e. it may lead to unexpected results.
    Returns:
        A boolean array of length equal to the number of dofs in the function space, where True indicates a free dof and False indicates a restricted dof (i.e. restricted by at least one boundary condition in the list).
    """
    
    if len(dofs) == 0:
        return np.ones(dim, dtype=bool)
    
    f_dofs = np.ones(dim, dtype=bool)
    counts = np.zeros(dim, dtype=int)
    for dof_array in dofs:
        f_dofs[dof_array] = False
        counts[dof_array] += 1
    
    if warn:
        conflicts = np.argwhere(counts > 1)
        if len(conflicts) > 0:
            from warnings import warn
            warn(f"Conflicting Dirichlet BCs at dofs {conflicts.flatten()}. This may lead to unexpected results.")
    return f_dofs