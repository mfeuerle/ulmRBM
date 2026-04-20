
from numbers import Number
import numpy as np

from dolfinx import fem

from ulmRBM.fenicsx import utils

from ulmRBM.core import Mu
from ulmRBM.affine import AffineList, AffineLinear, _ConstructNew

__all__ = [
    'utils',
    'AffineDirichletBC',
    'FEniCSxSpaceWithDirichletBCs',
    'free_dofs',
]

class AffineDirichletBC(AffineLinear[Mu, np.ndarray]):
    def __init__(self, space: fem.FunctionSpace, g: AffineList[Mu] | any, entities = None, dofs = None):
        """ Affine Dirichlet boundary condition on a FEniCSx function space.
        
        Args:
            space:
                FEniCSx function space on which the boundary condition is defined.
            g:
                AffineObject representing the boundary values of the Dirichlet boundary condition.
                If g is not an AffineList, it is converted to an AffineList using AffineList([1.0], [g]).
                Then, if g is (convertible to) an AffineLinear, i.e. a (affine) vector, it has to be of length equal to the number of dofs in the function space or equal to the number of dofs restricted by the boundary condition.
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
        
        
        if not isinstance(g, AffineList):
            g = AffineList([1.0], [g])
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
    

def free_dofs(bcs: list[AffineDirichletBC], warn: bool = True) -> np.ndarray[bool]:
    """Determine free dofs from a list of affine Dirichlet boundary conditions.
    Args:
        bcs:
            List of affine Dirichlet boundary conditions.
        warn:
            If True, a warning is issued if there are conflicting boundary conditions (i.e. multiple boundary conditions restricting the same dof). Note that in this case, the behavior is undefined, i.e. it may lead to unexpected results.
    Returns:
        A boolean array of length equal to the number of dofs in the function space, where True indicates a free dof and False indicates a restricted dof (i.e. restricted by at least one boundary condition in the list).
    """
    space = bcs[0].space
    dim = bcs[0].dim
    
    assert all(bc.space == space for bc in bcs)
    assert all(bc.dim == dim for bc in bcs)
    
    f_dofs = np.ones(dim, dtype=bool)
    counts = np.zeros(dim, dtype=int)
    for bc in bcs:
        f_dofs[bc.dofs] = False
        counts[bc.dofs] += 1
    
    if warn:
        conflicts = np.argwhere(counts > 1)
        if len(conflicts) > 0:
            from warnings import warn
            warn(f"Conflicting Dirichlet BCs at dofs {conflicts.flatten()}. This may lead to unexpected results.")
    return f_dofs