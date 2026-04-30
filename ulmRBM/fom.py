"""
Full-Order Model (FOM) classes.

Classes
-------
.. autosummary::
   :toctree: generated/
   
    FOM
    GalerkinFOM
"""

from __future__ import annotations


__all__ = [
    'FOM',
    'GalerkinFOM',
]

from collections.abc import Callable
from typing import Generic

import numpy as np
from scipy.sparse.linalg import eigsh

from ulmRBM.core import (
    NO_MU, Mu, Matrix, ParametricLinear, Vector, wrap_scalar,
)
from ulmRBM.solver import IterativeSolver, Solver, wrap_solver
from ulmRBM.affine import AffineLinear, wrap_affinelinear
from ulmRBM.products import InnerProduct, OperatorInnerProduct


class FOM(Generic[Mu]):
    r"""
    Full-order Petrov-Galerkin model given by a parametric linear system of equations.
    
    .. math::
        B(\mu) u(\mu) = f(\mu)
        
    with an optional output of interest functional
    
    .. math::
        s(\mu) = l(\mu) u(\mu),
    
    where :math:`B(\mu) \in \mathbb{R}^{m \times n}` is the system matrix, :math:`f(\mu) \in \mathbb{R}^{m}` is the right-hand side and :math:`l(\mu) \in \mathbb{R}^{p \times n}` is the output of interest functional, all with affine parameter dependence, and :math:`u(\mu) \in \mathbb{R}^n` is the unknown solution vector and :math:`s(\mu)\in \mathbb{R}^p` the optional output of interest. 
    
    Typically, :math:`B(\mu)` is a discretization of a parametric operator :math:`B_\mu : U \to V'`, :math:`f(\mu)` a discretization of a parametric functional :math:`f_\mu \in V'`, and thus :math:`u(\mu)` is a discrete approximation of the solution :math:`u_\mu \in U` of the parametric operator equation :math:`B_\mu u_\mu = f_\mu` in :math:`V'`. Thereby, :math:`U` and :math:`V` denote the trial and test spaces, equipped with (possibly parameter dependent) inner products :math:`(\cdot, \cdot)_U` and :math:`(\cdot, \cdot)_V`, respectively.
    """
    
    dim: tuple[int, int, int | None]
    r"""Dimensions :math:`(m,n,p)` of the system matrix :math:`B(\mu)` and output :math:`s(\mu)`, where :math:`m` is the discrete test space dimension, :math:`n` is the discrete trial space dimension, and :math:`p` is the output dimension. If no output of interest is given, :math:`p` is ``None``."""
    
    B: AffineLinear[Mu, Matrix]
    r"""Affine decomposition of the system matrix :math:`B(\mu) = \sum_{q=1}^Q \theta_q^B(\mu) B_q`."""
    
    f: AffineLinear[Mu, Vector]
    r"""Affine decomposition of the right-hand side :math:`f(\mu) = \sum_{q=1}^{Q_f} \theta_q^f(\mu) f_q`."""
    
    l: AffineLinear[Mu, Matrix] | None
    r"""Affine decomposition of the output(s) of interest functional, if any :math:`l(\mu) = \sum_{q=1}^{Q_l} \theta_q^l(\mu) l_q`."""
        
    U: InnerProduct[Mu]
    r"""Inner product :math:`(\cdot, \cdot)_U` on the trial space :math:`U`."""
    
    V: InnerProduct[Mu]
    r"""Inner product :math:`(\cdot, \cdot)_V` on the test space :math:`V`."""
    
    solver: Solver
    r"""Solver for the linear system :math:`B(\mu) u = f(\mu)`."""
    
    _stability_fun: Callable[[Mu, FOM[Mu]],float] | None = None
    """Optional explicit function to compute the stability constant, bypassing the default eigenvalue-based computation."""

    _continuity_fun: Callable[[Mu, FOM[Mu]],float] | None = None
    """Optional explicit function to compute the continuity constant, bypassing the default eigenvalue-based computation."""
    
    _supremizer_func: Callable[[Vector, FOM[Mu]], AffineLinear[Mu, Vector]] | None = None
    r"""Internal storage if a custom supremizer function is provided."""
    
    @property
    def dim(self):
        p = self.l.shape[0] if self.l is not None else None
        return (*self.B.shape,p)
    
    @property
    def solver(self) -> Solver:
        return self._solver
    @solver.setter
    def solver(self, solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector]):
        self._solver = wrap_solver(solver)
    
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix,
                 f: AffineLinear[Mu, Vector] | Vector,
                 U: InnerProduct[Mu],
                 V: InnerProduct[Mu],
                 l: AffineLinear[Mu, Matrix] | Matrix | None = None,
                 stability: Callable[[Mu, FOM[Mu]], float] | float | None = None,
                 continuity: Callable[[Mu, FOM[Mu]], float] | float | None = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = IterativeSolver(),
                 supremizer: Callable[[Vector, FOM[Mu]], ParametricLinear[Mu, Vector]] = None):
        """
        Args:
            B:
                Affine decomposition of the system matrix.
            f:
                Affine decomposition of the right-hand side.
            U:
                Inner product on the trial space.
            V:
                Inner product on the test space.
            l:
                Optional affine decomposition of the output(s) of interest functional.
            stability:
                Optional function (with signature ``(mu,fom)``) or constant to compute the stability constant at a given parameter value. If ``None``, the constant is computed via eigenvalue problems.
            continuity:
                Optional function (with signature ``(mu,fom)``) or constant to compute the continuity constant at a given parameter value. If ``None``, the constant is computed via an eigenvalue problems.
            solver:
                Solver for the linear system. Defaults to a iterative solver.
            supremizer:
                Custom function for the supremizing operator. If ``None``, a default implementation is used. See :meth:`supremizer`.
        """
        
        if B.shape[0] != f.shape[0]:
            raise ValueError("B and f must have compatible dimensions.")
        if B.shape[1] != U.shape[0]:
            raise ValueError("B and U must have compatible dimensions.")
        if B.shape[0] != V.shape[0]:
                raise ValueError("B and V must have compatible dimensions.")
        if l is not None and l.shape[1] != U.shape[0]:
            raise ValueError("l and U must have compatible dimensions.")
        
        self.B = wrap_affinelinear(B).compress()
        self.f = wrap_affinelinear(f).compress()
        self.l = wrap_affinelinear(l).compress() if l is not None else None
        self.U = U
        self.V = V
        self.solver = solver
        self._stability_fun = wrap_scalar(stability)
        self._continuity_fun = wrap_scalar(continuity)
        self._supremizer_func = supremizer
        
    def __repr__(self):
        shape = f"({self.dim[0]}, {self.dim[1]}"
        if self.dim[2] is not None: shape += f", {self.dim[2]}"
        shape += ")"
        return f"<{self.__class__.__name__} of dimension {shape}>"
    
    def stability(self, mu: Mu) -> float:
        r"""
        Compute the stability constant at parameter value :math:`\mu`. 
        
        For Galerkin models, this is the coercivity constant:
        
        .. math::
            \beta(\mu) = \inf_{u \in U} \frac{| \langle B(\mu) u, u \rangle_{U'\times U} |}{\|u\|_U^2}
            
        For Petrov-Galerkin models, this is the inf-sup constant:
        
        .. math::
            \beta(\mu) = \inf_{u \in U} \sup_{v\in V} \frac{| \langle B(\mu) u, v \rangle_{V'\times V} |}{\|u\|_U \|v\|_V}
        
        The constant is computed via eigenvalue problems unless an explicit function
        was provided during initialization.
        
        Args:
            mu:
                Parameter value at which to compute the stability constant.
                
        Returns:
            Stability constant :math:`\beta(\mu) > 0`.
        """
        if self._stability_fun is not None:
            return self._stability_fun(mu, self)
        else:
            return self._stability(mu)
    
    def _stability(self, mu: Mu) -> float:
        eigsh_opts = {}
        if self.B.shape[0] == self.B.shape[1]:
            BVinvB = OperatorInnerProduct(self.B, self.V.dual, self.solver)
            eigsh_opts['OPinv'] = BVinvB.dual(mu)
        else:
            from warnings import warn
            warn("Default stability constant computation for non-square B is inefficient, consider providing a custom stability function.")
            BVinvB = self.V.dual.restrict(self.B(mu))
        eigsh_opts['A'] = BVinvB(mu)
        eigsh_opts['M'] = self.U(mu)
        eigsh_opts['k'] = 1
        eigsh_opts['sigma'] = 0.0
        eigsh_opts['which'] = 'LM'
        eigsh_opts['return_eigenvectors'] = False
        val = eigsh(**eigsh_opts)[0]
        return np.sqrt(val)  
            
    def continuity(self, mu: Mu) -> float:
        r"""
        Compute the continuity constant at parameter value :math:`\mu`.
        
        .. math::
            \gamma(\mu) = \sup_{u \in U} \sup_{v\in V} \frac{| \langle B(\mu) u, v \rangle_{V'\times V} |}{\|u\|_U \|v\|_V}
        
        The constant is equivalent to the operator norm of :math:`B(\mu)` and is computed
        via eigenvalue problems unless an explicit function was provided during initialization.
        
        Args:
            mu:
                Parameter value at which to compute the continuity constant.
                
        Returns:
            Continuity constant :math:`\gamma(\mu) < \infty`.
        """
        if self._continuity_fun is not None:
            return self._continuity_fun(mu, self)
        else:
            return self._continuity(mu)
    
    def _continuity(self, mu: Mu) -> float:
        eigsh_opts = {}
        BVinvB = self.V.dual.restrict(self.B(mu))
        eigsh_opts['A'] = BVinvB(mu)
        eigsh_opts['M'] = self.U(mu)
        eigsh_opts['Minv'] = self.U.dual(mu)
        eigsh_opts['k'] = 1
        eigsh_opts['which'] = 'LM'
        eigsh_opts['return_eigenvectors'] = False
        val = eigsh(**eigsh_opts)[0]
        return np.sqrt(val)
        
    def solve(self, mu: Mu) -> Vector:
        r"""
        Solve the parametric system :math:`B(\mu) u = f(\mu)` for the given parameter value.
        
        Args:
            mu:
                Parameter value at which to solve the system.
        
        Returns:
            State vector :math:`u(\mu) \in \mathbb{R}^n`.
        """
        return self.solver(self.B(mu), self.f(mu))
    
    def output(self, mu: Mu, u: Vector | None = None) -> Vector:
        r"""
        Compute the output of interest :math:`s(\mu) = l(\mu) u(\mu)` at the given parameter value.
        
        Args:
            mu:
                Parameter value at which to compute the output.
            u:
                Optional state vector to use instead of solving for :math:`u(\mu)`. If ``None``, the state is computed via :meth:`solve`.
        
        Returns:
            Output of interest :math:`s(\mu) \in \mathbb{R}^p`.
        """
        if self.l is None:
            raise ValueError("No output functional defined for this model.")
        if u is None: u = self.solve(mu)
        return self.l(mu) @ u
    
    
    def supremizer(self, u: Vector) -> ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]:
        r"""
        Computes the supremizing operator in the test space applied to the given trial vector(s).
        
        The supremizing operator :math:`S(\mu): U \to V` is the unique isomorphic operator given by
        :math:`S(\mu) := R_V^{-1} B(\mu)` where :math:`R_V : V \to V'` is the Riesz map of the test space.
        This function returns :math:`S(\mu) u` as a function of :math:`\mu`.
        
        If a custom supremizer function was provided during initialization, it is used. Otherwise, the supremizer is constructed from the system matrix and test space inner product. Thereby, if the test space inner product is parameter-independent, the supremizer :math:`S(\mu)` is affine with respect to :math:`\mu` and the result is an :class:`AffineLinear`. Otherweise, a parameter-dependend function with no additional structure is resturned.
        
        This supremizer is used for `Trial2TestROM`.
        
        Args:
            u:
                Trial vector(s) :math:`(N,)` or :math:`(N, k)` to which the supremizer is applied.
        Returns:
            Supremizer applied to :math:`u`, i.e. :math:`S(\mu) u` as a function of :math:`\mu`.
        """
        if self._supremizer_func is not None:
            return self._supremizer_func(u, self)
        else:
            return self._supremizer(u)
        
    
    def _supremizer(self, u: AffineLinear[Mu, Vector] | Vector) -> ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]:
        if self.V.is_parametric:
            return lambda mu: self.V.dual.riesz(mu, self.B(mu) @ (wrap_affinelinear(u)(mu)) )
        else:
            Bu = self.B @ u
            return Bu.apply2data(lambda Bu_q: self.V.dual.riesz(NO_MU, Bu_q))
        

class GalerkinFOM(FOM[Mu]):
    r"""
    Full-order Galerkin problem.
    
    Galerkin problems are a special case of Petrov-Galerkin problems, see :class:`FOM` for documentation, where the trial and test spaces coincide, i.e. it holds :math:`U = V` and :math:`m = n`.
    """
    
    @property
    def V(self) -> InnerProduct[Mu]:
        return self.U
    @V.setter
    def V(self, value: InnerProduct[Mu] | None):
        if value not in (self.U, None):
            raise AttributeError("Cannot set V for Galerkin models, as U and V are identical.")
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix,
                 f: AffineLinear[Mu, Vector] | Vector, 
                 U: InnerProduct[Mu],
                 l: AffineLinear[Mu, Matrix] | Matrix | None = None,
                 stability: Callable[[Mu],float] | float | None = None,
                 continuity: Callable[[Mu],float] | float | None = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = IterativeSolver()):
        """
        Args:
            B:
                Affine decomposition of the system matrix.
            f:
                Affine decomposition of the right-hand side.
            U:
                Inner product on the trial and test space.
            l:
                Optional affine decomposition of the output(s) of interest functional.
            stability:
                Optional function or constant to compute the stability constant at a given parameter value. If ``None``, the constant is computed via eigenvalue problems.
            continuity:
                Optional function or constant to compute the continuity constant at a given parameter value. If ``None``, the constant is computed via an eigenvalue problems.
            solver:
                Solver for the linear system. Defaults to a iterative solver.
        """
        super().__init__(B, f, U, U, l, stability, continuity, solver)
        
    def _stability(self, mu: Mu) -> float:
        eigsh_opts = {}
        B = self.B(mu)
        eigsh_opts['A'] = 0.5 * (B.T + B)    
        eigsh_opts['M'] = self.U(mu)
        eigsh_opts['k'] = 1
        eigsh_opts['sigma'] = 0.0
        eigsh_opts['which'] = 'LM'
        eigsh_opts['return_eigenvectors'] = False
        val = eigsh(**eigsh_opts)[0]
        return val
    
    def _continuity(self, mu: Mu) -> float:
        eigsh_opts = {}
        B = self.B(mu)
        eigsh_opts['A'] = 0.5 * (B.T + B)
        eigsh_opts['M'] = self.U(mu)
        eigsh_opts['Minv'] = self.U.dual(mu) 
        eigsh_opts['k'] = 1
        eigsh_opts['which'] = 'LM'
        eigsh_opts['return_eigenvectors'] = False
        val = eigsh(**eigsh_opts)[0]
        return val