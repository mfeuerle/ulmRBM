"""
Classic (full-order) models for parametric linear systems of equations.
"""

from __future__ import annotations


__all__ = [
    'Model',
    'FOM',
    'GalerkinFOM',
]

from collections.abc import Callable

from ulmRBM.core import (
    NO_MU, Mu, Matrix, ParametricLinear, Vector
)
from ulmRBM.solver import Solver, IterativeSolver, wrap_solver
from ulmRBM.affine import AffineLinear, wrap_affinelinear
from ulmRBM.products import InnerProduct

from ._operators import ParametricOperator, ParametricGalerkinOperator
    
    
class Model(ParametricOperator[Mu]):
    r"""Model given by a parametric linear system of equations.
    
    Extends `ParametricOperator` by adding a right-hand side and an optional output of interest functional, thus representing a parametric linear system of the form
    
    .. math::
        B(\mu) u(\mu) = f(\mu)
        
    with an optional output of interest functional
    
    .. math::
        s(\mu) = l(\mu) u(\mu),
    
    where :math:`B(\mu) \in \mathbb{R}^{m \times n}` is the system matrix, :math:`f(\mu) \in \mathbb{R}^{m}` is the right-hand side and :math:`l(\mu) \in \mathbb{R}^{p \times n}` is the output of interest functional, all with affine parameter dependence, and :math:`u(\mu) \in \mathbb{R}^n` is the unknown solution vector and :math:`s(\mu)\in \mathbb{R}^p` the optional output of interest.
    """
    
    shape: tuple[int, int, int]
    r"""``(m, n, p)``, shape of the system matrix :math:`B(\mu)` and output :math:`s(\mu)`, where :math:`m` is the test space dimension, :math:`n` is the trial space dimension, and :math:`p` is the output dimension."""

    f: AffineLinear[Mu, Vector]
    r"""Affine decomposition of the right-hand side :math:`f(\mu) = \sum_{q=1}^{Q_f} \theta_q^f(\mu) f_q`."""
    
    l: AffineLinear[Mu, Vector] | None
    r"""Affine decomposition of the output(s) of interest functional :math:`l(\mu) = \sum_{q=1}^{Q_l} \theta_q^l(\mu) l_q`. Might be ``None``, if no output of interest is given."""
    
    @property
    def shape(self):
        p = self.l.shape[0] if self.l is not None else 0
        return (*super().shape,p)
    
    
    def __init__(self, f: AffineLinear[Mu, Vector] | Vector, l: AffineLinear[Mu, Vector] | Vector | None = None, solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None):
        r"""This class is not not designt to instantiate directly.
        
        Call this constructor only from child classes that also call the constructor of `ParametricOperator` prior to this constructor.
        
        Args:
            f:
                Affine decomposition of the right-hand side.
            l:
                Optional affine decomposition of the output(s) of interest functional.
            solver:
                Solver for the linear system. Defaults to `IterativeSolver`.
        """
        if self.B.shape[0] != f.shape[0]:
            raise ValueError("B and f must have compatible dimensions.")
        if l is not None and l.shape[1] != self.U.shape[0]:
            raise ValueError("l and U must have compatible dimensions.")
        
        if solver is None: solver = IterativeSolver()
    
        self.f = wrap_affinelinear(f).compress()
        self.l = wrap_affinelinear(l).compress() if l is not None else None
        self._solver = wrap_solver(solver)
    
    def solve(self, mu: Mu, u0=None) -> Vector:
        r"""
        Solve the parametric system :math:`B(\mu) u = f(\mu)` for the given parameter value.
        
        Args:
            mu:
                Parameter value at which to solve the system.
            u0:
                Optional initial guess for iterative solvers.
        
        Returns:
            State vector :math:`u(\mu) \in \mathbb{R}^n`.
        """
        return self._solver(self.B(mu), self.f(mu), u0)
    
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
    
    
class FOM(Model[Mu], ParametricOperator[Mu]):
    r"""Petrov-Galerkin full-order model given by a parametric linear system of equations.
    
    Extends `ParametricOperator` by adding a right-hand side and an optional output of interest functional, thus representing a parametric linear system of the form
    
    .. math::
        B(\mu) u(\mu) = f(\mu)
        
    with an optional output of interest functional
    
    .. math::
        s(\mu) = l(\mu) u(\mu),
    
    where :math:`B(\mu) \in \mathbb{R}^{m \times n}` is the system matrix, :math:`f(\mu) \in \mathbb{R}^{m}` is the right-hand side and :math:`l(\mu) \in \mathbb{R}^{p \times n}` is the output of interest functional, all with affine parameter dependence, and :math:`u(\mu) \in \mathbb{R}^n` is the unknown solution vector and :math:`s(\mu)\in \mathbb{R}^p` the optional output of interest.
    """
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix | ParametricOperator[Mu],
                 f: AffineLinear[Mu, Vector] | Vector,
                 U: InnerProduct[Mu] = None,
                 V: InnerProduct[Mu] = None,
                 l: AffineLinear[Mu, Vector] | Vector | None = None,
                 stability:  Callable[[Mu, ParametricOperator[Mu]], float] | float | str = None,
                 continuity: Callable[[Mu, ParametricOperator[Mu]], float] | float | str = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = None,
                 supremizer: Callable[[Vector, ParametricOperator[Mu]], ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]] = None):
        r"""
        Args:
            B:
                Affine decomposition of the system matrix.
            f:
                See `Model`.
            U:
                Inner product on the trial space.
            V:
                Inner product on the test space.
            l:
                See `Model`.
            stability, continuity:
                See `ParametricOperator`.
            solver:
                See `Model`.
            supremizer:
                See `ParametricOperator`.
        """
        
        if isinstance(B, ParametricOperator):
            U = U if U is not None else B.U
            V = V if V is not None else B.V
            stability  = stability if stability   is not None else B._stability_solver
            continuity = continuity if continuity is not None else B._continuity_solver
            supremizer = supremizer if supremizer is not None else B._supremizer_func
            B = B.B
        elif U is None or V is None:
            raise ValueError("Inner products U and V must be provided when B is not a ParametricOperator.")
        
        ParametricOperator.__init__(self, B, U, V, stability, continuity, supremizer)
        Model.__init__(self, f, l, solver)
        

class GalerkinFOM(Model[Mu], ParametricGalerkinOperator[Mu]):
    r"""Galerkin full-order model given by a parametric linear system of equations.
    
    Extends `ParametricGalerkinOperator` by adding a right-hand side and an optional output of interest functional, thus representing a parametric linear system of the form
    
    .. math::
        B(\mu) u(\mu) = f(\mu)
        
    with an optional output of interest functional
    
    .. math::
        s(\mu) = l(\mu) u(\mu),
    
    where :math:`B(\mu) \in \mathbb{R}^{n \times n}` is the system matrix, :math:`f(\mu) \in \mathbb{R}^{n}` is the right-hand side and :math:`l(\mu) \in \mathbb{R}^{p \times n}` is the output of interest functional, all with affine parameter dependence, and :math:`u(\mu) \in \mathbb{R}^n` is the unknown solution vector and :math:`s(\mu)\in \mathbb{R}^p` the optional output of interest.
    """
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix | ParametricGalerkinOperator[Mu],
                 f: AffineLinear[Mu, Vector] | Vector,
                 U: InnerProduct[Mu] = None,
                 l: AffineLinear[Mu, Vector] | Vector | None = None,
                 stability:  Callable[[Mu, ParametricGalerkinOperator[Mu]], float] | float | str = None,
                 continuity: Callable[[Mu, ParametricGalerkinOperator[Mu]], float] | float | str = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = None,
                 supremizer: Callable[[Vector, ParametricGalerkinOperator[Mu]], ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]] = None):
        r"""
        Args:
            B:
                Affine decomposition of the system matrix.
            f:
                See `Model`.
            U:
                Inner product on the trial space.
            V:
                Inner product on the test space.
            l:
                See `Model`.
            stability, continuity:
                See `ParametricOperator`.
            solver:
                See `Model`.
            supremizer:
                See `ParametricOperator`.
        """
        
        if isinstance(B, ParametricOperator):
            if not isinstance(B, ParametricGalerkinOperator):
                raise ValueError("For GalerkinFOM, the operator B must be a ParametricGalerkinOperator.")
            U = U if U is not None else B.U
            stability  = stability if stability   is not None else B._stability_solver
            continuity = continuity if continuity is not None else B._continuity_solver
            supremizer = supremizer if supremizer is not None else B._supremizer_func
            B = B.B
        elif U is None:
            raise ValueError("Inner product U must be provided when B is not a ParametricOperator.")
        
        ParametricGalerkinOperator.__init__(self, B, U, stability, continuity, supremizer)
        Model.__init__(self, f, l, solver)