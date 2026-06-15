"""
Exact stability and continuity estimators.
"""

from __future__ import annotations

__all__ = [
    'ExactStability',
    'ExactContinuity',
]

from ulmRBM.core import Mu

from ._interface import StabilityEstimator, ContinuityEstimator, EfficientConstantEstimator


class _ExactEstimator(EfficientConstantEstimator[Mu]):
    r"""
    Common base class for estimators that return the exact constant.
    """
    
    def upper_bound(self, mu: Mu) -> float:
        return self.__call__(mu)
    
    def lower_bound(self, mu: Mu) -> float:
        return self.__call__(mu)
    
    def error_bound(self, mu: Mu, rel: bool = True, abs: bool = False, lb: float = None, ub: float = None) -> float | tuple[float, float]:
        if rel+abs == 0: return []
        if rel+abs == 1: return 0.0
        return 0.0, 0.0
    
class ExactStability(_ExactEstimator[Mu], StabilityEstimator[Mu]):
    r"""
    Uses the exact stability constant.
    
    The stability constat estimator :math:`\hat\sigma(\mu)` is given by the exact stability constant of the full-order model.
    
    The estimator is online-efficient if the exact stability constant of the `ParametricOperator` can be evaluated fast via custom functions (e.g. if the constants are known analytically).
    """
    
    def __call__(self, mu: Mu) -> float:
        return self.B.stability(mu)
    
class ExactContinuity(_ExactEstimator[Mu], ContinuityEstimator[Mu]):
    r"""
    Uses the exact continuity constant.
    
    The continuity constat estimator :math:`\hat\sigma(\mu)` is given by the exact continuity constant of the full-order model.
    
    The estimator is online-efficient if the exact continuity constant of the `ParametricOperator` can be evaluated fast via custom functions (e.g. if the constants are known analytically).
    """
    
    def __call__(self, mu: Mu) -> float:
        return self.B.continuity(mu)