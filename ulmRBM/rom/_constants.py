
from __future__ import annotations

__all__ = [
    'ConstantsEstimator',
    'StabilityOptions',
    'StabilityEstimator',
    'StabilityExact',
    'StabilityMinTheta',
    'ContinuityOptions',
    'ContinuityEstimator',
    'ContinuityExact',
    'ContinuityMaxTheta',
]

from abc import abstractmethod
from collections.abc import Callable
from typing import Generic

from enum import StrEnum, auto

import numpy as np

from ulmRBM.core import Mu
from ulmRBM.fom import FOM

# maybe add a global flag to warn if any computations inside the rom use full-order dimensions


class StabilityOptions(StrEnum):
    """Enumeration for stability constant estimators for reduced-order models."""
    EXACT = auto()
    """:class:`StabilityExact`"""
    MIN_THETA = 'min-theta'
    """:class:`StabilityMinTheta`"""
    
    def get(self):
        """Return the corresponding stability estimator instance.
        
        Returns:
            Instance of the stability estimator corresponding to this option.
        """
        if self == StabilityOptions.EXACT:
            return StabilityExact()
        elif self == StabilityOptions.MIN_THETA:
            return StabilityMinTheta()
        else:
            raise ValueError(f"No estimator for option {self}")
    
class ContinuityOptions(StrEnum):
    """Enumeration for continuity constant estimators for reduced-order models."""
    EXACT = auto()
    """:class:`ContinuityExact`"""
    MAX_THETA = 'max-theta'
    """:class:`ContinuityMaxTheta`"""
    
    def get(self):
        """Return the corresponding continuity estimator instance.
        
        Returns:
            Instance of the continuity estimator corresponding to this option.
        """
        if self == ContinuityOptions.EXACT:
            return ContinuityExact()
        elif self == ContinuityOptions.MAX_THETA:
            return ContinuityMaxTheta()
        else:
            raise ValueError(f"No estimator for option {self}")

    
class ConstantsEstimator(Generic[Mu]):
    r"""
    Abstract base class for estimating stability and continuity constants.
    
    Constants estimators are used in reduced-order models to provide online-efficient
    approximations of the full-order model's stability and continuity constants,
    see :meth:`~FOM.stability` and :meth:`~FOM.continuity`.
    In particular, an online-efficient estimate for the stability constant is required
    for an online-efficient residual-based a posteriori error estimator.
    
    The estimators work in two phases:
    
    1. **Offline phase**: Precompute expensive data via :meth:`update` at selected parameters.
    2. **Online phase**: Provide fast estimates via :meth:`__call__` using the precomputed data.
    """
    
    _fom: FOM[Mu] | None = None
    
    def __init__(self, fom: FOM[Mu] = None):
        """
        Args:
            fom:
                Full-order model for which to estimate constants. Can be set later via the :attr:`fom` property.
        """
        
        self.fom: FOM[Mu] = fom
        """Full-order model for which constants are estimated."""
    
    @abstractmethod
    def __call__(self, mu: Mu) -> float:
        r"""
        Estimate the constant at parameter value :math:`\mu`.
        
        Args:
            mu:
                Parameter value at which to estimate the constant.
                
        Returns:
            Estimated constant value.
        """
        ...
    
    def update(self, mu: Mu):
        """
        Enrich the estimator using the parameter value to improve future estimates.
        
        The estimator precomputes or updates internal data based on the given
        parameter value, enhancing the quality of future :meth:`__call__` invocations.
        Subclasses may override this method to implement specific enrichment strategies.
        
        Args:
            mu:
                Parameter value used for enrichment.
        """
        pass
    
    
    @property
    def fom(self) -> FOM[Mu]:
        if self._fom is None:
            raise ValueError("FOM has not been set.")
        return self._fom
    
    @fom.setter
    def fom(self, fom: FOM[Mu]):
        if self._fom is not None and self._fom is not fom:
            raise ValueError("FOM has already been set and cannot be changed.")
        self._fom = fom
    


class StabilityEstimator(ConstantsEstimator[Mu]):
    """
    Namespace class for stability constant estimators.
    
    Stability estimators provide lower bounds :math:`\\beta_{\\text{LB}}(\\mu) \\leq \\beta(\\mu)` on
    the coercivity (Galerkin) or inf-sup (Petrov-Galerkin) constant
    :math:`\\beta(\\mu)` of the parametric operator :math:`B(\\mu)` of the FOM,
    see :meth:`~FOM.stability`. These lower bounds are essential for reliable
    a posteriori error estimation in reduced basis methods, as the error estimate is inversely
    proportional to the stability constant.
    """
    pass

class ContinuityEstimator(ConstantsEstimator[Mu]):
    r"""
    Namespace class for continuity constant estimators.
    
    Continuity estimators provide upper bounds :math:`\gamma_{\text{UB}}(\mu) \geq \gamma(\mu)`
    on the continuity constant :math:`\gamma(\mu)` (operator norm) of the parametric operator
    :math:`B(\mu)` of the FOM, see :meth:`~FOM.continuity`.
    """
    pass
    

class StabilityExact(StabilityEstimator[Mu]):
    """
    Exact stability constant of the FOM.
    
    This estimator delegates to the full-order model's :meth:`~FOM.stability` method,
    providing the exact stability constant.
    
    .. note::
        This estimator is not online-efficient.
    """
    def __call__(self, mu: Mu) -> float:
        return self.fom.stability(mu)
    
class ContinuityExact(ContinuityEstimator[Mu]):
    """
    Exact continuity constant of the FOM.
    
    This estimator delegates to the full-order model's :meth:`~FOM.continuity` method,
    providing the exact continuity constant.
    
    .. note::
        This estimator is not online-efficient.
    """
    def __call__(self, mu: Mu) -> float:
        return self.fom.continuity(mu)


class _MinThetaBase(Generic[Mu]):
    """
    Base class for min/max-theta based constant estimators.
    
    Implements the min/max-theta method for online-efficient constant estimation,
    which exploits the affine parameter dependence to precompute constants at
    selected parameter values during the offline phase.
    
    The method relies on the affine decomposition:
    
    .. math::
        B(\\mu) = \\sum_{q=1}^Q \\theta_q^B(\\mu) B_q
    
    and computes bounds by comparing theta coefficient ratios across parameters.
    """
    
    _m1 : Callable[[np.ndarray], float]
    """Function applied to ratio of theta coefficients (min for stability, max for continuity)."""
    
    _m2 : Callable[[list[float]], float]
    """Function applied to collect estimates (max for stability, min for continuity)."""
    
    _get_constant : Callable[[Mu], float]
    """Function to compute the exact constant at a given parameter value (stability or continuity)."""
    
    
    def __init__(self, fom: FOM[Mu] = None):
        """
        Args:
            fom:
                Full-order model for which to estimate constants.
        """
        super().__init__(fom)
        self._precomputed_thetas: list[np.ndarray] = []
        """List of precomputed theta coefficient vectors :math:`[\\theta^B(\\mu_1), \\theta^B(\\mu_2), \\ldots]` at enrichment parameters."""
        
        self._precomputed_constants: list[float] = []
        """List of precomputed exact constants :math:`[c(\\mu_1), c(\\mu_2), \\ldots]` at enrichment parameters."""
    
    def __call__(self, mu: Mu) -> float:
        if len(self._precomputed_constants) == 0:
            raise RuntimeError("No precomputed constants available. Call 'update' method at least once.")
        thetas = np.array([theta(mu) for theta in self.fom.B.theta])
        if np.any(thetas <= 0):
            raise ValueError("MinTheta estimator only supports positive theta functions.")
        
        estimates = []
        for p_thetas, p_const in zip(self._precomputed_thetas, self._precomputed_constants):
            estimates.append( p_const * self._m1(thetas/p_thetas) )
        return self._m2(estimates)
    
    def update(self, mu: Mu):
        """
        Precompute and store the exact constant and theta coefficients at the given parameter.
        
        Args:
            mu:
                Parameter value for enrichment.
        """
        thetas = np.array([theta(mu) for theta in self.fom.B.theta])
        if np.any(thetas <= 0):
            raise ValueError("MinTheta estimator only supports positive theta functions.")
        constant = self._get_constant(mu)
        self._precomputed_thetas.append(np.array(thetas))
        self._precomputed_constants.append(constant)
        
    
    # only to add the warning message:
    @property
    def fom(self) -> FOM[Mu]:
        if self._fom is None:
            raise ValueError("FOM has not been set.")
        return self._fom
    
    @fom.setter
    def fom(self, fom: FOM[Mu]):
        if self._fom is fom:
            return
        if self._fom is not None and self._fom is not fom:
            raise ValueError("FOM has already been set and cannot be changed.")
        if fom.U.is_parametric or fom.V.is_parametric:
            from warnings import warn
            warn("Min/max-theta with parametric inner products just gives just a error indicator, but not a rigorous bound. This could be extended to affine products, but was not done yet.")
        self._fom = fom
        
        

class StabilityMinTheta(_MinThetaBase[Mu], StabilityEstimator[Mu]):
    r"""
    Minimum-theta stability constant estimator.
    
    Provides online-efficient lower bounds for the stability constant by computing:
    
    .. math::
        \beta_{\text{LB}}(\mu) = \max_{k} \beta(\mu_k) \cdot \min_q \frac{\theta_q^B(\mu)}{\theta_q^B(\mu_k)}
    
    where :math:`\mu_k` are enrichment parameters where exact constants were precomputed using :meth:`update`.
    Requires all :math:`\theta_q^B(\mu) > 0` for all :math:`\mu` and :math:`q`.
    """
    _m1 = staticmethod(np.min)
    _m2 = staticmethod(np.max)
    _get_constant = lambda self, mu: self.fom.stability(mu)
    
class ContinuityMaxTheta(_MinThetaBase[Mu], ContinuityEstimator[Mu]):
    r"""
    Maximum-theta continuity constant estimator.
    
    Provides online-efficient upper bounds for the continuity constant by computing:
    
    .. math::
        \gamma_{\text{UB}}(\mu) = \min_{k} \gamma(\mu_k) \cdot \max_q \frac{\theta_q^B(\mu)}{\theta_q^B(\mu_k)}
    
    where :math:`\mu_k` are enrichment parameters where exact constants were precomputed using :meth:`update`.
    Requires all :math:`\theta_q^B(\mu) > 0` for all :math:`\mu` and :math:`q`.
    """
    _m1 = staticmethod(np.max)
    _m2 = staticmethod(np.min)
    _get_constant = lambda self, mu: self.fom.continuity(mu)