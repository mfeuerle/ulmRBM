"""
Interfaces for stability and continuity estimators for full-order model constants.
"""

from __future__ import annotations

__all__ = [
    'ConstantEstimator',
    'StabilityEstimator',
    'ContinuityEstimator',
    'EfficientConstantEstimator',
]

from abc import abstractmethod
from typing import Generic

import numpy as np

from ulmRBM.core import Mu
from ulmRBM.fom import FOM

class ConstantEstimator(Generic[Mu]):
    r"""
    Base class for estimating stability and continuity constants for full-order models.
    
    For a parameter :math:`\mu`, let :math:`\sigma(\mu)` be the parameter-dependent constant of the full-order model (stability or continuity constant). Then, the constant estimator provides an estimate :math:`\hat\sigma(\mu) \approx \sigma(\mu)`.

    The estimator can be used with an offline/online workflow. During the
    offline stage, :meth:`update` is called for selected training parameters
    :math:`\mu` for which subclasses might precompute expensive operations. During
    the online stage, :meth:`__call__` evaluates the estimator :math:`\hat\sigma(\mu)`
    for new parameter value using only the stored data.

    Subclasses implement the actual bound logic by defining :meth:`__call__`, ``_initialize`` and ``_update``.
    """
    
    fom: FOM[Mu]
    """The full-order model for which the constant is estimated."""
    
    n: int
    """Number of offline samples used for the estimator."""
    
    _initialized: bool = False
    
    @property
    def fom(self) -> FOM[Mu]:
        return self._fom
    
    def __init__(self, fom: FOM[Mu]):
        r"""
        Args:
            fom:
                Full-order model for which to estimate the constant.
        """
        self._fom = fom
        self.n = 0
    
    def update(self, mu: Mu):
        r"""
        Update the estimator with one offline sample.

        Args:
            mu:
                Parameter value :math:`\mu` used to enrich the offline data set.
        """
        if not self._initialized:
            self._initialize()
            self._initialized = True
        self.n += 1
        self._update(mu)
        
    def _initialize(self):
        pass
        
    def _update(self, mu: Mu):
        pass
    
    @abstractmethod
    def __call__(self, mu: Mu) -> float:
        r"""
        Approximate the constant :math:`\sigma(\mu)`.

        Args:
            mu:
                Parameter value :math:`\mu`.

        Returns:
            Estimated constant :math:`\hat\sigma(\mu)`.
        """
        ...
            
    def __repr__(self):
        return f"<{self.__class__.__name__} for {repr(self.fom)}>"
        

class StabilityEstimator(ConstantEstimator[Mu]):
    r"""
    Base class for lower-bound estimators of the stability constant.

    Provides a lower bound :math:`\sigma_{\text{LB}}(\mu) \leq \sigma(\mu)` for the stability constant, which is defined by

    .. math::
        \sigma(\mu) = \inf_{u \in U \setminus \{0\}}
        \frac{\sup_{v \in V \setminus \{0\}} |\langle B(\mu)u, v \rangle_{V'\times V}|}
        {\|u\|_U\,\|v\|_V}.

    For Galerkin problems, where :math:`U = V`, this reduces to the coercivity
    constant 
    
    .. math::
        \sigma(\mu) = \inf_{u \in U \setminus \{0\}}
        \frac{|\langle B(\mu)u, u \rangle_{U'\times U}|}{\|u\|_U^2}.

    Subclasses only need to implement :meth:`lower_bound`; the default
    :meth:`__call__` returns that lower bound.
    """
    
        
    @abstractmethod
    def lower_bound(self, mu: Mu) -> float:
        r"""
        Lower bound for the stability constant :math:`\sigma(\mu)`.

        Args:
            mu:
                Parameter value :math:`\mu`.

        Returns:
            Lower bound :math:`\sigma_{\text{LB}}(\mu)`.
        """
        ...
        
    def __call__(self, mu: Mu) -> float:
        rf"""{StabilityEstimator.lower_bound.__doc__}"""
        return self.lower_bound(mu)
    

class ContinuityEstimator(ConstantEstimator[Mu]):
    r"""
    Base class for upper-bound estimators of the continuity constant
    :math:`\sigma(\mu)`.

    Provides an upper bound :math:`\sigma_{\text{UB}}(\mu) \geq \sigma(\mu)` for the continuity constant. 
    The continuity constant is the operator norm of the bilinear form induced
    by :math:`B(\mu)`:

    .. math::
        \sigma(\mu) = \|B(\mu)\|_{\mathcal L(U,V')} =
        \sup_{u \in U \setminus \{0\}}\sup_{v \in V \setminus \{0\}}
        \frac{|\langle B(\mu)u, v \rangle_{V'\times V}|}{\|u\|_U\,\|v\|_V}.

    Subclasses only need to implement :meth:`upper_bound`; the default
    :meth:`__call__` returns that upper bound.
    """
    
    
    @abstractmethod
    def upper_bound(self, mu: Mu) -> float:
        r"""
        Upper bound for the continuity constant :math:`\sigma(\mu)`.

        Args:
            mu:
                Parameter value :math:`\mu`.

        Returns:
            Upper bound :math:`\sigma_{\text{UB}}(\mu)`.
        """
        ...
        
    def __call__(self, mu: Mu) -> float:
        rf"""{ContinuityEstimator.upper_bound.__doc__}"""
        return self.upper_bound(mu)
    
    
class EfficientConstantEstimator(ConstantEstimator[Mu]):
    r"""
    Base class for estimators that provide both lower and upper bounds.

    For a parameter :math:`\mu`, let :math:`\sigma(\mu)` be the parameter-dependent constant of the full-order model (stability or continuity constant). Then, the efficient constant estimator provides not only an estimate :math:`\hat\sigma(\mu) \approx \sigma(\mu)` but also bounds on the constant
    
    .. math::
        \sigma_{\text{LB}}(\mu) \leq \sigma(\mu) \leq \sigma_{\text{UB}}(\mu).
        
    This allows to quantify the approximation quality of the estimator via
    
    .. math::
        \|\sigma(\mu) - \hat\sigma(\mu)\| \leq \|\sigma_{\text{UB}}(\mu) - \sigma_{\text{LB}}(\mu)\|.
    
    and
    
    .. math::
        \frac{\|\sigma(\mu) - \hat\sigma(\mu)\|}{\|\sigma(\mu)\|} \leq \frac{\|\sigma_{\text{UB}}(\mu) - \sigma_{\text{LB}}(\mu)\|}{\|\sigma_{\text{LB}}(\mu)\|}.

    Subclasses implement :meth:`__call__`, :meth:`lower_bound` and :meth:`upper_bound`.
    """
    
    @abstractmethod
    def upper_bound(self, mu: Mu) -> float:
        r"""
        Upper bound for the constant :math:`\sigma(\mu)`.

        Args:
            mu:
                Parameter value :math:`\mu`.

        Returns:
            Upper bound :math:`\sigma_{\text{UB}}(\mu)`.
        """
        ...
        
    @abstractmethod
    def lower_bound(self, mu: Mu) -> float:
        r"""
        Lower bound for the constant :math:`\sigma(\mu)`.

        Args:
            mu:
                Parameter value :math:`\mu`.

        Returns:
            Lower bound :math:`\sigma_{\text{LB}}(\mu)`.
        """
        ...

    def error_bound(self, mu: Mu, rel: bool = True, abs: bool = False) -> float | tuple[float, float]:
        r"""
        Error bound for the approximation of the constant :math:`\sigma(\mu)`.

        Absolute error bound:
        
        .. math::
            \|\sigma(\mu) - \hat\sigma(\mu)\| \leq \|\sigma_{\text{UB}}(\mu) - \sigma_{\text{LB}}(\mu)\|.
        
        Relative error bound:
        
        .. math::
            \frac{\|\sigma(\mu) - \hat\sigma(\mu)\|}{\|\sigma(\mu)\|} \leq \frac{\|\sigma_{\text{UB}}(\mu) - \sigma_{\text{LB}}(\mu)\|}{\|\sigma_{\text{LB}}(\mu)\|}.
        
        Args:
            mu:
                Parameter value :math:`\mu`.
            rel:
                If ``True``, return a bound on the relative error.
            abs:
                If ``True``, return a bound on the absolute error.

        Returns:
            Either the relative error bound, the absolute error bound, or a tuple with the first being the relative error bound and the second being the absolute error bound, depending on the flags.
        """
        lb = self.lower_bound(mu)
        ub = self.upper_bound(mu)
        abs_err = np.abs(ub-lb)
        with np.errstate(divide='ignore'):
            rel_err = abs_err/np.abs(lb)
        
        if rel and abs: return rel_err, abs_err
        if rel: return rel_err
        if abs: return abs_err
    


