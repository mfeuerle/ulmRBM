from __future__ import annotations

__all__ = [
    'ThetaStability',
    'ThetaContinuity',
]

from collections.abc import Callable

import numpy as np

from ulmRBM.core import Mu
from ulmRBM.fom import FOM, GalerkinFOM

from ._interface import StabilityEstimator, ContinuityEstimator, EfficientConstantEstimator


class _ThetaBase(EfficientConstantEstimator[Mu]):
    r"""
    Min/Max-theta estimator.
    
    Not applicable to all full-order models, see notes below!
    
    For a parameter :math:`\mu`, let :math:`\sigma(\mu)` be the parameter-dependent constant of the full-order model (stability or continuity constant). Denote the affine decomposition of the full-order system matrix by :math:`B(\mu) = \sum_{q=1}^Q \theta_q(\mu) B_q` and let :math:`\sigma(\tilde\mu)` be known for some parameter :math:`\tilde\mu`.
    Then, for Galerkin problems, we have the lower and upper bounds
    
    .. math::
        \sigma(\tilde\mu)\min_{q=1,\ldots,Q} \frac{\theta_q(\mu)}{\theta_q(\tilde\mu)} =: \sigma_{\text{LB}}(\mu) \leq \sigma(\mu) \leq \sigma_{\text{UB}}(\mu) := \sigma(\tilde\mu) \max_{q=1,\ldots,Q} \frac{\theta_q(\mu)}{\theta_q(\tilde\mu)}.
        
    and for Petrov-Galerkin problems, we have
    
    .. math::
        \sigma(\tilde\mu)\min_{p,q=1,\ldots,Q,\ p\leq q} \sqrt{\frac{\theta_p(\mu)\theta_q(\mu)}{\theta_p(\tilde\mu)\theta_q(\tilde\mu)}} =: \sigma_{\text{LB}}(\mu) \leq \sigma(\mu) \leq \sigma_{\text{UB}}(\mu) := \sigma(\tilde\mu)\max_{p,q=1,\ldots,Q,\ p\leq q} \sqrt{\frac{\theta_p(\mu)\theta_q(\mu)}{\theta_p(\tilde\mu)\theta_q(\tilde\mu)}}.
        
    To enhance the quality of the bounds, one precomputes :math:`\sigma(\tilde\mu)` for several parameters :math:`\tilde\mu` using :meth:`update` and takes the largest lower bound and the smallest upper bound over these precomputed parameters :math:`\tilde\mu`. 
        
    These bounds only hold, if the operator :math:`B(\mu)` is affine semi-definite, i.e. if for Galerkin problems it holds for each :math:`q=1,\ldots,Q` that :math:`\theta_q(\mu) > 0` for all :math:`\mu` and :math:`B_q^T + B_q` is positive semi-definite, or :math:`\theta_q(\mu) < 0` for all :math:`\mu` and :math:`B_q^T + B_q` is negative semi-definite while for for Petrov-Galerkin problems it holds for each :math:`p,q=1,\ldots,Q` that :math:`\theta_p(\mu)\theta_q(\mu) > 0` for all :math:`\mu` and :math:`B_p^T V^{-1} B_q + B_q^T V^{-1} B_p` is positive semi-definite, or :math:`\theta_p(\mu)\theta_q(\mu) < 0` for all :math:`\mu` and :math:`B_p^T V^{-1} B_q + B_q^T V^{-1} B_p` is negative semi-definite, where :math:`V` is the test space inner product matrix of the full-order model.

    The implementation checks the signs of the affine coefficients, but the
    semi-definiteness of the underlying matrices must be guaranteed by the user.
    """
    
    _get_exact_constant: Callable[[Mu], float]
    
    _positive_theta: bool | None = None
    
    def _initialize(self):
        if self.fom.U.is_parametric or self.fom.V.is_parametric:
            raise ValueError("Min/Max-Theta does not support parametric inner products. This could be extended to affine products, but was not done yet.")
        self._sigmas = None
        self._thetas = None
    
    def _update(self, mu: Mu):
        if self._sigmas is None:
            self._sigmas = np.array([self._get_exact_constant(mu)])
            self._thetas = np.array(self._eval_theta(mu))
        else:
            self._sigmas = np.append(self._sigmas, self._get_exact_constant(mu))
            self._thetas = np.vstack([self._thetas, self._eval_theta(mu)])

    def upper_bound(self, mu: Mu):
        if not self.n > 0:
            raise RuntimeError("No precomputed constants available. Call 'update' method at least once.")
        theta = self._eval_theta(mu)
        max = np.max(theta.T / self._thetas.T, axis=0)
        if not isinstance(self.fom, GalerkinFOM): max = np.sqrt(max)
        return np.min(self._sigmas * max)
    
    def lower_bound(self, mu: Mu):
        if not self.n > 0:
            raise RuntimeError("No precomputed constants available. Call 'update' method at least once.")
        theta = self._eval_theta(mu)
        min = np.min(theta.T / self._thetas.T, axis=0)
        if not isinstance(self.fom, GalerkinFOM): min = np.sqrt(min)
        return np.max(self._sigmas * min)
    
    def _eval_theta(self, mu: Mu) -> np.ndarray:
        theta = np.array([theta(mu) for theta in self.fom.B.theta])
        
        if not isinstance(self.fom, GalerkinFOM):
            theta = np.outer(theta, theta)
            theta += np.triu(theta, k=1).T
            theta = theta[np.tril_indices(theta.shape[0])]
        
        if self._positive_theta is None:
            self._signs = np.sign(theta)
            
        if not np.all(self._signs == np.sign(theta)):
            raise ValueError("Min/Max-Theta only supports theta functions with fixed signs.")
        
        return theta.reshape(1,-1)
    
    
class ThetaStability(_ThetaBase[Mu], StabilityEstimator[Mu]):
    __doc__ = _ThetaBase.__doc__
    _get_exact_constant = lambda self, mu: self.fom.stability(mu)
        
class ThetaContinuity(_ThetaBase[Mu], ContinuityEstimator[Mu]):
    __doc__ = _ThetaBase.__doc__
    _get_exact_constant = lambda self, mu: self.fom.continuity(mu)
