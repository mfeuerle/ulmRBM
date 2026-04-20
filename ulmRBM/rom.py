"""
Reduced-Order Model (ROM) classes.

Reduced-Order Models
--------------------
.. autosummary::
    :toctree: generated/
    
     ROM
     Trial2TestROM
     GalerkinROM

Constant Estimators
-------------------

Options
~~~~~~~
.. autosummary::
    :toctree: generated/
     
     StabilityOptions
     ContinuityOptions
     
Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/
    
    ConstantsEstimator
    StabilityEstimator
    ContinuityEstimator
    
Stability Estimators
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/
    
    StabilityExact
    StabilityMinTheta
    
Continuity Estimators
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/
    
    ContinuityExact
    ContinuityMaxTheta
    
    
Residual Calculators
--------------------
.. autosummary::
    :toctree: generated/
    
    ResidualOptions
    ResidualCalculator
    DirectResidual
    AffineResidual
"""

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
    'ResidualCalculator',
    'ResidualOptions',
    'DirectResidual',
    'AffineResidual',
    'ROM',
    'Trial2TestROM',
    'GalerkinROM',
]

from abc import abstractmethod
from collections.abc import Callable
from typing import Generic

from enum import StrEnum, auto

import numpy as np

from ulmRBM.core import NO_MU, Mu, Matrix, Vector
from ulmRBM.solver import Solver, DirectSolver
from ulmRBM.fom import FOM, GalerkinFOM
from ulmRBM.products import InnerProduct, EuclideanInnerProduct, orthonormalize
from ulmRBM.affine import AffineLinear, wrap_affinelinear

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
    
class ResidualOptions(StrEnum):
    """Enumeration for residual dual norm computation methods for reduced-order models."""
    DIRECT = auto()
    """:class:`DirectResidual`"""
    AFFINE = auto()
    """:class:`AffineResidual`"""
    
    def get(self):
        """Return the corresponding residual calculator instance.
        
        Returns:
            Instance of the residual calculator corresponding to this option.
        """
        if self == ResidualOptions.DIRECT:
            return DirectResidual()
        elif self == ResidualOptions.AFFINE:
            return AffineResidual()
        else:
            raise ValueError(f"No residual for option {self}")

    
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


class ResidualCalculator(Generic[Mu]):
    r"""
    Base class for computing the dual norm of the full-order residual of a :class:`ROM`.
    
    Let :math:`f(\mu)` and :math:`B(\mu)` be the right-hand side and system matrix of the full-order model,
    :math:`U_{\text{basis}}` the reduced basis of the trial space, and :math:`u` the reduced solution vector.
    Then, :meth:`norm2` computes the square of the dual norm of the residual:
    
    .. math::
        \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}} u\|_{V'}^2.
        
    The residual calculater works in two phases:
    
    1. **Offline phase**: Precompute expensive data via :meth:`set_basis`, :meth:`add_basis` and :meth:`rotate_basis` to define :math:`U_{\text{basis}}`.
    2. **Online phase**: Provide fast computation of :math:`\|r(\mu; u)\|_{V'}^2` via :meth:`norm2` using the precomputed data.
    """
    
    
    fom: FOM[Mu]
    """Full-order model for which the dual norm of the residual is computed."""
    _fom: FOM[Mu] | None = None

        
    def set(self, fom: FOM[Mu] | ROM[Mu], basis: Vector = None):
        r"""
        Set the full-order model and trial basis for residual computation.
        
        Args:
            fom:
                Full-order model or reduced-order model. If a reduced-order model is provided,
                its underlying FOM and trial basis are used.
            basis:
                Trial basis :math:`U_{\text{basis}}` to use for residual computation. Only used if a full-order model is provided.
        """
        
        if isinstance(fom, ROM):
            basis = fom.U_basis
            fom = fom.fom
            
        self._set_fom(fom)
        if basis is not None:
            self.set_basis(basis)
    
    @property
    def fom(self) -> FOM[Mu]:
        if self._fom is None:
            raise ValueError("FOM has not been set, call .set(...) first.")
        return self._fom
    
    def _set_fom(self, fom: FOM[Mu]):
        if self._fom not in (None, fom):
            raise ValueError("FOM has already been set and cannot be changed.")
        self._fom = fom
    
    @abstractmethod
    def set_basis(self, basis: Vector):
        r"""
        Set the reduced trial basis :math:`U_{\text{basis}}` for residual computation.
        """
        ...
    
    @abstractmethod
    def add_basis(self, basis: Vector):
        r"""
        Extend the reduced trial basis :math:`U_{\text{basis}}` for residual computation.
        """
        ...
    
    @abstractmethod
    def rotate_basis(self, rotation: np.ndarray):
        r"""
        Rotate the reduced trial basis :math:`U_{\text{basis}}` for residual computation, i.e. replace :math:`U_{\text{basis}}` by :math:`U_{\text{basis}} \cdot \text{rotation}`.
        """
        ...
        
    @abstractmethod
    def norm2(self, mu: Mu, u: Vector) -> float:
        r"""
        Compute the squared dual norm of the residual at parameter :math:`\mu` for the reduced solution :math:`u`, i.e.
        
        .. math::
            \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}} u\|_{V'}^2
            
        where :math:`f(\mu)` and :math:`B(\mu)` are the right-hand side and system matrix of the full-order model,
        :math:`U_{\text{basis}}` the reduced basis of the trial space, and :math:`V'` the dual space of the test space.
        """
        ...
        
class DirectResidual(ResidualCalculator[Mu]):
    r"""
    Direct computation of the dual norm of the full-order residual
    
    .. math::
        \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}} u\|_{V'}^2
            
    using exactly this formula.     
    Thereby, :math:`f(\mu)` and :math:`B(\mu)` are the right-hand side and system matrix of the full-order model,
    :math:`U_{\text{basis}}` the reduced basis of the trial space, and :math:`V'` the dual space of the test space.
    
    .. note::
        This class is not online-efficient, as it requires full-order operations.
    """
    
    basis: Vector = None
    """Reduced trial basis :math:`U_{\text{basis}}` for residual computation."""
    
    def set_basis(self, basis: Vector):
        self.basis = basis
        
    def add_basis(self, basis: Vector):
        if self.basis is None:
            self.set_basis(basis)
        else:
            self.set_basis(np.hstack((self.basis, basis)))
        
    def rotate_basis(self, rotation: np.ndarray):
        self.set_basis(self.basis @ rotation)
        
    def norm2(self, mu: Mu, u: Vector) -> float:
        f"""{ResidualCalculator.norm2.__doc__}
    
        ... note::
            This method is not online-efficient, as it requires full-order operations.
        """
        f = self.fom.f(mu)
        Bu = self.fom.B(mu) @ (self.basis @ u)
        return self.fom.V.dual.inner(mu, f -  Bu)
    
class AffineResidual(ResidualCalculator[Mu]):
    r"""
    Online-efficient computation of the dual norm of the full-order residual
    
    .. math::
        \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}} u\|_{V'}^2
        
    by preccomputing all full-order operations before calling :meth:`norm2`.
    Thereby, :math:`f(\mu)` and :math:`B(\mu)` are the right-hand side and system matrix of the full-order model,
    :math:`U_{\text{basis}}` the reduced basis of the trial space, :math:`u` a element of the reduced trial space and :math:`V'` the dual space of the test space.
    
    The online-efficient computation relies on a affine decomposition of the residual norm:
    
    .. math::
        \|r(\mu; u)\|_{V'}^2 = \sum_{i,j=0}^{Q_r-1} \theta^r_i(\mu; u)\, \theta^r_j(\mu; u)\, R_{ij}
    
    where :math:`R_{ij} = (r_i, r_j)_{V'}` is the precomputed Gram matrix and :math:`r_i` and :math:`\theta^r_i(\mu; u)` are the affine decomposition of the residual :math:`r(\mu; u) = f(\mu) - B(\mu) U_{\text{basis}} u`, i.e.:
    
    .. math::
        r(\mu; u) = \sum_{i=0}^{Q_r-1} \theta^r_i(\mu; u)\, r_i
    
    with :math:`Q_r = Q_f + N \cdot Q_B`, with :math:`N` denoting the number of basis vectors in :math:`U_{\text{basis}}`, and:
    
    - for :math:`q = 0, \ldots, Q_f-1` set :math:`\theta^r_q(\mu; u) := \theta^f_q(\mu)` and :math:`r_q := R_{V'}f_q` (from RHS)
    - for :math:`i = 0, \ldots, N-1` and :math:`j = 0, \ldots, Q_B-1` with :math:`q := Q_f + i + j\cdot N` set :math:`\theta^r_q(\mu; u) := - u_i \theta_j^B(\mu)` and :math:`r_q := R_{V'} B_j \phi_i` (from system matrix and reduced basis)
    
    where :math:`R_{V'}:V'\to V` is the Riesz map of the dual space of the test space, :math:`U_{\text{basis}} = (\phi_0, \ldots, \phi_{N-1})` are the basis vectors of the reduced trial space, and using the notation :math:`f(\mu) = \sum_{q=0}^{Q_f-1} \theta^f_q(\mu) f_q` and :math:`B(\mu) = \sum_{q=0}^{Q_B-1} \theta^B_q(\mu) B_q` for the affine decompositions of :math:`f(\mu)` and :math:`B(\mu)`.
    
    This allows for an online-efficient computation of the dual norm in :math:`\mathcal{O}(Q_r^2)` operations (independent of full-order dimension).
    
    .. note::
        This online-efficient computation requires that the test space inner product is parameter-independent.
    """
    
    r: Vector
    r"""Precomputed Riesz representatives of the residual components :math:`(r_0, \ldots, r_{Q_r-1}) \in \mathbb{R}^{m \times Q_r}`."""
    
    R: np.ndarray
    r"""Precomputed Gram matrix of the residual components :math:`R_{ij} = (r_i, r_j)_{V'}`, :math:`i,j=0,\ldots,Q_r-1`."""
    
    @property
    def _Qf(self) -> int:
        return len(self.fom.f)
    @property
    def _QB(self) -> int:
        return len(self.fom.B)
    

    def theta(self, mu: Mu, u: Vector) -> np.ndarray:
        r"""
        Compute the affine coefficients for the residual decomposition.
        
        Args:
            mu:
                Parameter value.
            u:
                Reduced solution coefficients :math:`(N,)`.
                
        Returns:
            Affine coefficients :math:`[\theta_r^0, \ldots, \theta_r^{Q_r-1}]` of shape :math:`(Q_r,)`.
        """
        theta_f = np.array([theta(mu) for theta in self.fom.f.theta])
        theta_B = np.array([theta(mu) for theta in self.fom.B.theta])
        
        return np.concatenate((theta_f.reshape(-1), np.outer(-u, theta_B).reshape(-1)))
    
    
    def _set_fom(self, fom: FOM[Mu]):
        r"""
        Initialize the affine residual structures and precomputes the the first part of the residual decomposition that only depends on the right-hand side (and not the system matrix and reduced basis).
        """
        if fom.V.is_parametric:
            raise ValueError("OnlineResidual only supports FOMs with parameter-independent test space inner product.")
        super()._set_fom(fom)
        self.r = self.fom.V.dual.riesz(NO_MU, np.column_stack(self.fom.f.data))
        self.R = self.fom.V.inner(NO_MU, self.r)
        
    
    def set_basis(self, basis: Vector):
        f"""
        {ResidualCalculator.set_basis.__doc__}
        
        Precomputes the second part of the residual decomposition that depends on the system matrix and the reduced basis.
        """
        self.r = self.r[:, :self._Qf]
        self.R = self.R[:self._Qf, :self._Qf]
        self.add_basis(basis)
        
    def add_basis(self, basis: Vector):
        f"""
        {ResidualCalculator.add_basis.__doc__}
        
        Extends and precomputes the second part of the residual decomposition that depends on the system matrix and the reduced basis.
        """
        m = self.fom.dim[0]
        r_new = np.empty((m, basis.shape[1], self._QB))
        for q, (theta_q, B_q) in enumerate(self.fom.B):
            r_new[:,:,q] = self.fom.V.dual.riesz(NO_MU, B_q @ basis)
        r_new = r_new.reshape((m, -1))  # combine all basis vectors
        
        r_old = self.r
        
        R_old_old = self.R
        R_old_new = self.fom.V.inner(NO_MU, r_old, r_new)
        R_new_new = self.fom.V.inner(NO_MU, r_new)
        
        self.r = np.hstack([r_old, r_new])
        self.R = np.block([[R_old_old,   R_old_new],
                           [R_old_new.T, R_new_new]])

    def rotate_basis(self, rotation: np.ndarray):
        rotation = np.kron(rotation, np.eye(self._QB))
        self.r[:, self._Qf:] = self.r[:, self._Qf:] @ rotation
        self.R[self._Qf:, :] = rotation.T @ self.R[self._Qf:, :]
        self.R[:, self._Qf:] = self.R[:, self._Qf:] @ rotation
        
    def norm2(self, mu: Mu, u: Vector) -> float:
        f"""{ResidualCalculator.norm2.__doc__}
    
        ... note::
            This method is online-efficient, as it utilizes a precomputed affine decomposition of the residual.
        """
        theta = self.theta(mu, u)
        return np.abs(theta.T @ self.R @ theta) # ensure non-negativity to avoid NaN due to numerical errors





class ROM(FOM[Mu]):
    r"""
    Reduced-order model given by a Petrov-Galerkin projection of a full-order model.
    
    Constructs a reduced-order model by projecting the full-order model onto
    low-dimensional trial and test spaces :math:`U_N` and :math:`V_N` spanned by basis matrices. The reduced
    system has the form:
    
    .. math::
        B_N(\mu) u_N = f_N(\mu)
    
    where :math:`B_N(\mu) = V_N^T B(\mu) U_N` and :math:`f_N(\mu) = V_N^T f(\mu)`,
    with :math:`U_N \in \mathbb{R}^{n \times N}` the reduced trial basis and
    :math:`V_N \in \mathbb{R}^{m \times M}` the reduced test basis.
    Then, :math:`u_N \in \mathbb{R}^N` is the reduced solution vector and :math:`U_N u_N \in \mathbb{R}^n`
    the corresponding approximation in the full-order solution.
    The reduced test space depends affinely on the parameter :math:`\mu`, while the trial space is parameter-independent.
    """
    
    #Extension to affine trial space should be easy be extending the functionallity of ResidualCalculator and reusing existing code for the test space.
    
    fom: FOM[Mu]
    """Underlying full-order model."""
    U_basis: Vector
    """:math:`(n,N)` trial space basis matrix, where :math:`N` is the dimension of the reduced trial space."""
    V_basis: AffineLinear[Mu, Vector]
    """:math:`(m,M)` test space basis matrix, where :math:`M` is the dimension of the reduced test space."""
    estimate_fom_stability: StabilityEstimator[Mu]
    r"""Estimator :math:`\beta_{\text{LB}}(\mu)` for the FOM stability constant, required for error estimation."""
    estimate_fom_continuity: ContinuityEstimator[Mu]
    r"""Estimator :math:`\gamma_{\text{UB}}(\mu)` for the FOM continuity constant."""
    residual: ResidualCalculator[Mu]
    r"""Calculator for the dual norm of the residual, required for error estimation."""
    
    _U_basis: Vector = None
    _V_basis: AffineLinear[Mu, Vector] = None
        
    @property
    def U_basis(self) -> Vector:
        return self._U_basis
    @U_basis.setter
    def U_basis(self, value: Vector):
        self._U_basis = value
        self.U = self.fom.U.restrict(value)
        
    @property
    def V_basis(self) -> AffineLinear[Mu, Vector]:
        return self._V_basis
    @V_basis.setter
    def V_basis(self, value: AffineLinear[Mu, Vector]):
        self._V_basis = value
        self.V = self.fom.V.restrict(value)
        
    @property
    def estimate_fom_continuity(self) -> ContinuityEstimator[Mu]:
        if self._estimate_fom_continuity is None:
            raise RuntimeError("Set 'estimate_fom_continuity' before accessing it.")
        return self._estimate_fom_continuity
    @estimate_fom_continuity.setter
    def estimate_fom_continuity(self, estimator: ContinuityEstimator[Mu] | None):
        if estimator is not None:
            estimator.fom = self.fom
        self._estimate_fom_continuity = estimator
        
    @property
    def estimate_fom_stability(self) -> StabilityEstimator[Mu]:
        if self._estimate_fom_stability is None:
            raise RuntimeError("Set 'estimate_fom_stability' before accessing it.")
        return self._estimate_fom_stability
    @estimate_fom_stability.setter
    def estimate_fom_stability(self, estimator: StabilityEstimator[Mu] | None):
        if estimator is not None:
            estimator.fom = self.fom
        self._estimate_fom_stability = estimator
        
    @property
    def residual(self) -> ResidualCalculator[Mu]:
        if self._residual is None:
            raise RuntimeError("Set 'residual' before accessing it.")
        return self._residual
    @residual.setter
    def residual(self, residual: ResidualCalculator[Mu] | None):
        if residual is not None:
            residual.set(self)
        self._residual = residual
        
    @property
    def dim(self):
        return (0 if self.U_basis is None else self.U_basis.shape[1], 
                0 if self.V_basis is None else self.V_basis.shape[1])

    
    def __init__(self, 
                 fom: FOM[Mu],
                 U_basis: Vector = None,
                 V_basis: AffineLinear[Mu, Vector] = None,
                 stability: StabilityEstimator[Mu] | StabilityOptions = None,
                 continuity: ContinuityEstimator[Mu] | ContinuityOptions = None,
                 residual: ResidualCalculator[Mu] | ResidualOptions = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = DirectSolver()):
        r"""
        Args:
            fom:
                Full-order model to reduce.
            U_basis:
                Initial trial space basis :math:`(n, N)`. If ``None``, starts with empty basis.
            V_basis:
                Initial test space basis :math:`(m, M)`. If ``None``, starts with empty basis.
            stability:
                Estimator for the FOM stability constant. Required for online-efficient residual-based error estimation, see :meth:`estimate_error`. If ``None``, defaults to :class:`StabilityMinTheta`.
            continuity:
                Estimator for the FOM continuity constant. If ``None``, defaults to :class:`ContinuityExact`.
            residual:
                Calculator for the dual norm of the residual. Required for residual-based error estimation, see :meth:`estimate_error`. If ``None``, defaults to :class:`AffineResidual` if the FOM's test space inner product is parameter-independent, otherwise to :class:`DirectResidual`.
            solver:
                Solver for the reduced linear system. Defaults to :class:`DirectSolver`.
        """
        
        self.fom = fom
        
        if stability is None: stability = StabilityOptions.MIN_THETA
        if not isinstance(stability, StabilityEstimator):
            stability = StabilityOptions(stability).get()
        self.estimate_fom_stability = stability
                
        if continuity is None: continuity = ContinuityOptions.EXACT
        if not isinstance(continuity, ContinuityEstimator):
            continuity = ContinuityOptions(continuity).get()
        self.estimate_fom_continuity = continuity
        
        if residual is None:
            if fom.V.is_parametric:
                residual = ResidualOptions.DIRECT
                from warnings import warn
                warn("FOM has parameter-dependent test space inner product, using 'DirectResidual' for residual computation. Thus, error estimation IS NOT online-efficient.")
            else:
                residual = ResidualOptions.AFFINE
        if not isinstance(residual, ResidualCalculator):
            residual = ResidualOptions(residual).get()
        self.residual = residual
        
        self.solver = solver
        self.add_basis(U_basis, V_basis)
    
        
    def _assemble_rom(self):
        self.B = None
        self.f = None
        if self.V_basis is not None:
            self.f = self.V_basis.T @ self.fom.f
            if self.U_basis is not None:
                self.B = self.V_basis.T @ self.fom.B @ self.U_basis
                
    
    def add_basis(self, U_basis: Vector | None = None, V_basis: AffineLinear[Mu, Vector] | None = None):
        """
        Extend the reduced basis with new basis vectors.
        
        Adds new basis vectors to the reduced trial and test space bases, updating the reduced system matrices and residual data accordingly.
        
        Args:
            U_basis:
                New trial basis vectors :math:`(n, k)` to append.
            V_basis:
                New test basis vectors :math:`(m, l)` to append.
        """
        if U_basis is not None:
            self._add_basis_U(U_basis)
        if V_basis is not None:
            self._add_basis_V(V_basis)
        self._assemble_rom()
        
    def _add_basis_U(self, basis: Vector):
        if basis.ndim == 1: basis = basis.reshape(-1,1)
        if basis.shape[0] != self.fom.dim[1]:
            raise ValueError("Basis vector has incompatible dimension.")
        if self.U_basis is None:
            self.U_basis = basis
        else:
            self.U_basis = np.hstack([self.U_basis, basis])
        self.residual.add_basis(basis)
        
    def _add_basis_V(self, basis: AffineLinear[Mu, Vector]):
        if len(basis.shape) == 1: 
            basis = basis.apply2data(lambda dq: dq.reshape(-1,1))
        if basis.shape[0] != self.fom.dim[0]:
            raise ValueError("Basis vector has incompatible dimension.")
        if self.V_basis is None:
            V_basis = basis
        else:
            V_basis = AffineLinear()
            for ((theta, V_old), V_new) in zip(self.V_basis, basis.data):
                V_basis += [(theta, np.hstack([V_old, V_new]))]
        self.V_basis = V_basis
    
    def orthonormalize(self, full: bool = False, U: InnerProduct[Mu] | None = None, V: InnerProduct[Mu] | None = None):
        r"""
        Orthonormalize the trial and test space bases with respect to specified inner products.
        
        Performs orthonormalization using a parameter-independent
        inner product. This improves numerical stability and ensures well-conditioned
        reduced system matrices.
        
        Args:
            full:
                If ``True``, the residual calculation is performed from scratch, not using the rotation. Further, the test space basis is also computed from scratch instead of being rotated. Further, this option is passed to :func:`orthonormalize`. This is more stable but expensive. 
                If ``False``, update existing residual data and test-space data incrementally. This is faster but might accumulate errors over time.
            U:
                Parameter-independent inner product for trial space orthonormalization.
                If ``None`` and :attr:`fom.U` is parameter-independent, uses :attr:`fom.U`.
                Otherwise uses Euclidean inner product.
            V: 
                Parameter-independent inner product for test space orthonormalization.
                If ``None`` and :attr:`fom.V` is parameter-independent, uses :attr:`fom.V`.
                Otherwise uses Euclidean inner product.
        """
        if self.fom.U.is_parametric:
            U_default = EuclideanInnerProduct(self.fom.U.shape[0])
        else:
            U_default = self.fom.U
            
        if self.fom.V.is_parametric:
            V_default = EuclideanInnerProduct(self.fom.V.shape[0])
        else:
            V_default = self.fom.V
            
        self._orthonormalize_U(full, U, U_default)
        self._orthonormalize_V(full, V, V_default)
        self._assemble_rom()
        
    def _orthonormalize_U(self, full: bool, U: InnerProduct[Mu] | None, U_default: InnerProduct[Mu]) -> np.ndarray:
        if U is None: U = U_default                
        U_basis_orth, Q = orthonormalize(self.U_basis, U, full)
        self.U_basis = U_basis_orth
        if full:
            self.residual.set_basis(self.U_basis)
        else:
            self.residual.rotate_basis(Q)
        return Q
    
    def _orthonormalize_V(self, full: bool, V: InnerProduct[Mu] | None, V_default: InnerProduct[Mu]) -> np.ndarray:
        if V is None: V = V_default                
        V_basis_orth, Q = orthonormalize(self.V_basis, V, full)
        self.V_basis = V_basis_orth
        return Q
    
    
    def error(self, mu: Mu, u: Vector = None, u_fom: Vector = None) -> float:
        r"""
        Compute the true error between reduced and full-order solutions.
        
        Computes :math:`\|u(\mu) - U_N u_N(\mu)\|_U` where :math:`u(\mu)` is the
        full-order solution and :math:`U_N u_N(\mu)` is the reconstructed reduced solution.
        
        Args:
            mu:
                Parameter value at which to compute the error.
            u:
                Reduced-order solution vector :math:`(N,)`. If ``None``, computed via :meth:`solve`.
            u_fom:
                Full-order solution vector :math:`(n,)`. If ``None``, computed via :meth:`fom.solve`.
        
        Returns:
            True error :math:`\|u(\mu) - U_N u_N(\mu)\|_U`.
        """
        if u is None: u = self.solve(mu)
        if u_fom is None: u_fom = self.fom.solve(mu)
        return self.fom.U.norm(mu, u_fom - self.reconstruct(u))
    
    def estimate_error(self, mu: Mu, u: Vector = None) -> float:
        r"""
        Estimate the error using residual-based a posteriori error bound.
        
        Computes the error bound:
        
        .. math::
            \Delta_N(\mu) := \frac{\|f(\mu) - B(\mu) U_N u(\mu)\|_{V'}}{\beta_{\text{LB}}(\mu)}
        
        which provides an upper bound for :math:`\|u^*(\mu) - U_N u(\mu)\|_U`, where :math:`u^*(\mu)`
        is the (unknown) full-order solution, :math:`u` is a reduced-order solution, and :math:`\beta_{\text{LB}}(\mu)` is a lower bound for the stability constant of the full-order model. If the stability and continuity constants of the FOM are
        both equal to one (independent of :math:`\mu`), this bound is the true error.
        
        Args:
            mu:
                Parameter value at which to estimate the error.
            u:
                Reduced-order solution vector :math:`(N,)`. If ``None``, computed via :meth:`solve`.
        
        Returns:
            Error estimate :math:`\Delta_N(\mu)`.
            
        .. note::
            Due to taking a square root, the error estimator looses around 
        """
        if u is None: u = self.solve(mu)
        return np.sqrt(self.residual.norm2(mu, u)) / self.estimate_fom_stability(mu)
    
    def reconstruct(self, u: Vector) -> Vector:
        r"""
        Reconstruct the full-order solution from reduced coefficients.
        
        Computes :math:`U_N u_N` to obtain the full-order representation.
        
        Args:
            u:
                Reduced-order solution coefficients :math:`(N,)` or :math:`(N,k)`.
        
        Returns:
            Reconstructed full-order solution :math:`(n,)` or :math:`(n,k)`.
        """
        return self.U_basis @ u
    
    
    def stability(self, mu: Mu) -> float: return super().stability(mu)
    stability.__doc__ = f"""
        {FOM.stability.__doc__}
        
        .. note::
            This computes the stability constant of the reduced-order model. To get an estimate
            of the full-order model stability constant, use :attr:`estimate_fom_stability`.
        """
    
    def continuity(self, mu: Mu) -> float: return super().continuity(mu)
    continuity.__doc__ = f"""
        {FOM.continuity.__doc__}
        
        .. note::
            This computes the continuity constant of the reduced-order model. To get an estimate
            of the full-order model continuity constant, use :attr:`estimate_fom_continuity`.
        """

    
    
class Trial2TestROM(ROM[Mu]):
    r"""
    Reduced-order Petrov-Galerkin model with explicit trial to test space realation.
    
    In this reduced-order model, the test space is given by :math:`V_N := T U_N` where
    :math:`U_N` is the reduced trial spaceand :math:`T : U \to V` is a provided trial-to-test mapping function. Typically, :math:`T` is the supremizing operator of the underlying full-order model.
    For a documentation of the general ROM formulation, see :class:`ROM`.
    """
    
    _trial2test_fun: Callable[[Vector], AffineLinear[Mu, Vector]] | None = None
    r"""Internal storage if a custom trial2test function is provided."""
    
    def __init__(self, 
                 fom: FOM[Mu],
                 U_basis: Vector = None,
                 trial2test: Callable[[Vector], AffineLinear[Mu, Vector]] = None,
                 stability: StabilityEstimator[Mu] | StabilityOptions = None,
                 continuity: ContinuityEstimator[Mu] | ContinuityOptions = None,
                 residual: ResidualCalculator[Mu] | ResidualOptions = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = DirectSolver()):
        r"""
        Args:
            fom:
                Full-order model to reduce.
            U_basis:
                Initial trial space basis :math:`(n, N)`. If ``None``, starts with empty basis.
            trial2test:
                Function mapping trial basis vectors to test basis vectors, i.e. ``V_basis = trial2test(U_basis)``. If ``None``, and the test space inner product is parameter-independent, the supremizer of the full-order model is used, which will always yield an optimally stable ROM.
            stability:
                Estimator for the FOM stability constant. Required for online-efficient residual-based error estimation. If ``None``, defaults to :class:`StabilityMinTheta`.
            continuity:
                Estimator for the FOM continuity constant. If ``None``, defaults to :class:`ContinuityMaxTheta`.
            residual:
                Calculator for the dual norm of the residual. Required for residual-based error estimation. If ``None``, defaults to :class:`AffineResidual` if the FOM's test space inner product is parameter-independent, otherwise to :class:`DirectResidual`.
            solver:
                Solver for the reduced linear system. Defaults to direct solver.
        """
        if trial2test is None and fom.V.is_parametric:
            raise ValueError("Must provide 'trial2test' for FOMs with parameter-dependent test space inner product.")
        self._trial2test_fun = trial2test
        super().__init__(fom, U_basis, None, stability, continuity, residual, solver)
        
    def trial2test(self, basis: Vector) -> AffineLinear[Mu, Vector]:
        r"""
        Map trial basis vectors :math:`U_N` to test basis vectors :math:`V_N`.
        
        Args:
            basis:
                Trial basis vectors :math:`(n, k)`.
        
        Returns:
            Test basis vectors :math:`(m, k)`.
        """
        if self._trial2test_fun is not None:
            return self._trial2test_fun(basis)
        else:
            return self.fom.supremizer(basis)
        
        
    def _add_basis_U(self, basis: Vector):
        super()._add_basis_U(basis)
        super()._add_basis_V(self.trial2test(basis))
        
    def _add_basis_V(self, basis: AffineLinear[Mu, Vector]):
        raise ValueError("Cannot add test basis vectors directly when using a trial-to-test operator.")
        
    def _orthonormalize_U(self, full: bool, U: InnerProduct[Mu] | None, U_default: InnerProduct[Mu]) -> np.ndarray:
        QU = super()._orthonormalize_U(full, U, U_default)
        if full:
            self.V_basis = self.trial2test(self.U_basis)
        else:
            self.V_basis = self.V_basis @ QU
        return QU
    
    def _orthonormalize_V(self, full: bool, V: InnerProduct[Mu] | None, V_default: InnerProduct[Mu]) -> np.ndarray:
        if V is not None:
            raise ValueError("Can not orthonormalize test space basis explicitily as it is done implicitly via the trial-to-test operator.")
        
            

class GalerkinROM(GalerkinFOM[Mu], ROM[Mu]):
    r"""
    Reduced-order Galerkin model.
    
    A Galerkin reduced-order model is a special case of Petrov-Galerkin ROM, where :math:`V_N = U_N`. See :class:`ROM` for a documentation of the general ROM formulation.
    """
        
    def __init__(self, 
                 fom: GalerkinFOM[Mu],
                 U_basis: Vector = None,
                 stability: StabilityEstimator[Mu] | StabilityOptions = None,
                 continuity: ContinuityEstimator[Mu] | ContinuityOptions = None,
                 residual: ResidualCalculator[Mu] | ResidualOptions = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = DirectSolver()):
        r"""
        Args:
            fom:
                Full-order model to reduce.
            U_basis:
                Initial trial space basis :math:`(n, N)`. If ``None``, starts with empty basis.
            stability:
                Estimator for the FOM stability constant. Required for online-efficient residual-based error estimation. If ``None``, defaults to :class:`StabilityMinTheta`.
            continuity:
                Estimator for the FOM continuity constant. If ``None``, defaults to :class:`ContinuityMaxTheta`.
            residual:
                Calculator for the dual norm of the residual. Required for residual-based error estimation. If ``None``, defaults to :class:`AffineResidual` if the FOM's test space inner product is parameter-independent, otherwise to :class:`DirectResidual`.
            solver:
                Solver for the reduced linear system. Defaults to direct solver.
        """
        if fom.V is not fom.U:
            raise ValueError("FOM must be Galerkin (i.e., have U=V) to create a GalerkinROM.")
        ROM.__init__(self, fom, U_basis, None, stability, continuity, residual, solver)
        
    
    @property
    def V_basis(self) -> AffineLinear[Mu, Vector]:
        if self.U_basis is None:
            return None
        return wrap_affinelinear(self.U_basis)
    @V_basis.setter
    def V_basis(self, value: AffineLinear[Mu, Vector]):
        if value is None:
            return
        raise ValueError("Cannot set test basis vectors directly for Galerkin models as U=V.")
    
    def _add_basis_V(self, basis: AffineLinear[Mu, Vector]):
        raise ValueError("Cannot add test basis vectors directly for Galerkin models as U=V.")
    
    def _orthonormalize_V(self, full: bool, V: InnerProduct[Mu] | None, V_default: InnerProduct[Mu]) -> np.ndarray:
        if V is not None:
            raise ValueError("Can not orthonormalize test space basis explicitily for Galerkin models as U=V.")