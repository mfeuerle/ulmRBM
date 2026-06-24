
from __future__ import annotations


__all__ = [
    'ResidualNormEvaluator',
    'FullResidualNormEvaluator',
    'AffineResidualNormEvaluator',
    'TimeSteppingResidualNormEvaluator',
    'FullTimeSteppingResidualNormEvaluator',
    'AffineTimeSteppingResidualNormEvaluator',
]

from abc import abstractmethod
from typing import Generic, TYPE_CHECKING

import numpy as np

from ulmRBM.core import NO_MU, Mu, Vector
from ulmRBM.affine import AffineLinear
from ulmRBM.fom import TimeSteppingSolution, PrimalDualModel

if TYPE_CHECKING:
    from ulmRBM.rom import ROM, StationaryTimeSteppingGalerkinROM


class ResidualNormEvaluator(Generic[Mu]):
    r"""
    Base class for computing the dual norm of the full-order residual of a :class:`ROM`.
    
    Let :math:`f(\mu)` and :math:`B(\mu)` be the right-hand side and system matrix of the full-order model,
    :math:`U_{\text{basis}}` the reduced basis of the trial space, see `ROM.U_basis`, and :math:`u` the reduced solution vector.
    Then, :meth:`dual_norm` computes the dual norm of the residual:
    
    .. math::
        \|r(\mu; u)\|_{V'} = \|f(\mu) - B(\mu) U_{\text{basis}} u\|_{V'}.
        
    The residual calculater works in two phases:
    
    1. **Offline phase**: Precompute expensive data :meth:`add_basis` and :meth:`rotate_basis`, where :meth:`add_basis` is called by the `ROM` to inform the residual evaluator about any expansion to `ROM.U_basis`, while :meth:`rotate_basis` is called by the `ROM` to inform the residual evaluator about any rotation of the existing basis, e.g. when the basis is orthonormalized.
    2. **Online phase**: Provide fast computation of :math:`\|r(\mu; u)\|_{V'}` and :math:`\|f(\mu)\|_{V'}` via :meth:`dual_norm` and :meth:`dual_norm_rhs` using the precomputed data.
    """
    
    online_efficient: bool
    r"""Whether the residual norm evaluator provides online-efficient computation of the dual norm of the residual."""
    
    rom: ROM[Mu]
    r"""The reduced-order model for which the residual is evaluated."""
    
    def __init__(self, rom: ROM[Mu]):
        self.rom = rom
        self._initialize()
        
        if isinstance(rom, PrimalDualModel):
            self._output = ValueError("Output norm should never be accessed for a primal-dual model, how did you even get here?")
        elif rom.fom.l is None:
            self._output = ValueError("Residual norm evaluator cannot compute output error since the FOM has no output functional.")
        else:
            try:
                self._initialize_output()
                self._output = True
            except Exception as e:
                self._output = e
                
        if rom.U_basis is not None: self.add_basis(rom.U_basis)
        
    def _initialize(self):
        r"""
        Setup the the norm evaluater.
        """
        pass
    
    def _initialize_output(self) -> bool:
        r"""
        Setup the the norm evaluater for computing the dual norm of the output functional.
        """
        
        pass
        
    def add_basis(self, basis: Vector):
        r"""
        Extend the reduced trial basis :math:`U_{\text{basis}}`.
        """
        pass
    
    def rotate_basis(self, rotation: np.ndarray):
        r"""
        Rotate the reduced trial basis :math:`U_{\text{basis}}`, i.e. replace :math:`U_{\text{basis}}` by :math:`U_{\text{basis}} \cdot \text{rotation}`.
        """
        pass
        
    @abstractmethod
    def dual_norm(self, mu: Mu, u: Vector) -> float:
        r"""
        Compute the dual norm of the residual at parameter :math:`\mu` for the reduced solution :math:`u`, i.e.
        
        .. math::
            \|r(\mu; u)\|_{V'} = \|f(\mu) - B(\mu) U_{\text{basis}} u\|_{V'}
            
        where :math:`f(\mu)` and :math:`B(\mu)` are the right-hand side and system matrix of the full-order model,
        :math:`U_{\text{basis}}` the reduced basis of the trial space, and :math:`V'` the dual space of the test space.
        """
        ...
        
    @abstractmethod
    def dual_norm_rhs(self, mu: Mu) -> float:
        r"""
        Compute the dual norm of the right-hand side at parameter :math:`\mu`, i.e.
        
        .. math::
            \|f(\mu)\|_{V'}
            
        where :math:`f(\mu)` is the right-hand side of the full-order model and :math:`V'` the dual space of the test space.
        """
        ...
    
    
    def dual_norm_output(self, mu: Mu) -> float:
        r"""
        Compute the dual norm of the output functional at parameter :math:`\mu`, i.e.
        
        .. math::
            \|l(\mu)\|_{U'}
            
        where :math:`l(\mu)` is the output functional of the full-order model and :math:`U'` the dual space of the trial space.
        """
        if self._output is True:
            self._dual_norm_output(mu)
        else:
            raise self._output
        
    @abstractmethod
    def _dual_norm_output(self, mu: Mu) -> float:
        ...
        
    
    def __repr__(self):
        return f"<{self.__class__.__name__} for {repr(self.rom)}>"
    
    

class FullResidualNormEvaluator(ResidualNormEvaluator[Mu]):
    """Default implementation that delegates all computations to the full-order model. This is not online-efficient.
    """
    
    online_efficient = False
    
    def dual_norm(self, mu: Mu, u: Vector) -> float:
        f = self.rom.fom.f(mu)
        Bu = self.rom.fom.B(mu) @ (self.rom.reconstruct(mu, u))
        return self.rom.fom.V.dual.norm(mu, f -  Bu)
    
    def dual_norm_rhs(self, mu: Mu) -> float:
        return self.rom.fom.V.dual.norm(mu, self.rom.fom.f(mu))
    
    def _dual_norm_output(self, mu: Mu) -> float:
        return self.rom.fom.U.dual.norm(mu, self.rom.fom.l(mu))
    
    
    
class AffineResidualNormEvaluator(ResidualNormEvaluator[Mu]):
    """Exploiting the affine structure of the full-order model for online-efficient residual norm evaluation. Only applicable for FOMs with parameter-independent test space inner product.
    """
    
    online_efficient = True
    
    _r: Vector
    r"""Precomputed Riesz representatives of the residual components :math:`(r_0, \ldots, r_{Q_r-1}) \in \mathbb{R}^{m \times Q_r}`."""
    
    _R: np.ndarray
    r"""Precomputed Gram matrix of the residual components :math:`R_{ij} = (r_i, r_j)_{V'}`, :math:`i,j=0,\ldots,Q_r-1`."""
    
    def _theta_r(self, mu: Mu, u: Vector | None) -> np.ndarray:
        theta_f = np.array([theta(mu) for theta in self.rom.fom.f.theta]).reshape(-1)
        if u is None:
            return theta_f
        else:
            theta_B = np.array([theta(mu) for theta in self.rom.fom.B.theta])
            return np.concatenate((theta_f, np.outer(-u, theta_B).reshape(-1)))
        
        
    def _initialize(self):
        if self.rom.fom.V.is_parametric:
            raise ValueError("Affine residual evaluation only supports FOMs with parameter-independent test space inner product.")
    
        self._Qf = len(self.rom.fom.f)
        self._QB = len(self.rom.fom.B)
        
        self._r = self.rom.fom.V.dual.riesz(NO_MU, np.column_stack(self.rom.fom.f.data))
        self._R = self.rom.fom.V.inner(NO_MU, self._r)
        
    def _initialize_output(self):
        if self.rom.fom.U.dual.is_parametric:
            raise ValueError("Dual norm of the output is not affine as the FOMs has a parameter-dependent trial space inner product.")
        
        L = self.rom.fom.U.dual.riesz(NO_MU, np.column_stack(self.rom.fom.l.data))
        self._L = self.rom.fom.U.dual.inner(NO_MU, L)
        
    def add_basis(self, basis: Vector):
        m = self.rom.fom.shape[0]
        r_new = np.empty((m, basis.shape[1], self._QB))
        for q, (theta_q, Bq_basis) in enumerate(self.rom.fom.B @ basis):
            r_new[:,:,q] = self.rom.fom.V.dual.riesz(NO_MU, Bq_basis)
        r_new = r_new.reshape((m, -1))
        
        r_old = self._r
        
        R_old_old = self._R
        R_old_new = self.rom.fom.V.inner(NO_MU, r_old, r_new)
        R_new_new = self.rom.fom.V.inner(NO_MU, r_new)
        
        self._r = np.hstack([r_old, r_new])
        self._R = np.block([[R_old_old,   R_old_new],
                            [R_old_new.T, R_new_new]])
    
    def rotate_basis(self, rotation: np.ndarray):
        rotation = np.kron(rotation, np.eye(self._QB))
        self._r[:, self._Qf:] = self._r[:, self._Qf:] @ rotation
        self._R[self._Qf:, :] = rotation.T @ self._R[self._Qf:, :]
        self._R[:, self._Qf:] = self._R[:, self._Qf:] @ rotation
        
        
    def dual_norm(self, mu: Mu, u: Vector) -> float:
        theta = self._theta_r(mu, u)
        return np.sqrt(abs(theta.T @ self._R @ theta))
    
    def dual_norm_rhs(self, mu: Mu) -> float:
        theta_f = self._theta_r(mu, None)
        return np.sqrt(abs(theta_f.T @ self._R[:self._Qf, :self._Qf] @ theta_f))
    
    def _dual_norm_output(self, mu: Mu) -> float:
        theta_l = np.array([theta(mu) for theta in self.rom.fom.l.theta]).reshape(-1)
        return np.sqrt(abs(theta_l.T @ self._L @ theta_l))
    
    
    
    
class TimeSteppingResidualNormEvaluator(Generic[Mu]):
    r"""
    Base class for computing the dual norm of the full-order residual of a :class:`StationaryTimeSteppingGalerkinROM`.
    """
    
    online_efficient: bool
    r"""Whether the residual norm evaluator provides online-efficient computation of the dual norm of the residual."""
    
    rom: StationaryTimeSteppingGalerkinROM[Mu]
    r"""The reduced-order model for which the residual is evaluated."""
    
    def __init__(self, rom: StationaryTimeSteppingGalerkinROM[Mu]):
        self.rom = rom
        self._initialize()
        if rom.W_basis is not None: self.add_basis(rom.W_basis)
        
    def _initialize(self):
        r"""
        Setup the the norm evaluater.
        """
        pass
        
    def add_basis(self, basis: Vector):
        r"""
        Extend the reduced trial basis :math:`U_{\text{basis}}`.
        """
        pass
    
    def rotate_basis(self, rotation: np.ndarray):
        r"""
        Rotate the reduced trial basis :math:`U_{\text{basis}}`, i.e. replace :math:`U_{\text{basis}}` by :math:`U_{\text{basis}} \cdot \text{rotation}`.
        """
        pass
        
    @abstractmethod
    def dual_norm(self, mu: Mu, u: TimeSteppingSolution) -> float:
        r"""
        Compute the dual norm of the residual at parameter :math:`\mu` for the reduced solution :math:`u`.
        """
        ...
    
    @abstractmethod    
    def initial_error(self, mu: Mu, u: TimeSteppingSolution) -> float:
        r"""
        Compute the error of the initial value.
        """
        ...
    
    def __repr__(self):
        return f"<{self.__class__.__name__} for {repr(self.rom)}>"
    
    

class FullTimeSteppingResidualNormEvaluator(TimeSteppingResidualNormEvaluator[Mu]):
    """Default implementation that delegates all computations to the full-order model. This is not online-efficient.
    """
    
    online_efficient = False
    
    def dual_norm(self, mu: Mu, u: TimeSteppingSolution) -> float:
        u = self.rom.reconstruct(mu, u).u
        K = self.rom.fom.K
        LIu = self.rom.fom.LI.B(mu) @ u[:,1:K+1]
        LEu = self.rom.fom.LE.B(mu) @ u[:,0:K]
        b = self.rom.fom.b(mu)
        r = LEu + b - LIu
        return self.rom.fom.W.dual.norm(mu, r)
    
    def initial_error(self, mu: Mu, u: TimeSteppingSolution) -> float:
        u0 = self.rom.reconstruct(mu, u).u[:,0]
        return self.rom.fom.W.norm(mu, u0 - self.rom.fom.u0(mu))
        
        
    
    
class AffineTimeSteppingResidualNormEvaluator(TimeSteppingResidualNormEvaluator[Mu]):
    """Exploiting the affine structure of the full-order model for online-efficient residual norm evaluation. Only applicable for FOMs with parameter-independent test space inner product.
    """
    
    online_efficient = True
    
    def _initialize(self):
        pass
    
    def add_basis(self, basis: Vector):
        pass
    
    def rotate_basis(self, rotation: np.ndarray):
        pass
    
    def dual_norm(self, mu: Mu, u: TimeSteppingSolution) -> float:
        pass
    
    def initial_error(self, mu: Mu, u: TimeSteppingSolution) -> float:
        pass