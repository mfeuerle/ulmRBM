
from __future__ import annotations


__all__ = [
    'ResidualNormEvaluator',
    'FullResidualNormEvaluator',
    'AffineResidualNormEvaluator',
]

from abc import abstractmethod
from typing import Generic, TYPE_CHECKING

import numpy as np

from ulmRBM.core import NO_MU, Mu, Vector
from ulmRBM.affine import AffineLinear

if TYPE_CHECKING:
    from ulmRBM.rom_old import ROM


class ResidualNormEvaluator(Generic[Mu]):
    r"""
    Base class for computing the dual norm of the full-order residual of a :class:`ROM`.
    
    Let :math:`f(\mu)` and :math:`B(\mu)` be the right-hand side and system matrix of the full-order model,
    :math:`U_{\text{basis}}(\mu)` the reduced basis of the trial space, and :math:`u` the reduced solution vector.
    Then, :meth:`dual_norm` computes the dual norm of the residual:
    
    .. math::
        \|r(\mu; u)\|_{V'} = \|f(\mu) - B(\mu) U_{\text{basis}}(\mu) u\|_{V'}.
        
    The residual calculater works in two phases:
    
    1. **Offline phase**: Precompute expensive data via :meth:`set_basis`, :meth:`add_basis` and :meth:`rotate_basis` to define :math:`U_{\text{basis}}(\mu)`.
    2. **Online phase**: Provide fast computation of :math:`\|r(\mu; u)\|_{V'}` via :meth:`dual_norm` using the precomputed data.
    """
    
    online_efficient: bool
    r"""Whether the residual norm evaluator provides online-efficient computation of the dual norm of the residual."""
    
    rom: ROM[Mu]
    r"""The reduced-order model for which the residual is evaluated."""
    
    def __init__(self, rom: ROM[Mu]):
        self.rom = rom
        self._initialize()
        if rom.U_basis is not None: self.add_basis(rom.U_basis)
        
    def _initialize(self):
        pass
        
    def add_basis(self, basis: AffineLinear[Mu, Vector]):
        r"""
        Extend the reduced trial basis :math:`U_{\text{basis}}(\mu)` for residual computation.
        """
        pass
    
    def rotate_basis(self, rotation: np.ndarray):
        r"""
        Rotate the reduced trial basis :math:`U_{\text{basis}}` for residual computation, i.e. replace :math:`U_{\text{basis}}` by :math:`U_{\text{basis}} \cdot \text{rotation}`.
        """
        pass
        
    @abstractmethod
    def dual_norm(self, mu: Mu, u: Vector) -> float:
        r"""
        Compute the dual norm of the residual at parameter :math:`\mu` for the reduced solution :math:`u`, i.e.
        
        .. math::
            \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}}(\mu) u\|_{V'}^2
            
        where :math:`f(\mu)` and :math:`B(\mu)` are the right-hand side and system matrix of the full-order model,
        :math:`U_{\text{basis}}` the reduced basis of the trial space, and :math:`V'` the dual space of the test space.
        """
        ...
        
    @abstractmethod    
    def dual_norm_rhs(self, mu: Mu) -> float:
        ...
    
    def __repr__(self):
        return f"<{self.__class__.__name__} for {repr(self.rom)}>"
    
    

class FullResidualNormEvaluator(ResidualNormEvaluator[Mu]):#
    
    online_efficient = False
    
    def dual_norm(self, mu: Mu, u: Vector) -> float:
        f = self.rom.fom.f(mu)
        Bu = self.rom.fom.B(mu) @ (self.rom.reconstruct(mu, u))
        return self.rom.fom.V.dual.norm(mu, f -  Bu)
    
    def dual_norm_rhs(self, mu: Mu) -> float:
        return self.rom.fom.V.dual.norm(mu, self.rom.fom.f(mu))
    
    
    
class AffineResidualNormEvaluator(ResidualNormEvaluator[Mu]):
    
    online_efficient = True
    
    _r: Vector
    r"""Precomputed Riesz representatives of the residual components :math:`(r_0, \ldots, r_{Q_r-1}) \in \mathbb{R}^{m \times Q_r}`."""
    
    _R: np.ndarray
    r"""Precomputed Gram matrix of the residual components :math:`R_{ij} = (r_i, r_j)_{V'}`, :math:`i,j=0,\ldots,Q_r-1`."""
        
    def _initialize(self):
        if self.rom.fom.V.is_parametric:
            raise ValueError("Affine only supports FOMs with parameter-independent test space inner product.")
    
        self._Qf = len(self.rom.fom.f)
        self._QB = len(self.rom.fom.B)
        
        self._r = self.rom.fom.V.dual.riesz(NO_MU, np.column_stack(self.rom.fom.f.data))
        self._R = self.rom.fom.V.inner(NO_MU, self._r)
        
    def add_basis(self, basis: AffineLinear[Mu, Vector]):
        m = self.rom.fom.dim[0]
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
        theta = self._theta(mu, u)
        return np.sqrt(abs(theta.T @ self._R @ theta))
    
    def dual_norm_rhs(self, mu: Mu) -> float:
        theta_f = self._theta(mu, None)
        return np.sqrt(abs(theta_f.T @ self._R[:self._Qf, :self._Qf] @ theta_f))
    
    def _theta(self, mu: Mu, u: Vector | None) -> np.ndarray:
        theta_f = np.array([theta(mu) for theta in self.rom.fom.f.theta]).reshape(-1)
        if u is None:
            return theta_f
        else:
            theta_B = np.array([theta(mu) for theta in self.rom.fom.B.theta])
            return np.concatenate((theta_f, np.outer(-u, theta_B).reshape(-1)))