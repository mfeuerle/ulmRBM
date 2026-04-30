
from __future__ import annotations


__all__ = [
    'ResidualCalculator',
    'ResidualOptions',
    'DirectResidual',
    'AffineResidual',
]

from abc import abstractmethod
from typing import Generic, TYPE_CHECKING

from enum import StrEnum, auto

import numpy as np

from ulmRBM.core import NO_MU, Mu, Vector
from ulmRBM.affine import AffineLinear
from ulmRBM.fom import FOM

if TYPE_CHECKING:
    from ulmRBM.rom import ROM

# maybe add a global flag to warn if any computations inside the rom use full-order dimensions
    
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


class ResidualCalculator(Generic[Mu]):
    r"""
    Base class for computing the dual norm of the full-order residual of a :class:`ROM`.
    
    Let :math:`f(\mu)` and :math:`B(\mu)` be the right-hand side and system matrix of the full-order model,
    :math:`U_{\text{basis}}(\mu)` the reduced basis of the trial space, and :math:`u` the reduced solution vector.
    Then, :meth:`norm2` computes the square of the dual norm of the residual:
    
    .. math::
        \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}}(\mu) u\|_{V'}^2.
        
    The residual calculater works in two phases:
    
    1. **Offline phase**: Precompute expensive data via :meth:`set_basis`, :meth:`add_basis` and :meth:`rotate_basis` to define :math:`U_{\text{basis}}(\mu)`.
    2. **Online phase**: Provide fast computation of :math:`\|r(\mu; u)\|_{V'}^2` via :meth:`norm2` using the precomputed data.
    """
    
    
    fom: FOM[Mu]
    """Full-order model for which the dual norm of the residual is computed."""
    _fom: FOM[Mu] | None = None
    
    def __repr__(self):
        return f"<{self.__class__.__name__} for {repr(self.fom)}>"

        
    def set(self, fom: FOM[Mu] | ROM[Mu], basis: AffineLinear[Mu, Vector] = None):
        r"""
        Set the full-order model and trial basis for residual computation.
        
        Args:
            fom:
                Full-order model or reduced-order model. If a reduced-order model is provided,
                its underlying FOM and trial basis are used.
            basis:
                Trial basis :math:`U_{\text{basis}}(\mu)` to use for residual computation. Only used if a full-order model is provided.
        """
        from ulmRBM.rom import ROM
        
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
    def set_basis(self, basis: AffineLinear[Mu, Vector]):
        r"""
        Set the reduced trial basis :math:`U_{\text{basis}}(\mu)` for residual computation.
        """
        ...
    
    @abstractmethod
    def add_basis(self, basis: AffineLinear[Mu, Vector]):
        r"""
        Extend the reduced trial basis :math:`U_{\text{basis}}(\mu)` for residual computation.
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
            \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}}(\mu) u\|_{V'}^2
            
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
    
    basis: AffineLinear[Mu, Vector] = None
    r"""Reduced trial basis :math:`U_{\text{basis}}(\mu)` for residual computation."""
    
    def set_basis(self, basis: AffineLinear[Mu, Vector]):
        self.basis = basis
        
    def add_basis(self, basis: AffineLinear[Mu, Vector]):
        if self.basis is None:
            self.set_basis(basis)
        else:
            if len(self.basis) != len(basis):
                raise ValueError("Cannot add basis with different number of affine terms.")
            new_basis = AffineLinear()
            for ((theta, U_old), U_new) in zip(self.basis, basis.data):
                new_basis += [(theta, np.hstack([U_old, U_new]))]
            self.set_basis(new_basis)
                
            # self.set_basis(np.hstack((self.basis, basis)))
        
    def rotate_basis(self, rotation: np.ndarray):
        self.set_basis(self.basis @ rotation)
        
    def norm2(self, mu: Mu, u: Vector) -> float:
        f"""{ResidualCalculator.norm2.__doc__}
    
        ... note::
            This method is not online-efficient, as it requires full-order operations.
        """
        f = self.fom.f(mu)
        Bu = self.fom.B(mu) @ (self.basis(mu) @ u)
        return self.fom.V.dual.inner(mu, f -  Bu)
    
class AffineResidual(ResidualCalculator[Mu]):
    r"""
    Online-efficient computation of the dual norm of the full-order residual
    
    .. math::
        \|r(\mu; u)\|_{V'}^2 = \|f(\mu) - B(\mu) U_{\text{basis}}(\mu) u\|_{V'}^2
        
    by preccomputing all full-order operations before calling :meth:`norm2`.
    Thereby, :math:`f(\mu)` and :math:`B(\mu)` are the right-hand side and system matrix of the full-order model,
    :math:`U_{\text{basis}}` the reduced basis of the trial space, :math:`u` a element of the reduced trial space and :math:`V'` the dual space of the test space.
    
    The online-efficient computation relies on a affine decomposition of the residual norm:
    
    .. math::
        \|r(\mu; u)\|_{V'}^2 = \sum_{i,j=0}^{Q_r-1} \theta^r_i(\mu; u)\, \theta^r_j(\mu; u)\, R_{ij}
    
    where :math:`R_{ij} = (r_i, r_j)_{V'}` is the precomputed Gram matrix and :math:`r_i` and :math:`\theta^r_i(\mu; u)` are the affine decomposition of the residual :math:`r(\mu; u) = f(\mu) - B(\mu) U_{\text{basis}}(\mu) u`, i.e.:
    
    .. math::
        r(\mu; u) = \sum_{i=0}^{Q_r-1} \theta^r_i(\mu; u)\, r_i
    
    with :math:`Q_r = Q_f + N \cdot Q_B \cdot Q_{U_{\text{basis}}}`, with :math:`N` denoting the number of basis vectors in :math:`U_{\text{basis}}`, and:
    
    - for :math:`q = 0, \ldots, Q_f-1` set :math:`\theta^r_q(\mu; u) := \theta^f_q(\mu)` and :math:`r_q := R_{V'}f_q` (from RHS)
    - for :math:`i = 0, \ldots, N-1`, :math:`j = 0, \ldots, Q_B-1` and :math:`k=0,\ldots,Q_{U_{\text{basis}}}-1`, with :math:`q := Q_f + i + j\cdot N + k\cdot N \cdot Q_B` set :math:`\theta^r_q(\mu; u) := - u_i \theta_j^B(\mu)\theta_k^{U_{\text{basis}}}` and :math:`r_q := R_{V'} B_j \phi_i^k` (from system matrix and reduced basis)
    
    where :math:`R_{V'}:V'\to V` is the Riesz map of the dual space of the test space, :math:`U_{\text{basis}}(\mu) = \sum_{k=1}^{Q_{U_{\text{basis}}}} \theta_k^{Q_{U_{\text{basis}}}}(\mu) (\phi_0^k, \ldots, \phi_{N-1}^k)` are the basis vectors of the reduced trial space, and using the notation :math:`f(\mu) = \sum_{q=0}^{Q_f-1} \theta^f_q(\mu) f_q` and :math:`B(\mu) = \sum_{q=0}^{Q_B-1} \theta^B_q(\mu) B_q` for the affine decompositions of :math:`f(\mu)` and :math:`B(\mu)`.
    
    This allows for an online-efficient computation of the dual norm in :math:`\mathcal{O}(Q_r^2)` operations (independent of full-order dimension).
    
    .. note::
        This online-efficient computation requires that the test space inner product is parameter-independent.
    """
    
    r: Vector
    r"""Precomputed Riesz representatives of the residual components :math:`(r_0, \ldots, r_{Q_r-1}) \in \mathbb{R}^{m \times Q_r}`."""
    
    R: np.ndarray
    r"""Precomputed Gram matrix of the residual components :math:`R_{ij} = (r_i, r_j)_{V'}`, :math:`i,j=0,\ldots,Q_r-1`."""
    
    _Ubasis_thetas: AffineLinear[Mu, Vector] = None
    
    @property
    def _Qf(self) -> int:
        return len(self.fom.f)
    @property
    def _QB(self) -> int:
        return len(self.fom.B)
    @property
    def _QU(self) -> int:
        return len(self._Ubasis_thetas)
    

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
        
        # dirty hack to get the tetas of B @ basis
        B_tmp = self.fom.B.apply2data(lambda Bq: np.ones((0,0)))
        theta_BU = np.array([theta(mu) for theta in (B_tmp @ self._Ubasis_thetas).theta])
        
        return np.concatenate((theta_f.reshape(-1), np.outer(-u, theta_BU).reshape(-1)))
    
    
    def _set_fom(self, fom: FOM[Mu]):
        r"""
        Initialize the affine residual structures and precomputes the the first part of the residual decomposition that only depends on the right-hand side (and not the system matrix and reduced basis).
        """
        if fom.V.is_parametric:
            raise ValueError("OnlineResidual only supports FOMs with parameter-independent test space inner product.")
        super()._set_fom(fom)
        self.r = self.fom.V.dual.riesz(NO_MU, np.column_stack(self.fom.f.data))
        self.R = self.fom.V.inner(NO_MU, self.r)
        
    
    def set_basis(self, basis: AffineLinear[Mu, Vector]):
        f"""
        {ResidualCalculator.set_basis.__doc__}
        
        Precomputes the second part of the residual decomposition that depends on the system matrix and the reduced basis.
        """
        self.r = self.r[:, :self._Qf]
        self.R = self.R[:self._Qf, :self._Qf]
        self.add_basis(basis)
        
    def add_basis(self, basis: AffineLinear[Mu, Vector]):
        f"""
        {ResidualCalculator.add_basis.__doc__}
        
        Extends and precomputes the second part of the residual decomposition that depends on the system matrix and the reduced basis.
        """
        if self._Ubasis_thetas is None:
            self._Ubasis_thetas = basis.apply2data(lambda basisq: np.ones((0,0)))
        elif len(basis) != self._QU:
            raise ValueError("Can not add basis with different number of affine terms.")
        m = self.fom.dim[0]
        r_new = np.empty((m, basis.shape[1], self._QB * self._QU))
        for q, (theta_q, Bu_q) in enumerate(self.fom.B @ basis):
            r_new[:,:,q] = self.fom.V.dual.riesz(NO_MU, Bu_q)
        r_new = r_new.reshape((m, -1))  # combine all basis vectors
        
        r_old = self.r
        
        R_old_old = self.R
        R_old_new = self.fom.V.inner(NO_MU, r_old, r_new)
        R_new_new = self.fom.V.inner(NO_MU, r_new)
        
        self.r = np.hstack([r_old, r_new])
        self.R = np.block([[R_old_old,   R_old_new],
                           [R_old_new.T, R_new_new]])

    def rotate_basis(self, rotation: np.ndarray):
        rotation = np.kron(rotation, np.eye(self._QB*self._QU))
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