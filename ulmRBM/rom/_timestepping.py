"""
Time-stepping (full-order) models for initial value problems.
"""

from __future__ import annotations


__all__ = [
    'StationaryTimeSteppingGalerkinROM',
]

from collections.abc import Callable

import numpy as np

from ulmRBM.core import (
    NO_MU, Mu, Matrix, Vector
)
from ulmRBM.solver import Solver, DirectSolver
from ulmRBM.affine import AffineLinear
from ulmRBM.fom import ParametricGalerkinOperator, TimeSteppingSolution, StationaryTimeSteppingGalerkinModel
from ulmRBM.products import InnerProduct, EuclideanInnerProduct, orthonormalize

from ._constants import StabilityEstimator, ContinuityEstimator
from .__residual import TimeSteppingResidualNormEvaluator, FullTimeSteppingResidualNormEvaluator, AffineTimeSteppingResidualNormEvaluator

    
class StationaryTimeSteppingGalerkinROM(StationaryTimeSteppingGalerkinModel[Mu]):
    r"""Reduced-order model for time-stepping problems with a Galerkin operator and stationary operators.
    
    This reduced-order model approximates the space :math:`U` in `StationaryTimeSteppingGalerkinModel` by a low-dimensional subspace spanned :math:`U_{\text{basis}} \subset U`.
    """
    
    fom: StationaryTimeSteppingGalerkinModel[Mu]
    """Underlying full-order model."""
    
    U_basis: Vector | None
    """:math:`(n,N)` trial space basis matrix, where :math:`N` is the dimension of the reduced trial space."""
    
    _residual_evaluator: TimeSteppingResidualNormEvaluator[Mu]
    _LI_stability_estimator: StabilityEstimator[Mu]
    _LE_continuity_estimator: ContinuityEstimator[Mu]
    
    _need_assemble: bool
     
    @property
    def LI(self) -> ParametricGalerkinOperator[Mu, Matrix]:
        self.assemble()
        return self._LI
    @LI.setter
    def LI(self, value: AffineLinear[Mu, Matrix]):
        if value is not None:
            self._LI = ParametricGalerkinOperator(value, self.W, self._solver)
        else:
            self._LI = None
        
    @property
    def LE(self) -> ParametricGalerkinOperator[Mu, Matrix]:
        self.assemble()
        return self._LE
    @LE.setter
    def LE(self, value: AffineLinear[Mu, Matrix]):
        if value is not None:
            self._LE = ParametricGalerkinOperator(value, self.W)
        else:
            self._LE = None
        
    @property
    def b(self) -> np.ndarray[AffineLinear[Mu, Vector]]:
        self.assemble()
        return self._b
    @b.setter
    def b(self, value: np.ndarray[AffineLinear[Mu, Vector]]):
        self._b = value
        
    @property
    def u0(self) -> AffineLinear[Mu, Vector]:
        self.assemble()
        return self._u0
    @u0.setter
    def u0(self, value: AffineLinear[Mu, Vector]):
        self._u0 = value
        
    @property
    def U_basis(self) -> Vector:
        return self._U_basis
    @U_basis.setter
    def U_basis(self, value: Vector):
        self._U_basis = value
        self._need_assemble = True
        self.W = self.fom.W.restrict(self._U_basis)

    def __init__(self, fom: StationaryTimeSteppingGalerkinModel[Mu],
                 LI_stability: StabilityEstimator[Mu],
                 LE_continuity: ContinuityEstimator[Mu],
                 U_basis: Vector = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = DirectSolver(factorize=True),
                 residual: None | str = None):
        r"""
        Args:
            fom:
                Full-order model for which to construct the reduced-order model.
            LI_stability:
                Stability estimator for the implicit operator ``fom.LI``.
            LE_continuity:
                Continuity estimator for the explicit operator ``fom.LE``.
            U_basis:
                Initial reduced trial basis. If ``None``, starts with an empty basis.
            solver:
                Solver for the implicit operations. If ``None``, uses a direct solver with factorization.
            residual: 
                Method to evaluate the residual; takes values ``'affine'`` or ``'full'``.
                Required for residual-based error bounds, see `error_bound`. The option ``'affine'`` is only applicable if the test space inner product is parameter independent. This method exploits the affine structure of the problem and thus is online-efficent. The option ``'full'`` is a fall back that delegates all calculations to the full-order model and is thus not online-efficient. If ``None``, defaults to ``'affine'`` if possible, otherwise ``'full'``.
        """
        
        self.fom = fom
        
        if residual is None:
            if fom.W.is_parametric:
                from warnings import warn
                warn("An affine decomposition of the residual is not possible for FOMs with parameter-dependent test space inner product. Thus, the error bounds can not be evaluated online efficient.", UserWarning)
                residual = 'full'
            else:
                residual = 'affine'
        
        if LI_stability.B is not fom.LI:
            raise ValueError("The stability estimator does not match the implicit operator.")
        if LE_continuity.B is not fom.LE:
            raise ValueError("The continuity estimator does not match the explicit operator.")
        
        self.fom = fom
        self.t = fom.t
        self._LI_stability_estimator  = LI_stability
        self._LE_continuity_estimator = LE_continuity
        self._solver = solver
        if U_basis is not None: self.add_basis(U_basis)
        
        if residual == 'affine':
            # self._residual_evaluator = AffineTimeSteppingResidualNormEvaluator(self)
            self._residual_evaluator = FullTimeSteppingResidualNormEvaluator(self)
            from warnings import warn
            warn("Affine decomposition of the residual not implemented yet.", UserWarning)
        elif residual == 'full':
            self._residual_evaluator = FullTimeSteppingResidualNormEvaluator(self)
        else:
            raise ValueError("Invalid option for 'residual'. Must be either 'affine' or 'full' or None.")
        
        
    def assemble(self):
        if not self._need_assemble: return
        self._need_assemble = False
        
        if self.U_basis is not None:
            U_basis = self._U_basis
            self.LI = U_basis.T @ self.fom.LI.B @ U_basis
            self.LE = U_basis.T @ self.fom.LE.B @ U_basis
            b = np.empty(self.fom.K-1, dtype=AffineLinear)
            for k in range(self.fom.K-1):
                b[k] = U_basis.T @ self.fom.b[k]
            self.b = b
            
            if not self.fom.W.is_parametric:
                # orthogonal projection of u0 onto U_basis w.r.t. the self.fom.U inner product if U is paramter-independent
                solver = DirectSolver(factorize=True)
                Ub_U_Ub = self.W(NO_MU)
                Ub_U = U_basis.T @ self.fom.W(NO_MU)
                orthogonal_projection = lambda u: self.U_basis @ solver(Ub_U_Ub, Ub_U @ u)
                self.u0 = self.fom.u0.apply2data(orthogonal_projection)
            else:
                solver = DirectSolver(factorize=True)
                projection = lambda u: solver(self.U_basis, u)
                self.u0 = self.fom.u0.apply2data(projection)
            
        else:
            self.LI = None
            self.LE = None
            self.b = None
            self.u0 = None
        
        
    def add_basis(self, basis: Vector):
        if basis.ndim == 1: basis = basis.reshape(-1,1)
        if basis.shape[0] != self.fom.n:
            raise ValueError("Basis vector has incompatible dimension.")
        if self.U_basis is None:
            self.U_basis = basis
        else:
            self.U_basis = np.hstack([self.U_basis, basis])
        self._residual_evaluator.add_basis(basis)
        
        
    def orthonormalize(self, U: InnerProduct[Mu] | Matrix | None = None):
        """
        Orthonormalize the reduced basis with respect to a specified inner product.
        
        Performs orthonormalization using a parameter-independent
        inner product. This improves numerical stability and ensures well-conditioned
        reduced system matrices.
        
        Args:
            U:
                Parameter-independent inner product for trial space orthonormalization.
                If ``None`` and :attr:`fom.U` is parameter-independent, uses :attr:`fom.U`.
                Otherwise uses Euclidean inner product.
        """
        if U is None: 
            if self.fom.W.is_parametric:
                U = EuclideanInnerProduct(self.fom.W.shape[0])
            else:
                U = self.fom.W
                
        self.U_basis, Q = orthonormalize(self.U_basis, U)
        self._residual_evaluator.rotate_basis(Q)
        
        
    def resconstruct(self, mu: Mu, u: TimeSteppingSolution = None) -> TimeSteppingSolution:
        if u is None: u = self.solve(mu)
        return TimeSteppingSolution(u.mu, u.t, self.U_basis @ u.u)
    
    
    def error(self, mu: Mu, u: TimeSteppingSolution = None, u_fom: TimeSteppingSolution = None) -> np.ndarray:
        if u is None: u = self.solve(mu)
        if u_fom is None: u_fom = self.fom.solve(mu).u
        u = self.resconstruct(u).u
        return self.fom.W.norm(mu, u - u_fom)
        
        
    def error_bound(self, mu: Mu, u: TimeSteppingSolution = None) -> np.ndarray:
        if u is None: u = self.solve(mu)
        
        I_stability = self._LI_stability_estimator(mu)
        E_continuity = self._LE_continuity_estimator(mu)
        r = self._residual_evaluator.dual_norm(mu, u)
        
        delta = np.zeros(self.K+1)
        delta[0] = self._residual_evaluator.initial_error(mu, u)
        for k in range(self.K):
            delta[k+1] = 1/I_stability * (E_continuity * delta[k] + r[k])
            
        return delta