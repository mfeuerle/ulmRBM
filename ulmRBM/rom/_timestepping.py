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
from ulmRBM.fom import ParametricGalerkinOperator, TimeSteppingSolution, StationaryTimeSteppingGalerkinFOM
from ulmRBM.products import InnerProduct, EuclideanInnerProduct, orthonormalize

from ._constants import StabilityEstimator, ContinuityEstimator
from .__residual import TimeSteppingResidualNormEvaluator, FullTimeSteppingResidualNormEvaluator, AffineTimeSteppingResidualNormEvaluator

    
class StationaryTimeSteppingGalerkinROM(StationaryTimeSteppingGalerkinFOM[Mu]):
    r"""Reduced-order model for time-stepping problems with a Galerkin operator and stationary operators.
    
    This reduced-order model approximates the space :math:`W` in `StationaryTimeSteppingGalerkinFOM` by a low-dimensional subspace spanned :math:`W^N \subset W` spanned by :math:`W_{\text{basis}}`.
    """
    
    fom: StationaryTimeSteppingGalerkinFOM[Mu]
    """Underlying full-order model."""
    
    W_basis: Vector | None
    """:math:`(n,N)` trial space basis matrix, where :math:`N` is the dimension of the reduced trial space."""
    
    _residual_evaluator: TimeSteppingResidualNormEvaluator[Mu]
    _LI_stability_estimator: StabilityEstimator[Mu]
    _LE_continuity_estimator: ContinuityEstimator[Mu]
    
    _W_basis: Vector | None = None
    
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
    def W_basis(self) -> Vector:
        return self._W_basis
    @W_basis.setter
    def W_basis(self, value: Vector):
        self._W_basis = value
        self._need_assemble = True
        self.W = self.fom.W.restrict(self._W_basis)
        
    
    @property
    def n(self):
        return 0 if self.W_basis is None else self.W_basis.shape[1]

    def __init__(self, fom: StationaryTimeSteppingGalerkinFOM[Mu],
                 LI_stability: StabilityEstimator[Mu],
                 LE_continuity: ContinuityEstimator[Mu],
                 W_basis: Vector = None,
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
            W_basis:
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
        if W_basis is not None: self.add_basis(W_basis)
        
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
        r"""Assemble the reduced-order model based on the reduced basis.
        
        This method is in most cases called internally anyways. But if you want to ensure, that the reduced-order model is ready for the online stage, you might call this method.
        """
        if not self._need_assemble: return
        self._need_assemble = False
        
        if self.W_basis is not None:
            W_basis = self._W_basis
            self.LI = W_basis.T @ self.fom.LI.B @ W_basis
            self.LE = W_basis.T @ self.fom.LE.B @ W_basis
            self.b = W_basis.T @ self.fom.b
            
            if not self.fom.W.is_parametric:
                # orthogonal projection of u0 onto W_basis w.r.t. the self.fom.W inner product if W is paramter-independent
                solver = DirectSolver(factorize=True)
                Ub_U_Ub = self.W(NO_MU)
                Ub_U = W_basis.T @ self.fom.W(NO_MU)
                orthogonal_projection = lambda u: solver(Ub_U_Ub, Ub_U @ u)
                self.u0 = self.fom.u0.apply2data(orthogonal_projection)
            else:
                solver = DirectSolver(factorize=True)
                projection = lambda u: solver(self.W_basis, u)
                self.u0 = self.fom.u0.apply2data(projection)
            
        else:
            self.LI = None
            self.LE = None
            self.b = None
            self.u0 = None
        
        
    def add_basis(self, basis: Vector):
        r"""Add new basis vectors to ``W_basis``.
        
        Args:
            basis:
                New basis vectors :math:`(n, k)` to append.
        """
        if basis.ndim == 1: basis = basis.reshape(-1,1)
        if basis.shape[0] != self.fom.n:
            raise ValueError("Basis vector has incompatible dimension.")
        if self.W_basis is None:
            self.W_basis = basis
        else:
            self.W_basis = np.hstack([self.W_basis, basis])
        self._residual_evaluator.add_basis(basis)
        
        
    def orthonormalize(self, U: InnerProduct[Mu] | Matrix | None = None):
        """
        Orthonormalize ``W_basis`` with respect to a specified inner product.
        
        Performs orthonormalization using a parameter-independent
        inner product. This improves numerical stability and ensures well-conditioned
        reduced system matrices.
        
        Args:
            U:
                Parameter-independent inner product for orthonormalization.
                If ``None`` and :attr:`fom.W` is parameter-independent, uses :attr:`fom.W`.
                Otherwise uses Euclidean inner product.
        """
        if U is None: 
            if self.fom.W.is_parametric:
                U = EuclideanInnerProduct(self.fom.W.shape[0])
            else:
                U = self.fom.W
                
        self.W_basis, Q = orthonormalize(self.W_basis, U)
        self._residual_evaluator.rotate_basis(Q)
        
        
    def reconstruct(self, mu: Mu, u: TimeSteppingSolution = None) -> TimeSteppingSolution:
        r"""
        Reconstruct a full-order function from the reduced coefficients.
        
        Computes :math:`W_{\text{basis}} u_N(\mu)` to obtain the full-order representation of the reduced solution :math:`u_N(\mu)`.
        
        Args:
            mu:
                Parameter value at which to reconstruct the solution.
            u:
                Optional reduced-order solution. If ``None``, the reduced solution at :math:`\mu` is computed via :meth:`solve` and then reconstructed.
        
        Returns:
            :math:`(n,)` reduced-order approximation of the full-order solution.
        """
        if u is None: u = self.solve(mu)
        return TimeSteppingSolution(u.t, self.W_basis @ u.u)
    
    
    def error(self, mu: Mu, u: TimeSteppingSolution = None, u_fom: TimeSteppingSolution = None) -> np.ndarray:
        r"""
        Compute the true error between reduced and full-order solutions at each time step.
        
        Computes :math:`e_k := \|u(t_k;\mu) - W_{\text{basis}} u_N(t_k;\mu)\|_W` where :math:`u(t_k;\mu)` is the full-order solution and :math:`W_{\text{basis}} u_N(t_k;\mu)` is the reconstructed reduced solution at time :math:`t_k`.
        
        Args:
            mu:
                Parameter value at which to compute the error.
            u:
                Reduced-order solution. If ``None``, computed via :meth:`solve`.
            u_fom:
                Full-order solution. If ``None``, computed via :meth:`fom.solve`.
        
        Returns:
            True error :math:`e = (e_0, \ldots, e_K) \in \mathbb{R}^{K+1}`.
        """
        if u is None: u = self.solve(mu)
        if u_fom is None: u_fom = self.fom.solve(mu)
        u = self.reconstruct(mu, u)
        if len(u.t) != len(u_fom.t) or np.max(np.abs(u.t - u_fom.t)) > 1e-12:
            raise ValueError("Time points of ROM and FOM solution do not match.")
        return self.fom.W.norm(mu, u.u - u_fom.u)
        
        
    def error_bound(self, mu: Mu, u: TimeSteppingSolution = None) -> np.ndarray:
        r"""Guaranteed a-posteriori upper bound of the absolute error.
        
        Let :math:`e_k := \|u(t_k;\mu) - W_{\text{basis}} u_N(t_k;\mu)\|_W` be the error at time :math:`t_k`, where :math:`u(t_k;\mu)` is the full-order solution and :math:`W_{\text{basis}} u_N(t_k;\mu)` is the reconstructed reduced solution at time :math:`t_k`.
        
        Then, the following error bound is available:
        
        .. math::
            e_k \leq \Delta_k := \frac{1}{\sigma^I_{\text{LB}}(\mu)} (\gamma^E_{\text{UB}}(\mu) \Delta_{k-1} + \|r_k(\mu)\|_{W'}),
            
        where :math:`\sigma^I_{\text{LB}}(\mu)` is a lower bound for the stability constant of the implicit operator, :math:`\gamma^E_{\text{UB}}(\mu)` is an upper bound for the continuity constant of the explicit operator, and :math:`r_k(\mu)` is the residual at time step :math:`k`.
        
        Args:
            mu:
                Parameter value at which to estimate the error.
            u:
                Reduced-order solution. If ``None``, computed via :meth:`solve`.
            
        .. note::
            Due to the square-root effect, the bounds are only accurate up to ``sqrt(eps)`` where ``eps`` is the machine precision. Thus, for smaller errors, the lower bounds might be wrong and the upper bounds might be overestimated.
            
        .. note::
            As usula for time-stepping schemes, the error bound grows exponentially in the number of time steps with coefficient :math:`\frac{\sigma^I_{\text{LB}}(\mu)}{\gamma^E_{\text{UB}}(\mu)}`. Thus, for long time intervals, the error bound might be extremely pessimistic. 
        """
        if u is None: u = self.solve(mu)
        
        I_stability = self._LI_stability_estimator(mu)
        E_continuity = self._LE_continuity_estimator(mu)
        r = self._residual_evaluator.dual_norm(mu, u)
        
        delta = np.zeros(self.K+1)
        delta[0] = self._residual_evaluator.initial_error(mu, u)
        for k in range(self.K):
            delta[k+1] = 1/I_stability * (E_continuity * delta[k] + r[k])
            
        return delta