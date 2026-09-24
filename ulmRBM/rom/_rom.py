
from __future__ import annotations

__all__ = [
    'ROM',
    'GalerkinROM',
    'PrimalDualROM',
    'PrimalDualGalerkinROM',
]

from collections.abc import Callable

import numpy as np
from warnings import warn

from ulmRBM.core import Mu, Matrix, Vector
from ulmRBM.solver import Solver, DirectSolver
from ulmRBM.fom import Model, FOM, GalerkinFOM, PrimalDualModel
from ulmRBM.products import InnerProduct, EuclideanInnerProduct, orthonormalize
from ulmRBM.affine import AffineLinear, wrap_affinelinear

from ._constants import StabilityEstimator, ContinuityEstimator, ExactStability, ExactContinuity, EfficientConstantEstimator
from .__residual import ResidualNormEvaluator, AffineResidualNormEvaluator, FullResidualNormEvaluator

class ROM(FOM[Mu]):
    r"""
    Reduced-order model given by a Petrov-Galerkin projection of a full-order model.
    
    Constructs a reduced-order model by projecting the full-order model onto
    low-dimensional trial and test spaces :math:`U_N` and :math:`V_N` spanned by basis matrices. The reduced
    system has the form:
    
    .. math::
        B_N(\mu) u_N = f_N(\mu)
        
    with an optional output of interest functional
    
    .. math::
        s_N(\mu) = l_N(\mu) u_N,
    
    where :math:`B_N(\mu) = V_N^T(\mu) B(\mu) U_N`, :math:`f_N(\mu) = V_N^T(\mu) f(\mu)`
    and :math:`l_N(\mu) = l(\mu) U_N`,
    with :math:`U_N \in \mathbb{R}^{n \times N}` the reduced trial basis and
    :math:`V_N(\mu) \in \mathbb{R}^{m \times M}` the reduced test basis with affine parameter dependence.
    Then, :math:`u_N \in \mathbb{R}^N` is the reduced solution vector and :math:`U_N u_N \in \mathbb{R}^n`
    the corresponding approximation of the full-order solution, while :math:`s_N(\mu)\in\mathbb{R}^p` 
    are the output(s) of interest of the reduced solution. 
    
    The reduced test space basis :math:`V_N(\mu)` is defined implicitly by a trail-to-test operator :math:`T(\mu)`, i.e. :math:`V_N(\mu) = T(\mu) U_N`. If the test space inner product is parameter independent, the default trial-to-test operator is given by the supremizing operator of the full-order model which will always lead to a stable reduced-order model.
    """
    
    fom: Model[Mu]
    """Underlying full-order model."""
    U_basis: Vector | None
    """:math:`(n,N)` trial space basis matrix, where :math:`N` is the dimension of the reduced trial space."""
    V_basis: AffineLinear[Mu, Vector] | None
    """:math:`(m,M)` test space basis matrix, where :math:`M` is the dimension of the reduced test space."""
    
    _U_basis: Vector | None = None
    _V_basis: AffineLinear[Mu, Vector] | None = None
    _B: AffineLinear[Mu, Matrix] | None = None
    _f: AffineLinear[Mu, Vector] | None = None
    _l: AffineLinear[Mu, Matrix] | None = None
    
    _fom_stability_estimator: StabilityEstimator[Mu]
    r"""Estimator :math:`\beta_{\text{LB}}(\mu)` for the FOM stability constant, required for error estimation."""
    _fom_continuity_estimator: ContinuityEstimator[Mu]
    r"""Estimator :math:`\gamma_{\text{UB}}(\mu)` for the FOM continuity constant."""
    _residual_evaluator: ResidualNormEvaluator[Mu]
    r"""Evaluator for the dual norm of the residual, required for error estimation."""
    
    _trial2test: Callable[[Vector], AffineLinear[Mu, Vector]] | None = None
    r"""Internal storage if a custom trial2test function is provided."""
    
    _need_assemble_B: bool = False
    r"""Flag to indicate whether the reduced system matrices needs to be reassembled."""
    _need_assemble_f: bool = False
    r"""Flag to indicate whether the reduced right-hand side needs to be reassembled."""
    _need_assemble_l: bool = False
    r"""Flag to indicate whether the reduced output functional needs to be reassembled."""
    
    @property
    def B(self) -> AffineLinear[Mu, Matrix]:
        self._assemble_B()
        return self._B
        
    @property
    def f(self) -> AffineLinear[Mu, Vector]:
        self._assemble_f()
        return self._f
        
    @property
    def l(self) -> AffineLinear[Mu, Matrix] | None:
        self._assemble_l()
        return self._l

    @property
    def U_basis(self) -> Vector:
        return self._U_basis
    @U_basis.setter
    def U_basis(self, value: Vector):
        self._mark_for_assembly(U=True)
        self._U_basis = value
        self.U = self.fom.U.restrict(self._U_basis)
        
    @property
    def V_basis(self) -> AffineLinear[Mu, Vector]:
        return self._V_basis
    @V_basis.setter
    def V_basis(self, value: AffineLinear[Mu, Vector]):
        self._mark_for_assembly(V=True)
        self._V_basis = wrap_affinelinear(value)
        self.V = self.fom.V.restrict(self._V_basis)
        
    @property
    def shape(self):
        return (0 if self.U_basis is None else self.U_basis.shape[1],
                0 if self.V_basis is None else self.V_basis.shape[1],
                self.fom.shape[2])

    
    def __init__(self, 
                 fom: Model[Mu],
                 stability: StabilityEstimator[Mu],
                 continuity: ContinuityEstimator[Mu] = None,
                 trial2test: Callable[[Vector], AffineLinear[Mu, Vector]] = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = None,
                 residual: None | str = None):
        r"""
        Args:
            fom:
                Full-order model to reduce.
            stability:
                Estimator for the FOM stability constant. Required for online-efficient residual-based error bounds, see `error_bound` and `error_bounds` (expept for ``alb``).
            continuity:
                Estimator for the FOM continuity constant. Not required for `error_bound`, but for all other online-efficient error bounds, see `error_bounds` (exept for ``aub``). Provide an online-efficient implementation if you want to use any of thes error bounds in an online efficient way. If ``None``, defaults to `ExactContinuity`.
            trial2test:
                Trial-to-Test operator :math:`T(\mu)` to define the test space basis via :math:`V_N(\mu) = T(\mu) U_N`. If ``None`` and the test space inner product of ``fom`` is parameter independent, defaults to the supremizing operator of the full-order model (which always yields a well-posed reduced model), otherwise the user has to provide a custom operator.
            solver:
                Solver for the reduced linear system. Defaults to `DirectSolver`.
            residual: 
                Method to evaluate the residual; takes values ``'affine'`` or ``'full'``.
                Required for residual-based error bounds, see `error_bound` and `error_bounds`. The option ``'affine'`` is only applicable if the test space inner product is parameter independent. This method exploits the affine structure of the problem and thus is online-efficent. The option ``'full'`` is a fall back that delegates all calculations to the full-order model and is thus not online-efficient. If ``None``, defaults to ``'affine'`` if possible, otherwise ``'full'``.
        """
        
        if continuity is None: continuity = ExactContinuity(fom)
        if trial2test is None: trial2test = fom.supremizer
        if solver is None: solver = DirectSolver()
        
        if residual is None:
            if fom.V.is_parametric:
                warn("An affine decomposition of the residual is not possible for FOMs with parameter-dependent test space inner product. Thus, the error bounds can not be evaluated online efficient.", UserWarning)
                residual = 'full'
            else:
                residual = 'affine'
        
        if stability.B is not fom:
            raise ValueError("The fom of the stability estimator does not match the fom of the reduced order model.")
        if continuity.B is not fom:
            raise ValueError("The fom of the continuity estimator does not match the fom of the reduced order model.")
        
        self.fom = fom
        self._fom_stability_estimator  = stability
        self._fom_continuity_estimator = continuity
        self._solver = solver
        self._trial2test = trial2test
            
        if residual == 'affine':
            self._residual_evaluator = AffineResidualNormEvaluator(self)
        elif residual == 'full':
            self._residual_evaluator = FullResidualNormEvaluator(self)
        else:
            raise ValueError("Invalid option for 'residual'. Must be either 'affine' or 'full' or None.")
        
    
    def _mark_for_assembly(self, U: bool = False, V: bool = False):
        r"""Mark components of the reduced-order model for (re)assembly.
        
        Args:
            U:
                If ``True``, makrs every component that depends on the trial space basis for reassembly.
            V:
                If ``True``, makrs every component that depends on the test space basis for reassembly.
        """
        if U:
            self._need_assemble_B = True
            self._need_assemble_l = True
        if V:
            self._need_assemble_B = True
            self._need_assemble_f = True
            
    def _assemble_B(self):
        r"""Assemble the reduced system matrix :math:`B_N(\mu) = V_N^T(\mu) B(\mu) U_N`."""
        if self._need_assemble_B:
            self._need_assemble_B = False
            if self.V_basis is not None and self.U_basis is not None:
                self._B = self.V_basis.T @ self.fom.B @ self.U_basis
            else:
                self._B = None
    
    def _assemble_f(self):
        r"""Assemble the reduced right-hand side vector :math:`f_N(\mu) = V_N^T(\mu) f(\mu)`."""
        if self._need_assemble_f:
            self._need_assemble_f = False
            if self.V_basis is not None:
                self._f = self.V_basis.T @ self.fom.f
            else:
                self._f = None
                
    def _assemble_l(self):
        r"""Assemble the reduced output functional :math:`l_N(\mu) = l(\mu) U_N`."""
        if self._need_assemble_l:
            self._need_assemble_l = False
            if self.U_basis is not None and self.fom.l is not None:
                self._l = self.fom.l @ self.U_basis
            else:
                self._l = None
        
    def assemble(self):
        r"""Assemble the reduced-order model.
        
        After calling this methid, the reduced-order model is setup for the online phase.
        """
        self._assemble_B()
        self._assemble_f()
        self._assemble_l()
        
        
    def add_basis(self, basis: Vector):
        r"""Add new trial basis vectors and update the test basis via trial-2-test relation.
        
        Args:
            basis:
                New trial basis vectors :math:`(n, k)` to append.        
        """
        self._add_basis_U(basis)
        self._add_basis_V(self._trial2test(basis))
    
    def _add_basis_U(self, basis: Vector):
        if basis.ndim == 1: basis = basis.reshape(-1,1)
        if basis.shape[0] != self.fom.shape[0]:
            raise ValueError("Basis vector has incompatible dimension.")
        if self.U_basis is None:
            self.U_basis = basis
        else:
            self.U_basis = np.hstack([self.U_basis, basis])
        self._residual_evaluator.add_basis(basis)
        
    def _add_basis_V(self, basis: AffineLinear[Mu, Vector] | Vector):
        basis = wrap_affinelinear(basis)
        if len(basis.shape) == 1: 
            basis = basis.apply2data(lambda dq: dq.reshape(-1,1))
        if basis.shape[0] != self.fom.shape[0]:
            raise ValueError("Basis vector has incompatible dimension.")
        if self.V_basis is None:
            V_basis = basis
        else:
            if len(self.V_basis) != len(basis):
                raise ValueError("Cannot add basis with different number of affine terms.")
            V_basis = AffineLinear()
            for ((theta, V_old), V_new) in zip(self.V_basis, basis.data):
                V_basis += [(theta, np.hstack([V_old, V_new]))]
        self.V_basis = V_basis
        
    
    def orthonormalize(self, U: InnerProduct[Mu] | Matrix | None = None):
        """
        Orthonormalize the trial space basis with respect to a specified inner product.
        
        Performs orthonormalization using a parameter-independent
        inner product. This improves numerical stability and ensures well-conditioned
        reduced system matrices.
        
        Args:
            U:
                Parameter-independent inner product for trial space orthonormalization.
                If ``None`` and :attr:`fom.U` is parameter-independent, uses :attr:`fom.U`.
                Otherwise uses Euclidean inner product.
        """
        Q = self._orthonormalize_U(U)
        self.V_basis = self.V_basis @ Q
        
    def _orthonormalize_U(self, U: InnerProduct[Mu] | Matrix | None) -> np.ndarray: 
        if U is None: 
            if self.fom.U.is_parametric:
                U = EuclideanInnerProduct(self.fom.U.shape[0])
            else:
                U = self.fom.U
                
        self.U_basis, Q = orthonormalize(self.U_basis, U)
        self._residual_evaluator.rotate_basis(Q)
        return Q
    
        
    def reconstruct(self, mu: Mu, u: Vector = None) -> Vector:
        r"""
        Reconstruct a full-order function from the reduced coefficients.
        
        Computes :math:`U_N u_N(\mu)` to obtain the full-order representation of the reduced solution :math:`u_N(\mu)`.
        
        Args:
            mu:
                Parameter value at which to reconstruct the solution.
            u:
                Optional reduced-order solution(s) :math:`(N,)` or :math:`(N,k)`. If ``None``, the reduced solution at :math:`\mu` is computed via :meth:`solve` and then reconstructed.
        
        Returns:
            :math:`(n,)` or :math:`(n,k)` reduced-order approximation of the full-order solution .
        """
        if u is None: u = self.solve(mu)
        return self.U_basis @ u
    
    
    def error(self, mu: Mu, u: Vector = None, u_fom: Vector = None, abs: bool = True, rel: bool = False) -> float:
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
        if not (abs or rel):
            raise ValueError("At least one of 'abs' or 'rel' must be True.")
        
        if u_fom is None: u_fom = self.fom.solve(mu)
        abs_err = self.fom.U.norm(mu, u_fom - self.reconstruct(mu, u))
        if rel:
            rel_err = abs_err / self.fom.U.norm(mu, u_fom)
            if abs: return abs_err, rel_err
            else: return rel_err
        else: return abs_err 
    
    def error_bound(self, mu: Mu, u: Vector = None) -> float:
        r"""Guaranteed a-posteriori upper bound of the absolute error.
        
        Just a shorthand for ``error_bounds(mu, u, aub=True)['aub']``, as this is the most common error quantity to evaluate, see `error_bounds` for details.
        """
        return self.error_bounds(mu, u, aub=True)['aub']
        
    def error_bounds(self, mu: Mu, u: Vector = None, aub: bool = False, rub: bool = False, alb: bool = False, rlb: bool = False) -> dict:
        r"""Guaranteed a-posteriori error bounds on the reduced solution.
        
        For :math:`\mu`, let :math:`u(\mu)` be the full-order solution, :math:`u_N(\mu)` the reduced solution and :math:`U_N u_N(\mu)` the reconstructed reduced solution. Further, let :math:`\sigma_{\text{LB}}(\mu)` and :math:`\gamma_{\text{UB}}(\mu)` be lower and upper bounds for the stability and continuity constants of the full-order model, respectively, and :math:`r(\mu) = f(\mu) - B(\mu) U_N u_N(\mu)` the residual.
        
        Then, the following absolute error bounds are available:
        
        .. math::
            \frac{\|r(\mu)\|_{V'}}{\gamma_{\text{UB}}(\mu)} \leq \|u(\mu) - U_N u_N(\mu)\|_U \leq \frac{\|r(\mu)\|_{V'}}{\sigma_{\text{LB}}(\mu)}
            
        And the following relative error bounds are available:
        
        .. math::
            \frac{\|r(\mu)\|_{V'}}{\gamma_{\text{UB}}(\mu)}\frac{\sigma_{\text{LB}}(\mu)}{\|f(\mu)\|_{V'}} \leq \frac{\|u(\mu) - U_N u_N(\mu)\|_U}{\|u(\mu)\|_U} \leq \frac{\|r(\mu)\|_{V'}}{\sigma_{\text{LB}}(\mu)}\frac{\gamma_{\text{UB}}(\mu)}{\|f(\mu)\|_{V'}}
        
        Args:
            mu:
                Parameter value at which to estimate the error.
            u:
                Reduced-order solution vector :math:`(N,)`. If ``None``, computed via :meth:`solve`.
            aub:
                If ``True``, compute the absolute upper bound.
            rub:
                If ``True``, compute the relative upper bound.
            alb:
                If ``True``, compute the absolute lower bound (only available if the continuity estimator is a `EfficientConstantEstimator`).
            rlb:
                If ``True``, compute the relative lower bound (only available if the stability and continuity estimator are `EfficientConstantEstimator`).
        
        Returns:
            Dictionary containing the computed error bounds. The keys are 'aub', 'rub', 'alb' and 'rlb' for each requested bound.
            
        .. note::
            Due to the square-root effect, the bounds are only accurate up to ``sqrt(eps)`` where ``eps`` is the machine precision. Thus, for smaller errors, the lower bounds might be wrong and the upper bounds might be overestimated.
        """
            
        if aub or rub or rlb:
            beta_LB = self._fom_stability_estimator.lower_bound(mu)
        if alb or rub or rlb:
            gamma_UB = self._fom_continuity_estimator.upper_bound(mu)
            
        if rlb or rub:
            f = self._residual_evaluator.dual_norm_rhs(mu)
        
        if u is None: u = self.solve(mu)
        r = self._residual_evaluator.dual_norm(mu, u)
            
        err = dict()
        if aub: err['aub'] = r / beta_LB
        if alb: err['alb'] = r / gamma_UB
        if rub: err['rub'] = gamma_UB/beta_LB * r/f
        if rlb: err['rlb'] = beta_LB/gamma_UB * r/f
        return err

        
    def output_error(self, mu: Mu, u: Vector = None, u_fom: Vector = None) -> float:
        r"""Compute the true error in the output of interest.
        
        Computes :math:`|l(\mu) u(\mu) - l_N(\mu) u_N(\mu)|` where :math:`u(\mu)` is the
        full-order solution and :math:`u_N(\mu)` is the reduced output of interest.
        
        Args:
            mu:
                Parameter value at which to compute the error.
            u:
                Reduced-order solution vector :math:`(N,)`. If ``None``, computed via :meth:`solve`.
            u_fom:
                Full-order solution vector :math:`(n,)`. If ``None``, computed via :meth:`fom.solve`.
        
        Returns:
            True error in the output of interest.
        """
        return abs(self.fom.output(mu, u_fom) - self.output(mu, u))
    
    def output_error_bound(self, mu: Mu, u: Vector = None) -> float:
        r"""Guaranteed a-posteriori upper bound of the output error.
        
        Computes the error bound
        
        .. math::
            |l(\mu) u(\mu) - l_N(\mu) u_N(\mu)| \leq \|l(\mu)\|_{U'} \frac{\|r(\mu)\|_{V'}}{\sigma_{\text{LB}}(\mu)}
            
        where :math:`r(\mu) = f(\mu) - B(\mu) U_N u_N(\mu)` is the residual and :math:`\sigma_{\text{LB}}(\mu)` is a lower bound for the stability constant of the full-order model.
        
        .. note::
            For an better error bound, consider `PrimalDualROM` / `PrimalDualGalerkinROM`.
        
        Args:
            mu:
                Parameter value at which to compute the error bound.
            u:
                Reduced-order solution vector :math:`(N,)`. If ``None``, computed via :meth:`solve`.
                
        Returns:
            Guaranteed upper bound of the output error.
        """
        return self._residual_evaluator.dual_norm_output(mu) * self.error_bound(mu, u)


class GalerkinROM(GalerkinFOM[Mu], ROM[Mu]):
    r"""
    Reduced-order Galerkin model.
    
    A Galerkin reduced-order model is a special `ROM` for `GalerkinFOM`, where the trial space :math:`U` and test space :math:`V` are equal and thus also the reduced bases :math:`U_N` and :math:`V_N`, i.e. the trial-to-test operator of the reduced model is the identity.
    """
    
    @property
    def V_basis(self) -> AffineLinear[Mu, Vector]:
        if self.U_basis is None: return None
        return wrap_affinelinear(self.U_basis)
    
    def __init__(self, 
                 fom: GalerkinFOM[Mu],
                 stability: StabilityEstimator[Mu],
                 continuity: ContinuityEstimator[Mu] = None,
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = None,
                 residual: None | str = None):
        r"""
        Args:
            fom:
                Full-order model to reduce.
            stability:
                Estimator for the FOM stability constant. Required for online-efficient residual-based error bounds, see `error_bound` and `error_bounds` (expept for ``alb``).
            continuity:
                Estimator for the FOM continuity constant. Not required for `error_bound`, but for all other online-efficient error bounds, see `error_bounds` (exept for ``aub``). Provide an online-efficient implementation if you want to use any of thes error bounds in an online efficient way. If ``None``, defaults to `ExactContinuity`.
            solver:
                Solver for the reduced linear system. Defaults to :class:`DirectSolver`.
            residual: 'affine' or 'full' or None
                Method to evaluate the residual. Required for online-efficient residual-based error bounds, see `error_bound` and `error_bounds`. The option 'affine' is only applicable if the test space inner product is parameter independent. This method exploits the affine structure of the problem is thus online-efficent. The option 'full' is a fall back that delegates all calculations to the full-order model and is thus not online-efficient. If ``None``, defaults to 'affine' if the FOM test space inner product is parameter independent and to 'full' otherwise.
        """
        if not isinstance(fom, GalerkinFOM):
            raise ValueError("The FOM must be a GalerkinFOM for a GalerkinROM.")
        trial2test = lambda basis: wrap_affinelinear(basis)
        ROM.__init__(self, fom, stability, continuity, trial2test, solver, residual)
        
    def _mark_for_assembly(self, U = False, V = False):
        super()._mark_for_assembly(U, U)

    def add_basis(self, basis: AffineLinear[Mu, Vector] | Vector):
        self._add_basis_U(basis)
    
    def orthonormalize(self, U: InnerProduct[Mu] | Matrix | None = None):
        self._orthonormalize_U(U)
        
        
class _PrimalDualROM_Mixin(PrimalDualModel[Mu]):
    
    dual: ROM[Mu]
    """Dual model."""

    _B_mixed: AffineLinear[Mu, Matrix] | None = None
    _B_mixed_private: AffineLinear[Mu, Matrix] | None = None
    _need_assemble_B_mixed: bool = False
    r"""Flag to indicate whether the mixed operator needs to be reassembled."""

    @property
    def _B_mixed(self) -> AffineLinear[Mu, Matrix]:
        self._assemble_B_mixed()
        return self._B_mixed_private
    
    def __init__(self: PrimalDualROM[Mu], dual: ROM[Mu]):
        self.dual = dual
        self.dual._fom_stability_estimator  = self._fom_stability_estimator
        self.dual._fom_continuity_estimator = self._fom_continuity_estimator
        
        old_mark_for_assembly = self._mark_for_assembly
        def patched_mark_for_assembly(U: bool = False, V: bool = False):
            old_mark_for_assembly(U, V)
            if U: self._need_assemble_B_mixed = True
        self._mark_for_assembly = patched_mark_for_assembly
        
        dual_old_mark_for_assembly = self.dual._mark_for_assembly
        def dual_patched_mark_for_assembly(U: bool = False, V: bool = False):
            dual_old_mark_for_assembly(U, V)
            if U: self._need_assemble_B_mixed = True
        self.dual._mark_for_assembly = dual_patched_mark_for_assembly
    
    def _assemble_B_mixed(self):
        if self._need_assemble_B_mixed:
            self._need_assemble_B_mixed = False
            if self.dual.U_basis is not None and self.U_basis is not None:
                self._B_mixed_private = self.dual.U_basis.T @ self.fom.B @ self.U_basis
            else:
                self._B_mixed_private = None
                
    def assemble(self):
        ROM.assemble(self)
        self.dual.assemble()
        self._assemble_B_mixed()

    def output(self: PrimalDualROM[Mu], mu: Mu, u: Vector | None = None, z: Vector | None = None) -> Vector:
        if self.l is None:
            raise ValueError("No output functional defined for this model.")
        if u is None: u = self.solve(mu)
        if z is None: z = self.dual.solve(mu)
        return (self.l(mu) @ u # Classical primal output
                + self.dual.output(mu, z) + z.T @ self._B_mixed(mu) @ u) # Primal dual correction (primal residual evaluated at dual solution)
    
    def output_error_bound(self: PrimalDualROM[Mu], mu: Mu, u: Vector = None, z: Vector = None) -> float:
        if u is None: u = self.solve(mu)
        if z is None: z = self.dual.solve(mu)
        sigma = self._fom_stability_estimator.lower_bound(mu)
        r_primal = self._residual_evaluator.dual_norm(mu, u)
        r_dual = self.dual._residual_evaluator.dual_norm(mu, z)
        return r_primal * r_dual / sigma
        
        
class PrimalDualROM(_PrimalDualROM_Mixin[Mu], ROM[Mu]):
    r"""Primal-dual reduced-order Petrov-Galerkin model given, see `ROM` and `PrimalDualModel`.
    """
    
    def __init__(self, 
                 fom: PrimalDualModel[Mu],
                 stability: StabilityEstimator[Mu],
                 continuity: ContinuityEstimator[Mu] = None,
                 primal_trial2test: Callable[[Vector], AffineLinear[Mu, Vector]] = None,
                 dual_trial2test: Callable[[Vector], AffineLinear[Mu, Vector]] = None,
                 solver: list[Solver | Callable[[Matrix, Vector, Vector|None], Vector]] | Solver | Callable[[Matrix, Vector, Vector|None], Vector] = [None, None],
                 residual: list[None | str] = [None, None]):
        r"""
        Args:
            fom:
                Full-order model to reduce.
            stability:
                Estimator for the primal and dual full-order stability constant (they coincide). See `ROM` and `PrimalDualModel` for details.
            continuity:
                Estimator for the primal and dual full-order continuity constant (they coincide). See `ROM` and `PrimalDualModel` for details.
            primal_trial2test:
                Trial-to-Test operator for the primal model. See `ROM` and `PrimalDualModel` for details.
            dual_trial2test:
                Trial-to-Test operator for the dual model. See `ROM` and `PrimalDualModel` for details.
            solver:
                List of two solvers for the primal and dual model, respectively. Alternatively, a single solver which is used for both models. See also `ROM` and `PrimalDualModel`.
            residual:
                List of two residual evaluation methods. See `ROM` and `PrimalDualModel`.
        """
        
        try:
            solver = list(solver)
        except TypeError:
            solver = [solver, solver]
            
        if len(solver) != 2:
            raise ValueError("Solver must be a list of two solvers for primal and dual.")
        
        ROM.__init__(self, fom, stability, continuity, primal_trial2test, solver[0], residual[0])
        
        dual = ROM(fom.dual, ExactStability(fom.dual), ExactContinuity(fom.dual), dual_trial2test, solver[1], residual[1])
        
        super().__init__(dual)
        
        
class PrimalDualGalerkinROM(_PrimalDualROM_Mixin[Mu], GalerkinROM[Mu]):
    r"""Primal-dual reduced-order Galerkin model given, see `GalerkinROM` and `PrimalDualModel`.
    """

    dual: GalerkinROM[Mu]
    """Dual model."""
    
    def __init__(self, 
                 fom: PrimalDualModel[Mu],
                 stability: StabilityEstimator[Mu],
                 continuity: ContinuityEstimator[Mu] = None,
                 solver: list[Solver | Callable[[Matrix, Vector, Vector|None], Vector]] | Solver | Callable[[Matrix, Vector, Vector|None], Vector] = [None, None],
                 residual: list[None | str] = [None, None]):
        r"""
        Args:
            fom:
                Full-order model to reduce.
            stability:
                Estimator for the primal and dual full-order stability constant (they coincide). See `GalerkinROM` and `PrimalDualModel` for details.
            continuity:
                Estimator for the primal and dual full-order continuity constant (they coincide). See `GalerkinROM` and `PrimalDualModel` for details.
            solver:
                List of two solvers for the primal and dual model, respectively. Alternatively, a single solver which is used for both models. See also `GalerkinROM` and `PrimalDualModel`.
            residual:
                List of two residual evaluation methods. See `GalerkinROM` and `PrimalDualModel`.
        """
        
        try:
            solver = list(solver)
        except TypeError:
            solver = [solver, solver]
            
        if len(solver) != 2:
            raise ValueError("Solver must be a list of two solvers for primal and dual.")
        
        GalerkinROM.__init__(self, fom, stability, continuity, solver[0], residual[0])
        
        dual = GalerkinROM(fom.dual, ExactStability(fom.dual), ExactContinuity(fom.dual), solver[1], residual[1])
        
        super().__init__(dual)