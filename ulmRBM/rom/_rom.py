
from __future__ import annotations

__all__ = [
    'ROM',
    'GeneralROM',
    'Trial2TestROM',
    'GalerkinROM',
]

from collections.abc import Callable

import numpy as np

from ulmRBM.core import Mu, Matrix, Vector, NO_MU
from ulmRBM.solver import Solver, DirectSolver
from ulmRBM.fom import FOM, GalerkinFOM
from ulmRBM.products import InnerProduct, EuclideanInnerProduct, orthonormalize
from ulmRBM.affine import AffineLinear, wrap_affinelinear

from ._constants import StabilityEstimator, ContinuityEstimator, StabilityOptions, ContinuityOptions
from ._residual import ResidualCalculator, ResidualOptions

# maybe add a global flag to warn if any computations inside the rom use full-order dimensions


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
    
    where :math:`B_N(\mu) = V_N^T B(\mu) U_N`, :math:`f_N(\mu) = V_N^T f(\mu)`
    and :math:`l_N(\mu) = l(\mu) U_N`,
    with :math:`U_N \in \mathbb{R}^{n \times N}` the reduced trial basis and
    :math:`V_N \in \mathbb{R}^{m \times M}` the reduced test basis.
    Then, :math:`u_N \in \mathbb{R}^N` is the reduced solution vector and :math:`U_N u_N \in \mathbb{R}^n`
    the corresponding approximation in the full-order solution, while :math:`s_N(\mu)\in\mathbb{R}^p` 
    are the output(s) of interest of the reduced solution. 
    """
    
    #Extension to affine trial space should be easy be extending the functionallity of ResidualCalculator and reusing existing code for the test space.
    
    fom: FOM[Mu]
    """Underlying full-order model."""
    U_basis: AffineLinear[Mu, Vector] | None
    """:math:`(n,N)` trial space basis matrix, where :math:`N` is the dimension of the reduced trial space."""
    V_basis: AffineLinear[Mu, Vector] | None
    """:math:`(m,M)` test space basis matrix, where :math:`M` is the dimension of the reduced test space."""
    estimate_fom_stability: StabilityEstimator[Mu]
    r"""Estimator :math:`\beta_{\text{LB}}(\mu)` for the FOM stability constant, required for error estimation."""
    estimate_fom_continuity: ContinuityEstimator[Mu]
    r"""Estimator :math:`\gamma_{\text{UB}}(\mu)` for the FOM continuity constant."""
    residual: ResidualCalculator[Mu]
    r"""Calculator for the dual norm of the residual, required for error estimation."""
    
    _U_basis: AffineLinear[Mu, Vector] | None = None
    _V_basis: AffineLinear[Mu, Vector] | None = None
    
    _need_assemble: bool
    r"""Flag to indicate whether the reduced system matrices and vectors need to be reassembled."""
    
    @property
    def B(self) -> AffineLinear[Mu, Matrix]:
        if self._need_assemble: self._assemble_rom()
        return self._B
    @B.setter
    def B(self, value: AffineLinear[Mu, Matrix]):
        self._B = value
        
    @property
    def f(self) -> AffineLinear[Mu, Matrix]:
        if self._need_assemble: self._assemble_rom()
        return self._f
    @f.setter
    def f(self, value: AffineLinear[Mu, Matrix]):
        self._f = value
        
    @property
    def l(self) -> AffineLinear[Mu, Matrix] | None:
        if self._need_assemble: self._assemble_rom()
        return self._l
    @l.setter
    def l(self, value: AffineLinear[Mu, Matrix] | None):
        self._l = value

    @property
    def U_basis(self) -> AffineLinear[Mu, Vector]:
        return self._U_basis
    @U_basis.setter
    def U_basis(self, value: AffineLinear[Mu, Vector]):
        self._U_basis = wrap_affinelinear(value)
        self._need_assemble = True
        self.U = self.fom.U.restrict(self._U_basis)
        
    @property
    def V_basis(self) -> AffineLinear[Mu, Vector]:
        return self._V_basis
    @V_basis.setter
    def V_basis(self, value: AffineLinear[Mu, Vector]):
        self._V_basis = wrap_affinelinear(value)
        self._need_assemble = True
        self.V = self.fom.V.restrict(self._V_basis)
        
    @property
    def estimate_fom_continuity(self) -> ContinuityEstimator[Mu]:
        return self._estimate_fom_continuity
    @estimate_fom_continuity.setter
    def estimate_fom_continuity(self, estimator: ContinuityEstimator[Mu] | None):
        if estimator is not None:
            estimator.fom = self.fom
        self._estimate_fom_continuity = estimator
        
    @property
    def estimate_fom_stability(self) -> StabilityEstimator[Mu]:
        return self._estimate_fom_stability
    @estimate_fom_stability.setter
    def estimate_fom_stability(self, estimator: StabilityEstimator[Mu] | None):
        if estimator is not None:
            estimator.fom = self.fom
        self._estimate_fom_stability = estimator
        
    @property
    def residual(self) -> ResidualCalculator[Mu]:
        return self._residual
    @residual.setter
    def residual(self, residual: ResidualCalculator[Mu] | None):
        if residual is not None:
            residual.set(self)
        self._residual = residual
        
    @property
    def dim(self):
        return (0 if self.U_basis is None else self.U_basis.shape[1],
                0 if self.V_basis is None else self.V_basis.shape[1],
                self.fom.dim[2])

    
    def __init__(self, 
                 fom: FOM[Mu],
                 U_basis: AffineLinear[Mu, Vector] | Vector = None,
                 V_basis: AffineLinear[Mu, Vector] | Vector = None,
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
                warn("FOM has parameter-dependent test space inner product, using 'DirectResidual' for residual computation as the residula is not affine. Thus, error estimation IS NOT online-efficient.")
            else:
                residual = ResidualOptions.AFFINE
        if not isinstance(residual, ResidualCalculator):
            residual = ResidualOptions(residual).get()
        self.residual = residual
        
        self.solver = solver
        
        if U_basis is not None: self._add_basis_U(U_basis)
        if V_basis is not None: self._add_basis_V(V_basis)
    
        
    def _assemble_rom(self):
        r"""Assemble the reduced-order model system matrices and vectors based on the current trial and test bases."""
        self.B = None
        self.f = None
        self.l = None
        if self.V_basis is not None:
            self.f = self.V_basis.T @ self.fom.f
            if self.U_basis is not None:
                self.B = self.V_basis.T @ self.fom.B @ self.U_basis
        if self.U_basis is not None and self.fom.l is not None:
            self.l = self.fom.l @ self.U_basis
        self._need_assemble = False
    
    def _add_basis_U(self, basis: AffineLinear[Mu, Vector] | Vector):
        r"""Add new basis vectors to the trial space basis.
        
        Residual calculator is updated accordingly.
        
        Args:
            basis:
                New trial basis vectors :math:`(n, k)` to append.
        """
        basis = wrap_affinelinear(basis)
        if len(basis.shape) == 1:
            basis = basis.apply2data(lambda dq: dq.reshape(-1,1))
        if basis.shape[0] != self.fom.dim[0]:
            raise ValueError("Basis vector has incompatible dimension.")
        if self.U_basis is None:
            U_basis = basis
        else:
            if len(self.U_basis) != len(basis):
                raise ValueError("Cannot add basis with different number of affine terms.")
            U_basis = AffineLinear()
            for ((theta, U_old), U_new) in zip(self.U_basis, basis.data):
                U_basis += [(theta, np.hstack([U_old, U_new]))]
        self.U_basis = U_basis
        self.residual.add_basis(basis)
        
    def _add_basis_V(self, basis: AffineLinear[Mu, Vector] | Vector):
        r"""Add new basis vectors to the test space basis.
        
        Args:
            basis:
                New test basis vectors :math:`(m, k)` to append.
        """
        basis = wrap_affinelinear(basis)
        if len(basis.shape) == 1: 
            basis = basis.apply2data(lambda dq: dq.reshape(-1,1))
        if basis.shape[0] != self.fom.dim[0]:
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
        
        
    def _orthonormalize_U(self, full: bool, U: InnerProduct[Mu] | Matrix | None) -> np.ndarray:
        """
        Orthonormalize the trial space basis with respect to a specified inner product.
        
        Performs orthonormalization using a parameter-independent
        inner product. This improves numerical stability and ensures well-conditioned
        reduced system matrices.
        
        This is only possible if the trial basis is parameter-independent.
        
        Args:
            full:
                If ``True``, the residual calculation is performed from scratch and the option is passed to `ulmRBM.products.orthonormalize`. This is more stable but expensive. 
                If ``False``, update existing residual data incrementally. This is faster but might accumulate errors over time.
            U:
                Parameter-independent inner product for trial space orthonormalization.
                If ``None`` and :attr:`fom.U` is parameter-independent, uses :attr:`fom.U`.
                Otherwise uses Euclidean inner product.
        """
        if self.U_basis.is_parametric:
            raise ValueError("Trial basis is parameter-dependent, cannot orthonormalize.")
        
        if U is None: 
            if self.fom.U.is_parametric:
                U = EuclideanInnerProduct(self.fom.U.shape[0])
            else:
                U = self.fom.U
                
        self.U_basis, Q = orthonormalize(self.U_basis(NO_MU), U, full)
        
        if full:
            self.residual.set_basis(self.U_basis)
        else:
            self.residual.rotate_basis(Q)
            
        return Q
    
    def _orthonormalize_V(self, full: bool, V: InnerProduct[Mu] | Matrix | None) -> np.ndarray:
        """
        Orthonormalize the test space basis with respect to a specified inner product.
        
        Performs orthonormalization using a parameter-independent
        inner product. This improves numerical stability and ensures well-conditioned
        reduced system matrices.
        
        This is only possible if the test basis is parameter-independent.
        
        Args:
            full:
                See `ulmRBM.products.orthonormalize`.
            V:
                Parameter-independent inner product for test space orthonormalization.
                If ``None`` and :attr:`fom.V` is parameter-independent, uses :attr:`fom.V`.
                Otherwise uses Euclidean inner product.
        """
        if self.V_basis.is_parametric:
            raise ValueError("Test basis is parameter-dependent, cannot orthonormalize.")
        
        if V is None:
            if self.fom.V.is_parametric:
                V = EuclideanInnerProduct(self.fom.V.shape[0])
            else:
                V = self.fom.V
                          
        self.V_basis, Q = orthonormalize(self.V_basis(NO_MU), V, full)
        
        return Q
    
    
    def error(self, mu: Mu, u: Vector = None, u_fom: Vector = None) -> float:
        r"""
        Compute the true error between reduced and full-order solutions.
        
        Computes :math:`\|u(\mu) - U_N(\mu) u_N(\mu)\|_U` where :math:`u(\mu)` is the
        full-order solution and :math:`U_N(\mu) u_N(\mu)` is the reconstructed reduced solution.
        
        Args:
            mu:
                Parameter value at which to compute the error.
            u:
                Reduced-order solution vector :math:`(N,)`. If ``None``, computed via :meth:`solve`.
            u_fom:
                Full-order solution vector :math:`(n,)`. If ``None``, computed via :meth:`fom.solve`.
        
        Returns:
            True error :math:`\|u(\mu) - U_N(\mu) u_N(\mu)\|_U`.
        """
        if u_fom is None: u_fom = self.fom.solve(mu)
        return self.fom.U.norm(mu, u_fom - self.reconstruct(mu, u))
    
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
        return self.residual.dual_norm(mu, u) / self.estimate_fom_stability(mu)

    def reconstruct(self, mu: Mu, u: Vector = None) -> Vector:
        r"""
        Reconstruct a full-order function from the reduced coefficients.
        
        Computes :math:`U_N(\mu) u_N` to obtain the full-order representation.
        
        Args:
            mu:
                Parameter value at which to reconstruct the solution.
            u:
                Optional reduced-order coefficients :math:`(N,)` or :math:`(N,k)`. If ``None``, the reduced solution at :math:`\mu` is computed via :meth:`solve` and used for reconstruction.
        
        Returns:
            Reconstructed full-order function :math:`(n,)` or :math:`(n,k)`.
        """
        if u is None: u = self.solve(mu)
        return self.U_basis(mu) @ u
    
    
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
        
        
class GeneralROM(ROM[Mu]):
    r"""
    General reduced-order model where trial and test space are controlled independently.
    
    See `ROM`
    """
        
    add_basis_U = ROM._add_basis_U
    add_basis_V = ROM._add_basis_V
    orthonormalize_U = ROM._orthonormalize_U
    orthonormalize_V = ROM._orthonormalize_V
        

class Trial2TestROM(ROM[Mu]):
    r"""
    Reduced-order Petrov-Galerkin model with explicit trial to test space relation.
    
    In this reduced-order model, the test space is given by :math:`V_N(\mu) := T(\mu) U_N(\mu)` where
    :math:`U_N` is the reduced trial space and :math:`T(\mu) : U \to V` is a provided trial-to-test mapping function. Typically, :math:`T(\mu)` is the supremizing operator of the underlying full-order model, see `FOM.supremizer`.
    For a documentation of the general ROM formulation, see `ROM`.
    """
    
    _trial2test_fun: Callable[[Vector], AffineLinear[Mu, Vector]] | None = None
    r"""Internal storage if a custom trial2test function is provided."""
    
    def __init__(self, 
                 fom: FOM[Mu],
                 U_basis: AffineLinear[Mu, Vector] | Vector = None,
                 trial2test: Callable[[AffineLinear[Mu, Vector]], AffineLinear[Mu, Vector]] = None,
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
                Function mapping trial basis vectors to test basis vectors, i.e. ``V_basis = trial2test(U_basis)``. If ``None``, and the test space inner product is parameter-independent, the supremizer of the full-order model is used, which will always yield a stable ROM.
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
        
    def trial2test(self, basis: AffineLinear[Mu, Vector]) -> AffineLinear[Mu, Vector]:
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
        
    def add_basis(self, basis: AffineLinear[Mu, Vector] | Vector):
        r"""Add new trial basis vectors and update the test basis via trial-2-test relation.
        
        Args:
            basis:
                New trial basis vectors :math:`(n, k)` to append.        
        """
        basis = wrap_affinelinear(basis)
        self._add_basis_U(basis)
        self._add_basis_V(self.trial2test(basis))
    
    def orthonormalize(self, full: bool = False, U: InnerProduct[Mu] | Matrix | None = None):
        """
        Orthonormalize the trial space basis with respect to a specified inner product.
        
        Performs orthonormalization using a parameter-independent
        inner product. This improves numerical stability and ensures well-conditioned
        reduced system matrices.
        
        This is only possible if the trial basis is parameter-independent.
        
        Args:
            full:
                If ``True``, the residual and test basis calculation is performed from scratch and the option is passed to `ulmRBM.products.orthonormalize`. This is more stable but expensive. 
                If ``False``, update existing residual data and test space incrementally. This is faster but might accumulate errors over time.
            U:
                Parameter-independent inner product for trial space orthonormalization.
                If ``None`` and :attr:`fom.U` is parameter-independent, uses :attr:`fom.U`.
                Otherwise uses Euclidean inner product.
        """        
        QU = self._orthonormalize_U(full, U)
        if full:
            self.V_basis = self.trial2test(self.U_basis)
        else:
            self.V_basis = self.V_basis @ QU
        

class GalerkinROM(GalerkinFOM[Mu], Trial2TestROM[Mu]):
    r"""
    Reduced-order Galerkin model.
    
    A Galerkin reduced-order model is a special case of `Trial2TestROM`, where :math:`V_N = U_N`.
    """
        
    def __init__(self, 
                 fom: GalerkinFOM[Mu],
                 U_basis: AffineLinear[Mu, Vector] | Vector = None,
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
        trial2test = lambda basis: basis
        Trial2TestROM.__init__(self, fom, U_basis, trial2test, stability, continuity, residual, solver)

    def add_basis(self, basis: AffineLinear[Mu, Vector] | Vector):
        self._add_basis_U(basis)
    
    def orthonormalize(self, full: bool = False, U: InnerProduct[Mu] | Matrix | None = None):
        self._orthonormalize_U(full, U)