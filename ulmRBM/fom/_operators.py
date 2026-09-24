"""
Parametric operators with associated trial and test spaces.
"""

from __future__ import annotations


__all__ = [
    'ParametricOperator',
    'ParametricGalerkinOperator',
]

from collections.abc import Callable
from typing import Generic

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, LinearOperator, aslinearoperator, onenormest

from ulmRBM.core import (
    NO_MU, Mu, Matrix, ParametricLinear, Vector, wrap_scalar,
)
from ulmRBM.solver import IterativeSolver, DirectSolver, Solver, wrap_solver
from ulmRBM.affine import AffineLinear, wrap_affinelinear
from ulmRBM.products import InnerProduct, OperatorInnerProduct



class ParametricOperator(Generic[Mu]):
    r"""
    Affine Petrov-Galerkin Operator :math:`B(\mu) : U \to V'`.
    
    For two spaces :math:`U` and :math:`V` of dimension :math:`n` and :math:`m` and (possibly parameter-dependent) inner products :math:`(\cdot, \cdot)_U` and :math:`(\cdot, \cdot)_V`, the operator :math:`B(\mu)` is matrix of shape :math:`(m, n)`, which is affine with respect to the parameter :math:`\mu`.
    """
    
    shape: tuple[int, int]
    r"""``(m, n)``, shape of the operator matrix :math:`B(\mu)`."""
    
    B: AffineLinear[Mu, Matrix]
    r"""Affine decomposition of the operator matrix :math:`B(\mu) = \sum_{q=1}^Q \theta_q^B(\mu) B_q`."""
        
    U: InnerProduct[Mu]
    r"""Inner product :math:`(\cdot, \cdot)_U` on the trial space :math:`U`."""
    
    V: InnerProduct[Mu]
    r"""Inner product :math:`(\cdot, \cdot)_V` on the test space :math:`V`."""
        
    _stability_solver: Callable[[Mu, ParametricOperator[Mu]],float] | str
    """Optional explicit function to compute the stability constant, bypassing the default eigenvalue-based computation."""

    _continuity_solver: Callable[[Mu, ParametricOperator[Mu]],float] | str
    """Optional explicit function to compute the continuity constant, bypassing the default eigenvalue-based computation."""
    
    _take_square_root_eigenvalues: bool = True
    """needed for the Petrov-Galerkin and Galerkin cases"""
    
    _supremizer_func: Callable[[Vector, ParametricOperator[Mu]], AffineLinear[Mu, Vector]] | None = None
    r"""Internal storage if a custom supremizer function is provided."""
    
    _eigsh_options_stability: dict = {
        'v0': None,
        'ncv': None,
        'maxiter': None,
        'tol': 1e-10,
        'rng': None
    }
    r"""Options for the eigenvalue solver used in the default sparse stability constant computations. If neccessary, these can be updated by the user after initialization, e.g. ``op._eigsh_options['tol'] = 1e-8``."""
    
    _eigsh_options_continuity: dict = {
        'v0': None,
        'ncv': None,
        'maxiter': None,
        'tol': 1e-10,
        'rng': None
    }
    r"""Options for the eigenvalue solver used in the default sparse continuity constant computations. If neccessary, these can be updated by the user after initialization, e.g. ``op._eigsh_options['tol'] = 1e-8``."""
    
    @property
    def shape(self):
        return self.B.shape
    
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix,
                 U: InnerProduct[Mu],
                 V: InnerProduct[Mu],
                 stability:  Callable[[Mu, ParametricOperator[Mu]], float] | float | str = None,
                 continuity: Callable[[Mu, ParametricOperator[Mu]], float] | float | str = None,
                 supremizer: Callable[[Vector, ParametricOperator[Mu]], ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]] = None):
        r"""
        Args:
            B:
                Affine decomposition of the operator matrix.
            U:
                Inner product on the trial space.
            V:
                Inner product on the test space.
            stability:
                Parameter how the stability constant is computed. Might take the values ``'iterative'`` (default), ``'direct'``, ``'estimate'``, a scalar number ``s`` or a custom callable with signature ``s,x = stability(mu, op)`` or ``s = stability(mu, op)``.  Defaults to ``'iterative'``.
                
                If set to ``'iterative'``, the stability constant is computed via the iterative solver `scipy.sparse.linalg.eigsh` (you might tweak its options by changing ``op._eigsh_options_stability``). If set to ``'direct'``, the stability constant is computed via the dense solver `scipy.linalg.eigvals`. If set to ``'estimate'``, the stability  constant is estimated by the lower bound :math:`\sqrt{1/\|A^{-1}M\|_1}` using `scipy.sparse.linalg.onenormest`.
                
                Explanation to the values ``s`` and ``x``:
                Consider the generalized eigenvalue problem :math:`A x = \lambda M x`, with :math:`A := B(\mu)^T V(\mu)^{-1} B(\mu)` and :math:`M := U(\mu)`, with the operator matrix :math:`B(\mu)` and the inner product matrices :math:`U(\mu)` and :math:`V(\mu)` on trial and test space respectively. The stability constant is then given by squareroot of the smallest eigenvalue. Thus, ``s`` is :math:`\sqrt{\lambda_{\text{min}}}` and ``x`` the corresponding eigenvector.
            continuity:
                See the description of the ``stability`` parameter, with the only difference, that the continuity constant is given by the squareroot of the largest eigenvalue instead of the smallest. Thus, if ``'estimate'`` is selected, the continuity constant is estimated by the upper bound :math:`\sqrt{\|M^{-1}A\|_1}`.  If ``'iterative'`` was selected, you might tweak the options of the underlying eigenvalue solver by changing ``op._eigsh_options_continuity``.
            supremizer:
                Custom function for the supremizing operator. If ``None``, a default implementation is used. See :meth:`supremizer`.
        """
        
        if stability is None: stability = 'iterative'
        if continuity is None: continuity = 'iterative'
        
        if B.shape[1] != U.shape[0]:
            raise ValueError("B and U must have compatible dimensions.")
        if B.shape[0] != V.shape[0]:
                raise ValueError("B and V must have compatible dimensions.")
        
        if isinstance(stability, str):
            if stability not in ['direct', 'iterative', 'estimate']:
                raise ValueError("Invalid value for 'stability' parameter. Expected 'direct', 'iterative', 'estimate' or a callable.")
        else:
            stability = wrap_scalar(stability)
        
        if isinstance(continuity, str):
            if continuity not in ['direct', 'iterative', 'estimate']:
                raise ValueError("Invalid value for 'continuity' parameter. Expected 'direct', 'iterative', 'estimate' or a callable.")
        else:
            continuity = wrap_scalar(continuity)
            
        self.B = wrap_affinelinear(B).compress()
        self.U = U
        self.V = V
        self._stability_solver = stability
        self._continuity_solver = continuity
        self._supremizer_func = supremizer
        self._eigsh_options_stability  = self._eigsh_options_stability.copy()
        self._eigsh_options_continuity = self._eigsh_options_continuity.copy()
        
    def __repr__(self):
        shape = f"({self.shape[0]}"
        for s in self.shape[1:]: 
            if s is None: s = 0
            shape += f", {s}"
        shape += ")"
        return f"<{self.__class__.__name__} of shape {shape}>"
    
    
    def stability(self, mu: Mu, eigenvector: bool = False) -> float:
        r"""
        Compute the stability constant at parameter value :math:`\mu`. 
        
        For Galerkin operators, this is the coercivity constant:
        
        .. math::
            \beta(\mu) = \inf_{u \in U} \frac{| \langle B(\mu) u, u \rangle_{U'\times U} |}{\|u\|_U^2}
            
        For Petrov-Galerkin operators, this is the inf-sup constant:
        
        .. math::
            \beta(\mu) = \inf_{u \in U} \sup_{v\in V} \frac{| \langle B(\mu) u, v \rangle_{V'\times V} |}{\|u\|_U \|v\|_V}
        
        Args:
            mu:
                Parameter value at which to compute the stability constant.
            eigenvector:
                Whether to return the corresponding eigenvector of the underlying eigenvalue problem defining the stability constant.
                
        Returns:
            Stability constant :math:`\beta(\mu) > 0`.
        """
        # for eigenvaector with only eigenvalue, probably use a shift
        if isinstance(self._stability_solver, str):
            A, M, Ainv, Minv = self._get_eigenvalue_operators(mu)
            
            if self._stability_solver == 'estimate':
                if eigenvector:
                    from warnings import warn
                    warn("Stability estimation does not provide an eigenvector, but 'eigenvector=True' was requested. Using default iterative implementation.")
                    val, vec = self._stability_sparse(A, M, Ainv, Minv)
                else:
                    val = self._stability_estimate(A, M, Ainv, Minv)
            
            elif self._stability_solver == 'direct':
                val, vec = self._stability_dense(A, M, Ainv, Minv)
                
            else:
                val, vec = self._stability_sparse(A, M, Ainv, Minv)
        
        else:
            val_vec = self._stability_solver(mu, self)
            try:
                val, vec = val_vec
            except:
                val = val_vec
                if eigenvector:
                    from warnings import warn
                    warn("Provided stability function does not return an eigenvector, but 'eigenvector=True' was requested. Using default iterative implementation.")
                    A, M, Ainv, Minv = self._get_eigenvalue_operators(mu)
                    val, vec = self._stability_sparse(A, M, Ainv, Minv)

        if eigenvector:
            return val, vec
        else:
            return val
        
    def continuity(self, mu: Mu, eigenvector: bool = False) -> float:
        r"""
        Compute the continuity constant at parameter value :math:`\mu`.
        
        .. math::
            \gamma(\mu) = \sup_{u \in U} \sup_{v\in V} \frac{| \langle B(\mu) u, v \rangle_{V'\times V} |}{\|u\|_U \|v\|_V}
        
        The constant is equivalent to the operator norm of :math:`\|B(\mu)\|_{L(U,V')}`.
        
        Args:
            mu:
                Parameter value at which to compute the continuity constant.
            eigenvector:
                Whether to also return the corresponding eigenvector of the underlying eigenvalue problem defining the continuity constant.
                
        Returns:
            Continuity constant :math:`\gamma(\mu) < \infty`.
        """
        if isinstance(self._continuity_solver, str):
            A, M, Ainv, Minv = self._get_eigenvalue_operators(mu)
            
            if self._continuity_solver == 'estimate':
                if eigenvector:
                    from warnings import warn
                    warn("Continuity estimation does not provide an eigenvector, but 'eigenvector=True' was requested. Using default iterative implementation.")
                    val, vec = self._continuity_sparse(A, M, Ainv, Minv)
                else:
                    val = self._continuity_estimate(A, M, Ainv, Minv)
            
            elif self._continuity_solver == 'direct':
                val, vec = self._continuity_dense(A, M, Ainv, Minv)
                
            else:
                val, vec = self._continuity_sparse(A, M, Ainv, Minv)
        
        else:
            val_vec = self._continuity_solver(mu, self)
            try:
                val, vec = val_vec
            except:
                val = val_vec
                if eigenvector:
                    from warnings import warn
                    warn("Provided continuity function does not return an eigenvector, but 'eigenvector=True' was requested. Using default sparse implementation.")
                    A, M, Ainv, Minv = self._get_eigenvalue_operators(mu)
                    val, vec = self._continuity_sparse(A, M, Ainv, Minv)
            
        if eigenvector:
            return val, vec
        else:
            return val
        
    def _get_eigenvalue_operators(self, mu: Mu):
        M = self.U(mu)
        Minv = self.U.dual(mu)
        if self.B.shape[0] == self.B.shape[1]:
            if hasattr(self, '_solver'):
                solver = self._solver
            else:
                solver = DirectSolver()
            A = OperatorInnerProduct(self.B, self.V.dual, solver)
            return A(mu), M, A.dual(mu), Minv
        else:
            A = self.V.dual.restrict(self.B(mu))
            return A(mu), M, None, Minv
        
    def _stability_dense(self, A, M, Ainv, Minv):
        if not isinstance(A, np.ndarray):
            A = A @ np.eye(A.shape[0])
        if not isinstance(M, np.ndarray):
            M = M @ np.eye(M.shape[0])
        eigs, vecs = eigh(A, M)
        val, vec = np.abs(eigs[0]), vecs[:,0]
        if self._take_square_root_eigenvalues:
            val = np.sqrt(val)
        return val, vec
    
    def _continuity_dense(self, A, M, Ainv, Minv):
        if not isinstance(A, np.ndarray):
            A = A @ np.eye(A.shape[0])
        if not isinstance(M, np.ndarray):
            M = M @ np.eye(M.shape[0])
        eigs, vecs = eigh(A, M)
        val, vec = np.abs(eigs[-1]), vecs[:,-1]
        if self._take_square_root_eigenvalues:
            val = np.sqrt(val)
        return val, vec


    def _stability_sparse(self, A, M, Ainv, Minv):
        eigsh_opts = self._eigsh_options_stability.copy()
        eigsh_opts['A'] = A
        eigsh_opts['M'] = M
        if Ainv is not None:
            eigsh_opts['OPinv'] = Ainv
        eigsh_opts['k'] = 1
        eigsh_opts['sigma'] = 0.0
        eigsh_opts['which'] = 'LA'
        val, vec = eigsh(**eigsh_opts)
        if self._take_square_root_eigenvalues:
            val = np.sqrt(val)
        return val[0], vec[:,0]

    def _continuity_sparse(self, A, M, Ainv, Minv):
        eigsh_opts = self._eigsh_options_continuity.copy()
        eigsh_opts['A'] = A
        eigsh_opts['M'] = M
        eigsh_opts['Minv'] = Minv
        eigsh_opts['k'] = 1
        eigsh_opts['which'] = 'LA'
        val, vec = eigsh(**eigsh_opts)
        if self._take_square_root_eigenvalues:
            val = np.sqrt(val)
        return val[0], vec[:,0]
    
    
    def _stability_estimate(self, A, M, Ainv, Minv):
        if Ainv is None:
            solver = IterativeSolver(spd=True)
            matmul = lambda x: solver(A, x)
            Ainv = LinearOperator(shape=A.shape, matvec=matmul, matmat=matmul, rmatvec=matmul, rmatmat=matmul, dtype=A.dtype)
        Ainv = aslinearoperator(Ainv)
        M = aslinearoperator(M)
        val = 1/onenormest(Ainv @ M)
        if self._take_square_root_eigenvalues:
            val = np.sqrt(val)
        return val
    
    def _continuity_estimate(self, A, M, Ainv, Minv):
        A = aslinearoperator(A)
        Min = aslinearoperator(Minv)
        val = onenormest(A @ Min)
        if self._take_square_root_eigenvalues:
            val = np.sqrt(val)
        return val
    
    def supremizer(self, u: Vector) -> ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]:
        r"""
        Application of the supremizing operator to given vector(s).
        
        The supremizing operator :math:`S(\mu): U \to V` is the unique isomorphic operator given by
        :math:`S(\mu) := R_V^{-1} B(\mu)` where :math:`R_V : V \to V'` is the Riesz map of :math:`V`.
        This function returns for given :math:`u` the result :math:`S(\mu) u` as a function in :math:`\mu`.
        
        If a custom supremizer function was provided during initialization, it is used. Otherwise, the supremizer is constructed from the system matrix and test space inner product. Thereby, if the test space inner product is parameter-independent, the supremizer :math:`S(\mu)` is affine with respect to :math:`\mu` and the result is an `AffineLinear`. Otherweise, a parameter-dependend function with no additional structure is returned.
        
        Args:
            u:
                Trial vector(s) :math:`(N,)` or :math:`(N, k)` to which the supremizer is applied.
        Returns:
            Supremizer applied to :math:`u`, i.e. :math:`S(\mu) u` as a function of :math:`\mu`.
        """
        if self._supremizer_func is not None:
            return self._supremizer_func(u, self)
        else:
            if self.V.is_parametric:
                return lambda mu: self.V.dual(mu) @ (self.B(mu) @ wrap_affinelinear(u)(mu))
            else:
                return self.V.dual(NO_MU) @ (self.B @ u)
        
        
        
class ParametricGalerkinOperator(ParametricOperator[Mu]):
    r"""
    Affine Galerkin Operator :math:`B(\mu) : U \to U'`.
    
    This is a special case of the more general :class:`ParametricOperator`, where the trial and test space coincide, i.e. :math:`U = V` and :math:`m = n`.
    """
    
    _take_square_root_eigenvalues = False
    
    @property
    def V(self) -> InnerProduct[Mu]:
        return self.U
    @V.setter
    def V(self, value: InnerProduct[Mu] | None):
        if value not in (self.U, None):
            raise AttributeError("Cannot set V to something else the U for Galerkin models, as U=V.")
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix,
                 U: InnerProduct[Mu],
                 stability:  Callable[[Mu, ParametricGalerkinOperator[Mu]], float] | float | str = None,
                 continuity: Callable[[Mu, ParametricGalerkinOperator[Mu]], float] | float | str = None,
                 supremizer: Callable[[Vector, ParametricGalerkinOperator[Mu]], ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]] = None):
        r"""
        Args:
            B:
                Affine decomposition of the operator matrix.
            U:
                Inner product on the trial space.
            stability:
                Parameter how the stability constant is computed. Might take the values ``'iterative'`` (default), ``'direct'``, ``'estimate'``, a scalar number ``s`` or a custom callable with signature ``s,x = stability(mu, op)`` or ``s = stability(mu, op)``. Defaults to ``'iterative'``.
                
                If set to ``'iterative'``, the stability constant is computed via the iterative solver `scipy.sparse.linalg.eigsh` (you might tweak its options by changing ``op._eigsh_options_stability``). If set to ``'direct'``, the stability constant is computed via the dense solver `scipy.linalg.eigvals`. If set to ``'estimate'``, the stability  constant is estimated by the lower bound :math:`1/\|A^{-1}M\|_1` using `scipy.sparse.linalg.onenormest`.
                
                Explanation to the values ``s`` and ``x``:
                Consider the generalized eigenvalue problem :math:`A x = \lambda M x`, with :math:`A := 0.5 (B(\mu)^T + B(\mu))` and :math:`M := U(\mu)`, with the operator matrix :math:`B(\mu)` and the inner product matrix :math:`U(\mu)` on the trial space. The stability constant is then given by the smallest eigenvalue. Thus, ``s`` is :math:`\lambda_{\text{min}}` and ``x`` the corresponding eigenvector.
            continuity:
                See the description of the ``stability`` parameter, with the only difference, that the continuity constant is given by the largest eigenvalue instead of the smallest. Thus, if ``'estimate'`` is selected, the continuity constant is estimated by the upper bound :math:`\|M^{-1}A\|_1`.  If ``'iterative'`` was selected, you might tweak the options of the underlying eigenvalue solver by changing ``op._eigsh_options_continuity``.
            supremizer:
                Custom function for the supremizing operator. If ``None``, a default implementation is used. See :meth:`supremizer`. For Galerkin models, the supremizer is normally not needed.
        """
        super().__init__(B, U, U, stability, continuity, supremizer)
    
    def _get_eigenvalue_operators(self, mu: Mu):
        B = self.B(mu)
        A = 0.5 * (B.T + B)
        return A, self.U(mu), None, self.U.dual(mu)