"""
Full-Order Model (FOM) classes.

Classes
-------
.. autosummary::
   :toctree: generated/
   
    FOM
    GalerkinFOM
"""

from __future__ import annotations


__all__ = [
    'FOM',
    'GalerkinFOM',
]

from collections.abc import Callable
from typing import Generic

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh, LinearOperator, aslinearoperator, onenormest

from ulmRBM.core import (
    NO_MU, Mu, Matrix, ParametricLinear, Vector, wrap_scalar,
)
from ulmRBM.solver import IterativeSolver, Solver, wrap_solver
from ulmRBM.affine import AffineLinear, wrap_affinelinear
from ulmRBM.products import InnerProduct, OperatorInnerProduct


class FOM(Generic[Mu]):
    r"""
    Full-order Petrov-Galerkin model given by a parametric linear system of equations.
    
    .. math::
        B(\mu) u(\mu) = f(\mu)
        
    with an optional output of interest functional
    
    .. math::
        s(\mu) = l(\mu) u(\mu),
    
    where :math:`B(\mu) \in \mathbb{R}^{m \times n}` is the system matrix, :math:`f(\mu) \in \mathbb{R}^{m}` is the right-hand side and :math:`l(\mu) \in \mathbb{R}^{p \times n}` is the output of interest functional, all with affine parameter dependence, and :math:`u(\mu) \in \mathbb{R}^n` is the unknown solution vector and :math:`s(\mu)\in \mathbb{R}^p` the optional output of interest. 
    
    Typically, :math:`B(\mu)` is a discretization of a parametric operator :math:`B_\mu : U \to V'`, :math:`f(\mu)` a discretization of a parametric functional :math:`f_\mu \in V'`, and thus :math:`u(\mu)` is a discrete approximation of the solution :math:`u_\mu \in U` of the parametric operator equation :math:`B_\mu u_\mu = f_\mu` in :math:`V'`. Thereby, :math:`U` and :math:`V` denote the trial and test spaces, equipped with (possibly parameter dependent) inner products :math:`(\cdot, \cdot)_U` and :math:`(\cdot, \cdot)_V`, respectively.
    """
    
    dim: tuple[int, int, int | None]
    r"""Dimensions :math:`(m,n,p)` of the system matrix :math:`B(\mu)` and output :math:`s(\mu)`, where :math:`m` is the discrete test space dimension, :math:`n` is the discrete trial space dimension, and :math:`p` is the output dimension. If no output of interest is given, :math:`p` is ``None``."""
    
    B: AffineLinear[Mu, Matrix]
    r"""Affine decomposition of the system matrix :math:`B(\mu) = \sum_{q=1}^Q \theta_q^B(\mu) B_q`."""
    
    f: AffineLinear[Mu, Vector]
    r"""Affine decomposition of the right-hand side :math:`f(\mu) = \sum_{q=1}^{Q_f} \theta_q^f(\mu) f_q`."""
    
    l: AffineLinear[Mu, Matrix] | None
    r"""Affine decomposition of the output(s) of interest functional, if any :math:`l(\mu) = \sum_{q=1}^{Q_l} \theta_q^l(\mu) l_q`."""
        
    U: InnerProduct[Mu]
    r"""Inner product :math:`(\cdot, \cdot)_U` on the trial space :math:`U`."""
    
    V: InnerProduct[Mu]
    r"""Inner product :math:`(\cdot, \cdot)_V` on the test space :math:`V`."""
    
    _solver: Solver
    r"""Solver for the linear system :math:`B(\mu) u = f(\mu)`."""
    
    _stability_solver: Callable[[Mu, FOM[Mu]],float] | str
    """Optional explicit function to compute the stability constant, bypassing the default eigenvalue-based computation."""

    _continuity_solver: Callable[[Mu, FOM[Mu]],float] | str
    """Optional explicit function to compute the continuity constant, bypassing the default eigenvalue-based computation."""
    
    _take_square_root_eigenvalues: bool = True
    """needed for the Petrov-Galerkin and Galerkin cases"""
    
    _supremizer_func: Callable[[Vector, FOM[Mu]], AffineLinear[Mu, Vector]] | None = None
    r"""Internal storage if a custom supremizer function is provided."""
    
    _eigsh_options_stability: dict = {
        'v0': None,
        'ncv': None,
        'maxiter': None,
        'tol': 1e-10,
        'rng': None
    }
    r"""Options for the eigenvalue solver used in the default sparse stability constant computations. If neccessary, these can be updated by the user after initialization, e.g. ``fom._eigsh_options['tol'] = 1e-8``."""
    
    _eigsh_options_continuity: dict = {
        'v0': None,
        'ncv': None,
        'maxiter': None,
        'tol': 1e-10,
        'rng': None
    }
    r"""Options for the eigenvalue solver used in the default sparse continuity constant computations. If neccessary, these can be updated by the user after initialization, e.g. ``fom._eigsh_options['tol'] = 1e-8``."""
    
    @property
    def dim(self):
        p = self.l.shape[0] if self.l is not None else None
        return (*self.B.shape,p)
    
    @property
    def _solver(self) -> Solver:
        return self._solver
    @_solver.setter
    def _solver(self, solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector]):
        self._solver = wrap_solver(solver)
    
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix,
                 f: AffineLinear[Mu, Vector] | Vector,
                 U: InnerProduct[Mu],
                 V: InnerProduct[Mu],
                 l: AffineLinear[Mu, Matrix] | Matrix | None = None,
                 stability: Callable[[Mu, FOM[Mu]], float] | float | str = 'iterative',
                 continuity: Callable[[Mu, FOM[Mu]], float] | float | str = 'iterative',
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = IterativeSolver(),
                 supremizer: Callable[[Vector, FOM[Mu]], ParametricLinear[Mu, Vector]] = None):
        r"""
        Args:
            B:
                Affine decomposition of the system matrix.
            f:
                Affine decomposition of the right-hand side.
            U:
                Inner product on the trial space.
            V:
                Inner product on the test space.
            l:
                Optional affine decomposition of the output(s) of interest functional.
            stability:
                Parameter how the stability constant is computed. Might take the values ``'iterative'`` (default), ``'direct'``, ``'estimate'``, a scalar number ``s`` or a custom callable with signature ``s,x = stability(mu, fom)`` or ``s = stability(mu, fom)``.
                
                If set to ``'iterative'``, the stability constant is computed via the iterative solver `scipy.sparse.linalg.eigsh` (you might tweak its options by changing ``fom._eigsh_options_stability``). If set to ``'direct'``, the stability constant is computed via the dense solver `scipy.linalg.eigvals`. If set to ``'estimate'``, the stability  constant is estimated by the lower bound :math:`\sqrt{1/\|A^{-1}M\|_1}` using `scipy.sparse.linalg.onenormest`.
                
                Explanation to the values ``s`` and ``x``:
                Consider the generalized eigenvalue problem :math:`A x = \lambda M x`, with :math:`A := B(\mu)^T V(\mu)^{-1} B(\mu)` and :math:`M := U(\mu)`, with the system matrix :math:`B(\mu)` and the inner product matrices :math:`U(\mu)` and :math:`V(\mu)` on trial and test space respectively. The stability constant is then given by squareroot of the smallest eigenvalue. Thus, ``s`` is :math:`\sqrt{\lambda_{\text{min}}}` and ``x`` the corresponding eigenvector.
            continuity:
                See the description of the ``stability`` parameter, with the only difference, that the continuity constant is given by the squareroot of the largest eigenvalue instead of the smallest. Thus, if ``'estimate'`` is selected, the continuity constant is estimated by the upper bound :math:`\sqrt{\|M^{-1}A\|_1}`.  If ``'iterative'`` was selected, you might tweak the options of the underlying eigenvalue solver by changing ``fom._eigsh_options_continuity``.
            solver:
                Solver for the linear system. Defaults to a iterative solver.
            supremizer:
                Custom function for the supremizing operator. If ``None``, a default implementation is used. See :meth:`supremizer`.
        """
        
        if B.shape[0] != f.shape[0]:
            raise ValueError("B and f must have compatible dimensions.")
        if B.shape[1] != U.shape[0]:
            raise ValueError("B and U must have compatible dimensions.")
        if B.shape[0] != V.shape[0]:
                raise ValueError("B and V must have compatible dimensions.")
        if l is not None and l.shape[1] != U.shape[0]:
            raise ValueError("l and U must have compatible dimensions.")
        
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
        self.f = wrap_affinelinear(f).compress()
        self.l = wrap_affinelinear(l).compress() if l is not None else None
        self.U = U
        self.V = V
        self._solver = solver
        self._stability_solver = stability
        self._continuity_solver = continuity
        self._supremizer_func = supremizer
        self._eigsh_options_stability  = self._eigsh_options_stability.copy()
        self._eigsh_options_continuity = self._eigsh_options_continuity.copy()
        
    def __repr__(self):
        shape = f"({self.dim[0]}, {self.dim[1]}"
        if self.dim[2] is not None: shape += f", {self.dim[2]}"
        shape += ")"
        return f"<{self.__class__.__name__} of dimension {shape}>"
    
    
    def stability(self, mu: Mu, eigenvector: bool = False) -> float:
        r"""
        Compute the stability constant at parameter value :math:`\mu`. 
        
        For Galerkin models, this is the coercivity constant:
        
        .. math::
            \beta(\mu) = \inf_{u \in U} \frac{| \langle B(\mu) u, u \rangle_{U'\times U} |}{\|u\|_U^2}
            
        For Petrov-Galerkin models, this is the inf-sup constant:
        
        .. math::
            \beta(\mu) = \inf_{u \in U} \sup_{v\in V} \frac{| \langle B(\mu) u, v \rangle_{V'\times V} |}{\|u\|_U \|v\|_V}
        
        Args:
            mu:
                Parameter value at which to compute the stability constant.
                
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
            A = OperatorInnerProduct(self.B, self.V.dual, self._solver)
            return A(mu), M, A.dual(mu), Minv
        else:
            A = self.V.dual.restrict(self.B(mu))
            return A(mu), M, None, Minv
        
    def _stability_dense(self, A, M, Ainv, Minv):
        if not isinstance(A, np.ndarray) or issparse(A):
            A = A @ np.eye(A.shape[0])
        if not isinstance(M, np.ndarray) or issparse(M):
            M = M @ np.eye(M.shape[0])
        eigs, vecs = eigh(A, M)
        val, vec = np.abs(eigs[0]), vecs[:,0]
        if self._take_square_root_eigenvalues:
            val = np.sqrt(val)
        return val, vec
    
    def _continuity_dense(self, A, M, Ainv, Minv):
        if not isinstance(A, np.ndarray) or issparse(A):
            A = A @ np.eye(A.shape[0])
        if not isinstance(M, np.ndarray) or issparse(M):
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
        return val, vec

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
        return val, vec
    
    
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
    
        
    def solve(self, mu: Mu, u0=None) -> Vector:
        r"""
        Solve the parametric system :math:`B(\mu) u = f(\mu)` for the given parameter value.
        
        Args:
            mu:
                Parameter value at which to solve the system.
            u0:
                Optional initial guess for iterative solvers.
        
        Returns:
            State vector :math:`u(\mu) \in \mathbb{R}^n`.
        """
        return self._solver(self.B(mu), self.f(mu), u0)
    
    def output(self, mu: Mu, u: Vector | None = None) -> Vector:
        r"""
        Compute the output of interest :math:`s(\mu) = l(\mu) u(\mu)` at the given parameter value.
        
        Args:
            mu:
                Parameter value at which to compute the output.
            u:
                Optional state vector to use instead of solving for :math:`u(\mu)`. If ``None``, the state is computed via :meth:`solve`.
        
        Returns:
            Output of interest :math:`s(\mu) \in \mathbb{R}^p`.
        """
        if self.l is None:
            raise ValueError("No output functional defined for this model.")
        if u is None: u = self.solve(mu)
        return self.l(mu) @ u
    
    
    def supremizer(self, u: Vector) -> ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]:
        r"""
        Application of the supremizing operator to given trial vector(s).
        
        The supremizing operator :math:`S(\mu): U \to V` is the unique isomorphic operator given by
        :math:`S(\mu) := R_V^{-1} B(\mu)` where :math:`R_V : V \to V'` is the Riesz map of the test space.
        This function returns for given :math:`u` the result :math:`S(\mu) u` as a function in :math:`\mu`.
        
        If a custom supremizer function was provided during initialization, it is used. Otherwise, the supremizer is constructed from the system matrix and test space inner product. Thereby, if the test space inner product is parameter-independent, the supremizer :math:`S(\mu)` is affine with respect to :math:`\mu` and the result is an `AffineLinear`. Otherweise, a parameter-dependend function with no additional structure is resturned.
        
        Args:
            u:
                Trial vector(s) :math:`(N,)` or :math:`(N, k)` to which the supremizer is applied.
        Returns:
            Supremizer applied to :math:`u`, i.e. :math:`S(\mu) u` as a function of :math:`\mu`.
        """
        if self._supremizer_func is not None:
            return self._supremizer_func(u, self)
        else:
            return self._supremizer(u)
        
    
    def _supremizer(self, u: AffineLinear[Mu, Vector] | Vector) -> ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]:
        if self.V.is_parametric:
            return lambda mu: self.V.dual(mu) @ (self.B(mu) @ wrap_affinelinear(u)(mu))
        else:
            return self.V.dual(NO_MU) @ (self.B @ u)
        

class GalerkinFOM(FOM[Mu]):
    r"""
    Full-order Galerkin problem.
    
    Galerkin problems are a special case of Petrov-Galerkin problems, see :class:`FOM` for documentation, where the trial and test spaces coincide, i.e. it holds :math:`U = V` and :math:`m = n`.
    """
    
    _take_square_root_eigenvalues = False
    
    @property
    def V(self) -> InnerProduct[Mu]:
        return self.U
    @V.setter
    def V(self, value: InnerProduct[Mu] | None):
        if value not in (self.U, None):
            raise AttributeError("Cannot set V for Galerkin models, as U and V are identical.")
    
    def __init__(self,
                 B: AffineLinear[Mu, Matrix] | Matrix,
                 f: AffineLinear[Mu, Vector] | Vector, 
                 U: InnerProduct[Mu],
                 l: AffineLinear[Mu, Matrix] | Matrix | None = None,
                 stability: Callable[[Mu],float] | float | str = 'iterative',
                 continuity: Callable[[Mu],float] | float | str = 'iterative',
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] = IterativeSolver()):
        r"""
        Args:
            B:
                Affine decomposition of the system matrix.
            f:
                Affine decomposition of the right-hand side.
            U:
                Inner product on the trial space.
            l:
                Optional affine decomposition of the output(s) of interest functional.
            stability:
                Parameter how the stability constant is computed. Might take the values ``'iterative'`` (default), ``'direct'``, ``'estimate'``, a scalar number ``s`` or a custom callable with signature ``s,x = stability(mu, fom)`` or ``s = stability(mu, fom)``.
                
                If set to ``'iterative'``, the stability constant is computed via the iterative solver `scipy.sparse.linalg.eigsh` (you might tweak its options by changing ``fom._eigsh_options_stability``). If set to ``'direct'``, the stability constant is computed via the dense solver `scipy.linalg.eigvals`. If set to ``'estimate'``, the stability  constant is estimated by the lower bound :math:`1/\|A^{-1}M\|_1` using `scipy.sparse.linalg.onenormest`.
                
                Explanation to the values ``s`` and ``x``:
                Consider the generalized eigenvalue problem :math:`A x = \lambda M x`, with :math:`A := 0.5 (B(\mu)^T + B(\mu))` and :math:`M := U(\mu)`, with the system matrix :math:`B(\mu)` and the inner product matrices :math:`U(\mu)` on the trial space. The stability constant is then given by the smallest eigenvalue. Thus, ``s`` is :math:`\lambda_{\text{min}}` and ``x`` the corresponding eigenvector.
            continuity:
                See the description of the ``stability`` parameter, with the only difference, that the continuity constant is given by the largest eigenvalue instead of the smallest. Thus, if ``'estimate'`` is selected, the continuity constant is estimated by the upper bound :math:`\|M^{-1}A\|_1`.  If ``'iterative'`` was selected, you might tweak the options of the underlying eigenvalue solver by changing ``fom._eigsh_options_continuity``.
            solver:
                Solver for the linear system. Defaults to a iterative solver.
            supremizer:
                Custom function for the supremizing operator. If ``None``, a default implementation is used. See :meth:`supremizer`.
        """
        super().__init__(B, f, U, U, l, stability, continuity, solver)
    
    def _get_eigenvalue_operators(self, mu: Mu):
        B = self.B(mu)
        A = 0.5 * (B.T + B)
        return A, self.U(mu), None, self.U.dual(mu)