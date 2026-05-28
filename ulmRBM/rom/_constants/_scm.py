"""
Successive Constraint Method estimators for stability and continuity constants.
"""

from __future__ import annotations

__all__ = [
    'SCMStability',
    'SCMContinuity',
]

from collections.abc import Callable

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh, LinearOperator, aslinearoperator, onenormest

from ulmRBM.core import Mu, NO_MU
from ulmRBM.fom import FOM, GalerkinFOM
from ulmRBM.products import OperatorInnerProduct

from ._interface import StabilityEstimator, ContinuityEstimator, EfficientConstantEstimator

class _SCMBase(EfficientConstantEstimator[Mu]):
    r"""Base class for SCM-based estimators for stability and continuity constants.
    
    See 
        A. Quarteroni, A. Manzoni, and F. Negri. Reduced Basis Methods for Partial Differential Equations.An Introduction. Vol. 92. UNITEXT. Springer, 2015. doi: 10.1007/978-3-319-15431-2
    """
    
    _eigsh_options: dict = {
        'v0': None,
        'ncv': None,
        'maxiter': None,
        'tol': 1e-4,
        'rng': None
    }
    
    def __init__(self, 
                 fom: FOM[Mu],
                 mus: list[Mu],
                 Me: int = 40,
                 Mp: int = 40,
                 dist: Callable[[Mu, list[Mu]], np.ndarray[float]] = None,
                 eigenvalues: Callable[[int | tuple[int,int], FOM[Mu]], tuple[float, float]] | str = 'iterative'):
        r"""
        Args:
            fom:
                Full-order model for which to estimate the constant.
            mus:
                Parameter set from which to select the ``Mp`` in the online linear optimization problem. Selecting these parameters is done online for each new parameter value. Thus, the SCM online-phase scales with the length of ``mus``. On the other hand, being able to select parameters form ``mus`` that are close to the parameter for which the bounds are evaluated online might enhance the quality of the bounds.
            Me:
                Number of constraints added to the online linear optimization problem based on the exact constants :math:`\sigma(\tilde\mu)`, where the parameters :math:`\tilde\mu` are the ``Me`` nearest neigbors selected from the parameter set that was created using :meth:`update`.
            Mp:
                Number of constraints added to the online linear optimization problem based on approximations :math:`\hat\sigma(\tilde\mu)`, where the parameters :math:`\tilde\mu` are the ``Mp`` nearest neighbors selected from the parameter set ``mus``.
                
                for the ``Mp`` nearest neighbors :math:`\tilde\mu \in \mathcal{P}_p`.
            dist:
                Distance function used to select the ``Me`` and ``Mp`` nearest parameters :math:`\tilde\mu`. The default uses the Euclidean distance, provided that the parameters are convertible to numpy arrays. If a custom function is proved, ``dist(mu, mus)[i]`` should return the distance between ``mu`` and ``mus[i]``.
            eigenvalues:
                The online linear optimization problem containes some box constraints on the unknowns. These bounds are given by the largest and smallest eigenvalue of several generalized eigenvalue problems :math:`Ax = \lambda Ux`, where :math:`U` is the inner product matrix on the trial space. Denoting the affine decomposition of the full-order system matrix ba :math:`B(\mu) = \sum_{q=1}^Q \theta_q(\mu) B_q`. For Galerkin problems it holds :math:`A_q := 0.5 (B_q + B_q^T)`, :math:`q=1,\ldots,Q`. For Petrov-Galerkin problems, it holds :math:`A_{qq} := B_q^T V^{-1} B_q` for :math:`q=1,\ldots,Q` and :math:`A_{pq} := B_p^T V^{-1} B_q + B_q^T V^{-1} B_p` for :math:`p,q=1,\ldots,Q` with :math:`p<q`, where :math:`V^{-1}` is the inverse of the inner product matrix on the test space (or the inner product of the dual space).                
                The ``eigenvalues`` parameter defines how these eigenvalues are computed. If ``eigenvalues`` is set to ``'direct'``, the eigenvalues are computed via the dense solver `scipy.linalg.eigvals`. If ``eigenvalues`` is set to ``'iterative'``, the eigenvalues are computed via iterative solver `scipy.sparse.linalg.eigsh` (you might tweak its options by changing ``fom._eigsh_options``). If ``eigenvalues`` is set to ``'estimate'``, the eigenvalues are estimated by :math:`\|U^{-1}A\|_1` and :math:`-\|U^{-1}A\|_1` using `scipy.sparse.linalg.onenormest`, exept for :math:`A_{qq}`, where the smallest eigenvalue is estimated by zero. Alternatively, a custom function can be provided that returns bounds for the eigenvalues and is called with ``eigenvalues((q,), self.fom)`` for Galerkin problems and ``eigenvalues((p, q), self.fom)`` for Petrov-Galerkin problems.
        """
        super().__init__(fom)
        
        if self.fom.U.is_parametric or self.fom.V.is_parametric:
            raise ValueError("SCM does not support parametric inner products.")
        
        if isinstance(eigenvalues, str) and eigenvalues not in ['direct', 'iterative', 'estimate']:
            raise ValueError("Invalid value for 'eigenvalues' parameter. Expected 'direct', 'iterative', 'estimate' or a callable.")
        
        if dist is None:
            def dist(mu, mus):
                mu = np.asarray(mu)
                mus = np.asarray(mus)
                if mu.ndim == 0:
                    mu = mu.reshape(1)
                    mus = mus.reshape(mus.shape[0], 1)
                return np.linalg.norm(mus - mu, axis=tuple(range(1, mus.ndim)))
        
        self._mus = np.array(mus)
        self._Me = Me
        self._Mp = Mp
        self._dist = dist
        self._eigenvalue_solver = eigenvalues
        self._eigsh_options = self._eigsh_options.copy()
        
        
    def _get_exact_constant(self, mu: Mu) -> tuple[float, np.ndarray]:
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    def _lower_bound(self, mu: Mu) -> float:
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    def _upper_bound(self, mu: Mu) -> float:
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    def lower_bound(self, mu: Mu) -> float:
        if not self.n > 0:
            raise RuntimeError("No precomputed constants available. Call 'update' method at least once.")
        
        val = abs(self._lower_bound(mu))
        if not isinstance(self.fom, GalerkinFOM): 
            val = np.sqrt(val)
        return val
    
    def upper_bound(self, mu: Mu) -> float:
        if not self.n > 0:
            raise RuntimeError("No precomputed constants available. Call 'update' method at least once.")
        
        val = abs(self._upper_bound(mu))
        if not isinstance(self.fom, GalerkinFOM): 
            val = np.sqrt(val)
        return val
    
    
    def _eval_theta(self, mu: Mu) -> np.ndarray:
        theta = np.array([theta(mu) for theta in self.fom.B.theta])
        if not isinstance(self.fom, GalerkinFOM):
            theta = np.tril(np.outer(theta, theta))
            theta = theta[np.tril_indices(theta.shape[0])]
        return theta
    
    
    def _nearest_neighbors(self, mu: Mu, mus: list[Mu], M: int) -> list[Mu]:
        if M == 0: return []
        if M >= len(mus): return np.arange(len(mus))
        return np.argsort(self._dist(mu, mus))[:M]
        
        
    def _initialize(self):
        self._sigmas_LB = np.zeros(len(self._mus))
        self._thetas_LB = np.vstack([self._eval_theta(mu) for mu in self._mus])
        self._C = self._mus[:0] # get empty array with correct shape
        self._sigmas = np.zeros(0,)
        self._thetas = np.zeros((0, self._thetas_LB.shape[1]))
        self._Y_UB = np.zeros((0, self._thetas_LB.shape[1]))
        self._box = self._get_box()
        
        
    def _update(self, mu: Mu):
        idx = np.argwhere(self._dist(mu, self._mus) < 1e-14)
        self._mus       = np.delete(self._mus, idx, axis=0)
        self._sigmas_LB = np.delete(self._sigmas_LB, idx, axis=0)
        self._thetas_LB = np.delete(self._thetas_LB, idx, axis=0)
        
        theta = self._eval_theta(mu)
        sigma, w = self._get_exact_constant(mu)
        w = w.reshape(-1,1)
        
        if isinstance(self.fom, GalerkinFOM):
            y = np.array((w.T @ self.fom.B @ w).data) / self.fom.U.norm(NO_MU, w)**2
        else:
            sigma = sigma**2
            y = self.fom.V.dual.inner(NO_MU, np.hstack((self.fom.B @ w).data)) / self.fom.U.norm(NO_MU, w)**2
            y += np.triu(y, k=1).T
            y = y[np.tril_indices(y.shape[0])]
        
        self._C      = np.concatenate((self._C, [mu]), axis=0)
        self._sigmas = np.append(self._sigmas, sigma)
        self._thetas = np.vstack([self._thetas, theta.reshape(1,-1)])
        self._Y_UB   = np.vstack([self._Y_UB, y.reshape(1,-1)])
        
        if isinstance(self.fom, GalerkinFOM):
            self._sigmas_LB = np.array([self(mu_) for mu_ in self._mus])
        else:
            self._sigmas_LB = np.array([self(mu_)**2 for mu_ in self._mus])
            
        
    def _get_box(self):
        if isinstance(self.fom, GalerkinFOM):
            lower = np.zeros((len(self.fom.B),))
            upper = np.zeros((len(self.fom.B),))
            for p in range(len(self.fom.B)):
                A = 0.5 * (self.fom.B.data[p].T + self.fom.B.data[p])
                lower[p], upper[p] = self._eigenvalues(A, (p,))
        else:
            lower = np.zeros((len(self.fom.B), len(self.fom.B)))
            upper = np.zeros((len(self.fom.B), len(self.fom.B)))
            for p in range(len(self.fom.B)):
                A = OperatorInnerProduct(self.fom.B.data[p], self.fom.V.dual)(NO_MU)
                lower[p,p], upper[p,p] = self._eigenvalues(A, (p, p))
                for q in range(p):
                    def matmul(v):
                        val  = self.fom.B.data[p].T @ self.fom.V.dual.riesz(NO_MU, self.fom.B.data[q] @ v)
                        val += self.fom.B.data[q].T @ self.fom.V.dual.riesz(NO_MU, self.fom.B.data[p] @ v)
                        return val
                    A = LinearOperator(self.fom.U.shape, matvec=matmul, matmat=matmul, rmatvec=matmul, rmatmat=matmul, dtype=float)
                    lower[p,q], upper[p,q] = self._eigenvalues(A, (p, q))
            lower = lower[np.tril_indices(lower.shape[0])]
            upper = upper[np.tril_indices(upper.shape[0])]
        return np.hstack([lower.reshape(-1,1), upper.reshape(-1,1)])
        
        
    def _eigenvalues(self, A, pq):
        r"""Solves the eigenvalue problem :math:`Av = \lambda Uv` and returns the smallest and largest eigenvalue.
        
        Args:
            A:
                Symmetric matrix or linear operator defining the eigenvalue problem.
            pq:
                Tuple ``(q,)`` (Galerkin) and ``(p,q)`` (Petrov-Galerkin). Used to check if ``p==q``, as only in this case it is known, that the spectrum is either positive or negative (including zero), which makes solving easier.
        """
        if self._eigenvalue_solver == 'direct':
            A = A @ np.eye(A.shape[0])
            U = self.fom.U(NO_MU) @ np.eye(A.shape[0])
            eigs = eigvalsh(A, U, overwrite_a=True, overwrite_b=True)
            values = [eigs.real.min(), eigs.real.max()]
        
        elif self._eigenvalue_solver == 'estimate':
            A = aslinearoperator(A)
            Minv = aslinearoperator(self.fom.U.dual(NO_MU))
            val_max_esti = onenormest(Minv@A)
            if len(pq) == 1 or pq[0] != pq[1]:
                values = [-val_max_esti, val_max_esti]
            else:
                values = [0.0, val_max_esti]
                
        elif self._eigenvalue_solver == 'iterative':
            eigsh_opts = self._eigsh_options.copy()
            eigsh_opts['A'] = A
            eigsh_opts['M'] = self.fom.U(NO_MU)
            eigsh_opts['Minv'] = self.fom.U.dual(NO_MU)
            eigsh_opts['return_eigenvectors'] = False
            
            if len(pq) == 1 or pq[0] != pq[1]:
                eigsh_opts['k'] = 2
                eigsh_opts['which'] = 'BE'
                values = eigsh(**eigsh_opts)
                
            else:
                values = [None, None]
                eigsh_opts['k'] = 1
                eigsh_opts['which'] = 'LM'
                values[0] = eigsh(**eigsh_opts)[0]
                
                del eigsh_opts['Minv']
                eigsh_opts['sigma'] = 0.0
                try:
                    values[1] = eigsh(**eigsh_opts)[0]
                except ValueError as e:
                    if "Error in inverting M: function gmres_loose did not converge" in str(e):
                        values[1] = 0.0
                    else:
                        raise e
                    
        else:
            values = self._eigenvalue_solver(pq, self.fom)
                
        return np.sort(values)
    
    
class SCMStability(_SCMBase[Mu], StabilityEstimator[Mu]):
    r"""
    SCM estimator the stability constant.
    
    The lower bound is calculated via a small linear optimization problem.
    
    See 
        A. Quarteroni, A. Manzoni, and F. Negri. Reduced Basis Methods for Partial Differential Equations.An Introduction. Vol. 92. UNITEXT. Springer, 2015. doi: 10.1007/978-3-319-15431-2
    """
    
    
    
    def _get_exact_constant(self, mu: Mu) -> tuple[float, np.ndarray]:
        return self.fom.stability(mu, eigenvector=True)
    
    def _lower_bound(self, mu: Mu) -> float:
        theta = self._eval_theta(mu)
        e = self._nearest_neighbors(mu, self._C,   self._Me)
        p = self._nearest_neighbors(mu, self._mus, self._Mp)
        
        A = np.concatenate((self._thetas[e], self._thetas_LB[p], theta.reshape(1,-1)))
        b = np.concatenate((self._sigmas[e], self._sigmas_LB[p], [0]))
        
        return linprog(theta, -A, -b, bounds=self._box).fun
    
    def _upper_bound(self, mu: Mu) -> float:
        theta = self._eval_theta(mu)
        return np.min(self._Y_UB @ theta)
    
    
class SCMContinuity(_SCMBase[Mu], ContinuityEstimator[Mu]):
    r"""
    SCM estimator the continuity constant.
    
    The upper bound is calculated via a small linear optimization problem.
    
    See 
        A. Quarteroni, A. Manzoni, and F. Negri. Reduced Basis Methods for Partial Differential Equations.An Introduction. Vol. 92. UNITEXT. Springer, 2015. doi: 10.1007/978-3-319-15431-2
    """
    
    def _get_exact_constant(self, mu: Mu) -> tuple[float, np.ndarray]:
        return self.fom.continuity(mu, eigenvector=True)
    
    def _lower_bound(self, mu: Mu) -> float:
        theta = self._eval_theta(mu)
        return np.max(self._Y_UB @ theta)
    
    def _upper_bound(self, mu: Mu) -> float:
        theta = self._eval_theta(mu)
        e = self._nearest_neighbors(mu, self._C,   self._Me)
        if self._sigmas_LB.max() != 0.0:
            p = self._nearest_neighbors(mu, self._mus, self._Mp)
            
            A = np.concatenate((self._thetas[e], self._thetas_LB[p]))
            b = np.concatenate((self._sigmas[e], self._sigmas_LB[p]))
        else:
            A = self._thetas[e]
            b = self._sigmas[e]
        
        return linprog(-theta, A, b, bounds=self._box).fun