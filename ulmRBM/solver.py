"""
Linear system solvers for the ulmRBM package.

Classes
-------
.. autosummary::
   :toctree: generated/
   
    Solver
    DirectSolver
    IterativeSolver
    
Wrapper Utilities
-----------------
.. autosummary::
   :toctree: generated/
   
    wrap_solver
"""

from __future__ import annotations

__all__ = [
    'Solver',
    'DirectSolver',
    'IterativeSolver',
    'wrap_solver',
]

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator, cg, gmres, lsmr

from abc import ABC, abstractmethod
from collections.abc import Callable

from ulmRBM.core import Matrix, Vector


class Solver(ABC):
    """Abstract base class for linear system solvers."""
    
    @abstractmethod
    def __call__(self, A: Matrix, b: Vector, x0: Vector = None) -> Vector:
        r"""
        Solve the linear system A x = b.
        
        Args:
            A : (m,n) :obj:`Matrix`, m>=n
            b : (m,) or (m,k) :obj:`Vectors`
            x0 : (n,) or (n,k) :obj:`Vectors`, optional
                Initial guess for iterative solvers. Ignored by direct solvers.
        
        Returns:
            (n,) or (n,k) :obj:`Vectors` :
                Solution or least-squares approximation of ``Ax = b``.
        """
        pass


class _WrapCallableAsSolver(Solver):
    """Wrapper to use a callable as a Solver."""
    
    def __init__(self, func: Callable[[Matrix, Vector, Vector|None], Vector], needs1d: bool = False):
        self._func = func
        self._needs1d = needs1d

    def __call__(self, A: Matrix, b: Vector, x0: Vector = None) -> Vector:
        if A.shape[0] < A.shape[1]:
            raise ValueError("Underdetermined systems are not supported.")
        
        # Preserve original shape (x0, priority over b)
        if x0 is not None:
            is_1d = x0.ndim == 1
            if x0.ndim == 1: x0 = x0.reshape(-1, 1)
        else:
            is_1d = b.ndim == 1
            
        if b.ndim == 1: b = b.reshape(-1, 1)
        
        if not self._needs1d:
            try:
                x = self._func(A, b, x0)
            except Exception:
                self._needs1d = True
                
        if self._needs1d:        
            x = np.empty((A.shape[1], b.shape[1]))
            for i in range(b.shape[1]):
                x0_i = x0[:, i].reshape(-1) if x0 is not None else None
                x[:, i] = self._func(A, b[:, i].reshape(-1), x0_i)
                
        if x.ndim == 1: x = x.reshape(-1, 1)
        
        return x.reshape(-1) if is_1d else x


class DirectSolver(Solver):
    """
    Direct solver using numpy/scipy linear algebra routines.
    
    For square systems, uses :func:`numpy.linalg.solve` (dense) or 
    :func:`scipy.sparse.linalg.spsolve` (sparse).
    
    For overdetermined systems, uses :func:`numpy.linalg.lstsq` for dense and sparse, as there is no direct sparse least-squares solver.
    """
    
    def __call__(self, A: np.ndarray | sp.sparse.sparray, b: Vector, x0: Vector = None) -> Vector:
        if sp.sparse.issparse(A):
            A = A.tocsc()
            if A.shape[0] == A.shape[1]:
                if sp.sparse.issparse(b): b = b.tocsc()
                solver = lambda A, b, x0: sp.sparse.linalg.spsolve(A, b)
            else:
                from warnings import warn
                warn("no direct least-squares solver for sparse matrices; converting to dense", sp.sparse.SparseEfficiencyWarning)
                A = A.toarray()
                solver = lambda A, b, x0: np.linalg.lstsq(A, b, rcond=None)[0]
        elif isinstance(A, np.ndarray):
            if A.shape[0] == A.shape[1]:
                solver = lambda A, b, x0: np.linalg.solve(A, b)
            else:
                solver = lambda A, b, x0: np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            raise TypeError("Matrix A must be a NumPy array or SciPy sparse array.")
        
        return _WrapCallableAsSolver(solver)(A, b, x0)  # mostly wrap it to handle multiple rhs correctly


class IterativeSolver(Solver):
    """
    Iterative solver using scipy's iterative methods.
    
    Depending on the matrix type and ``spd`` flag, uses:
    
    - :func:`scipy.sparse.linalg.cg` for SPD square systems
    - :func:`scipy.sparse.linalg.gmres` for non-SPD square systems
    - :func:`scipy.sparse.linalg.lsmr` for rectangular (overdetermined) systems
    """
    
    def __init__(self, spd: bool = False, rtol: float = None, atol: float = None, 
                 btol: float = None, maxiter: int = None, damp: float = None, conlim: float = None, 
                 restart: int = None, M: LinearOperator = None):
        r"""
        Parameters
        ----------
        spd :
            Whether to use a solver for symmetric positive definite matrices.
            If ``True``, :func:`cg` is used, otherwise :func:`gmres`. 
            If the system is not square, :func:`lsmr` is used regardless of this flag.
        rtol :
            Relative tolerance for CG and GMRES.
        atol :
            Absolute tolerance for CG, GMRES, and LSMR.
        btol :
            Second tolerance for LSMR.
        maxiter :
            Maximum number of iterations.
        damp :
            Damping parameter for LSMR.
        conlim :
            Condition number limit for LSMR.
        restart :
            Number of iterations between restarts for GMRES.
        M :
            Preconditioner for CG and GMRES.
        """
        self.spd = spd
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.M = M
        self.btol = btol
        self.damp = damp
        self.conlim = conlim
        self.restart = restart
        
    def __call__(self, A: Matrix, b: Vector, x0: Vector = None) -> Vector:
        
        kwargs = {}
        if self.maxiter is not None:kwargs['maxiter'] = self.maxiter
        if self.atol is not None:   kwargs['atol']    = self.atol
        
        if self.spd:
            if A.shape[0] != A.shape[1]:
                raise ValueError("SPD solver can only be used for square matrices.")
            if self.rtol is not None:   kwargs['rtol']    = self.rtol
            if self.M is not None:      kwargs['M']       = self.M
            solver = lambda A, b, x0: cg(A, b, x0=x0, **kwargs)[0]
        else:
            if A.shape[0] != A.shape[1]:
                if self.btol is not None:   kwargs['btol']    = self.btol
                if self.damp is not None:   kwargs['damp']    = self.damp
                if self.conlim is not None: kwargs['conlim']  = self.conlim
                solver = lambda A, b, x0: lsmr(A, b, x0=x0, **kwargs)[0]
            else:
                if self.rtol is not None:   kwargs['rtol']    = self.rtol
                if self.restart is not None:kwargs['restart'] = self.restart
                if self.M is not None:      kwargs['M']       = self.M
                solver = lambda A, b, x0: gmres(A, b, x0=x0, **kwargs)[0]
        
        solver = _WrapCallableAsSolver(solver, needs1d=True)  # mostly wrap it to handle multiple rhs correctly
        return solver(A, b, x0)


def wrap_solver(solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector]) -> Solver:
    """
    Wrap a callable as a Solver if necessary.
    
    Args:
        solver :
            Either a :class:`Solver` instance or a callable with signature
            ``(A, b, x0=None) -> x``.
    
    Returns:
        A :class:`Solver` instance.
    """
    if not isinstance(solver, Solver):
        solver = _WrapCallableAsSolver(solver)
    return solver
