"""
Algorithms for building constant estimatators.
"""

from __future__ import annotations

__all__ = [
    'greedy_constant_estimator',
]

from collections.abc import Callable

import time
from warnings import warn
import numpy as np

from ulmRBM.core import Mu

from ._interface import EfficientConstantEstimator

def _get_max_error(estimator: EfficientConstantEstimator[Mu], mu_train: np.ndarray) -> tuple[float, float, int]:
    err = np.array([estimator.error_bound(mu, rel=True, abs=True) for mu in mu_train])
    rel_err = err[:,0]
    abs_err = err[:,1]
    
    idx = np.argmax(rel_err)
    rel_max = rel_err[idx]
    
    if np.isinf(rel_max):
        inf_mask = np.argwhere(np.isinf(rel_err)).flatten()
        idx = inf_mask[np.argmax(abs_err[inf_mask])]
        abs_max = abs_err[idx]
    else: 
        abs_max = abs_err.max()
    return rel_max, abs_max, idx


def greedy_constant_estimator(
    estimator: EfficientConstantEstimator[Mu],
    mu_train: list[Mu],
    Nmax: int = 100,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    callback: Callable[[EfficientConstantEstimator[Mu], Mu], None] | None = None
) -> tuple[int, np.ndarray, np.ndarray]:
    r"""Greedy algorithm to construct a constant estimator.
    
    Iteratively updates the estimater using the parameter in ``mu_train`` with the largest relative error.
    
    Args:
        estimator: 
            The constant estimator to enrich.
        mu_train: 
            Training set of parameter values.
        Nmax: 
            Maximum number of parameters to add (not total number).
        rtol: 
            Relative error tolerance for stopping criterion.
        atol: 
            Absolute error tolerance for stopping criterion.
        callback: 
            Optional callback called after each enrichment with signature ``(estimator, mu)``, where mu is the parameter value of the last enrichment.
    """
    
    
    print(f"Starting greedy for constant estimator")
    start_time = time.time()
    
    mu_train = np.array(mu_train)
    
    abs_err_decay = []
    rel_err_decay = []
    selected_mu_idx = []
    original_idx = np.arange(len(mu_train))
    
    flag = 1
    start = 0
    
    if estimator.n == 0:
        start = 1
    
        print(f"{time.time() - start_time:6.1f}s: Iteration {0:3d}: initializing & updating estimator (idx={0})...")
        estimator.update(mu_train[0])
        
        if callback is not None:
            callback(estimator, mu_train[0])
        
        original_idx = np.delete(original_idx, 0)
        mu_train = np.delete(mu_train, 0, axis=0)
        selected_mu_idx.append(0)


    for n in range(start, Nmax):
        
        if len(mu_train) == 0:
            flag = 2
            break
    
        rel_err, abs_err, idx = _get_max_error(estimator, mu_train)
        abs_err_decay.append(abs_err)
        rel_err_decay.append(rel_err)
        mu_idx = original_idx[idx]
        
        if abs_err <= atol or rel_err <= rtol:
            flag = 0
            break
        
        print(f"{time.time() - start_time:6.1f}s: Iteration {n:3d}: current error: rel={rel_err:.2e}, abs={abs_err:.2e}, updating estimator (idx={mu_idx})...")
        
        estimator.update(mu_train[idx])
        
        if callback is not None:
            callback(estimator, mu_train[idx])
        
        original_idx = np.delete(original_idx, idx)
        mu_train = np.delete(mu_train, idx, axis=0)
        selected_mu_idx.append(mu_idx)
        
    codes= {0: "tolerance reached", 1: "Nmax reached", 2: "training set exhausted"}
        
    print(f"{time.time() - start_time:6.1f}s: Iteration {len(selected_mu_idx):3d}: stopping due to {codes[flag]}, final error: rel={rel_err:.2e}, abs={abs_err:.2e})\n")

    return flag, np.array(rel_err_decay), np.array(abs_err_decay), np.array(selected_mu_idx, dtype=int)
    
    