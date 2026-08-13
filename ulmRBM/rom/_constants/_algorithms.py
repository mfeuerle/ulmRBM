"""
Algorithms for building constant estimatators.
"""

from __future__ import annotations

__all__ = [
    'greedy_constant_estimator',
]

from collections.abc import Callable

import time
import numpy as np

from ulmRBM.core import Mu

from ._interface import EfficientConstantEstimator, StabilityEstimator

def _get_max_error(estimator: EfficientConstantEstimator[Mu], mu_train: np.ndarray) -> tuple[float, float, int]:
    err = np.array([estimator.error_bound(mu, rel=True, abs=True) for mu in mu_train])
    rel_err = err[:,0]
    abs_err = err[:,1]
    
    idx = np.argmax(rel_err)
    rel_max = rel_err[idx]
    
    if np.isinf(rel_max):
        inf_mask = np.argwhere(np.isinf(rel_err)).flatten()
        idx = inf_mask[np.argmax(abs_err[inf_mask])]
    return rel_max, idx


def greedy_constant_estimator(
    estimator: EfficientConstantEstimator[Mu],
    mu_train: list[Mu],
    N: int = 100,
    tol: float = 1e-1,
    callback: Callable[[EfficientConstantEstimator[Mu], Mu], None] | None = None
) -> tuple[int, np.ndarray, np.ndarray]:
    r"""Greedy algorithm to construct a constant estimator.
    
    Iteratively updates the estimator using the parameter in ``mu_train`` with the largest relative error.
    
    Args:
        estimator: 
            The constant estimator to enrich.
        mu_train: 
            Training set of parameter values.
        N: 
            Maximum number of iterations / updates to the estimator (not maximum total number of updates, which is  ``estimator.n + N``).
        tol: 
            Error tolerance for stopping criterion.
        callback: 
            Optional callback called after each enrichment with signature ``(estimator, mu)``, where mu is the parameter value of the last enrichment.
            
    Returns
    ---------
    flag:
        0 if tolerance reached, 1 if Nmax reached, 2 if training set exhausted.
    err:
        ``err[i]`` is the maximum error over the training set at iteration ``i``. Note that the last entry might be ``np.nan``, if the error of the last iteration was not computed (e.g. if ``Nmax`` is reached).
    idx: 
        Array of selected parameter indices, i.e. ``mu_train[idx]`` gives the selected parameter values.
    """
    
    print(f"Starting greedy for {estimator.__class__.__name__} estimator")
    start_time = time.time()
    
    mu_train = np.array(mu_train)
    
    err_decay = []
    selected_mu_idx = []
    original_idx = np.arange(len(mu_train))
    
    flag = 1
    start = 0
    
    if estimator.n == 0:
        start = 1
    
        print(f"{time.time() - start_time:6.1f}s: Iteration {0:3d}: initializing & updating estimator at mu_train[{0}]...")
        estimator.update(mu_train[0])
        
        if callback is not None:
            callback(estimator, mu_train[0])
        
        original_idx = np.delete(original_idx, 0)
        mu_train = np.delete(mu_train, 0, axis=0)
        selected_mu_idx.append(0)
        
        err = np.nan


    for n in range(start, N):
        
        if len(mu_train) == 0:
            flag = 2
            break
    
        err, idx = _get_max_error(estimator, mu_train)
        err_decay.append(err)
        mu_idx = original_idx[idx]
        
        if err <= tol:
            flag = 0
            break
        
        print(f"{time.time() - start_time:6.1f}s: Iteration {n:3d}{f" (n={estimator.n:3d})" if not start else ""}: max. error={err:.2e}, updating estimator at mu_train[{mu_idx}]...")
        
        estimator.update(mu_train[idx])
        
        if callback is not None:
            callback(estimator, mu_train[idx])
        
        original_idx = np.delete(original_idx, idx)
        mu_train = np.delete(mu_train, idx, axis=0)
        selected_mu_idx.append(mu_idx)
        
    codes= {0: "tolerance reached", 1: "Nmax reached", 2: "training set exhausted"}
        
    print(f"{time.time() - start_time:6.1f}s: Iteration {len(selected_mu_idx):3d}{f" (n={estimator.n:3d})" if not start else ""}: stopping due to {codes[flag]}, max. error={err:.2e}\n")
    
    if flag != 0:
        err_decay.append(np.nan)

    return flag, np.array(err_decay),  np.array(selected_mu_idx, dtype=int)
    
    