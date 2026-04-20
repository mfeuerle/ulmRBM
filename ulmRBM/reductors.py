r"""
Algorithms to construct reduced models.

Functions
-----------------
.. autosummary::
    :toctree: generated/
    
    greedy_algorithm

"""

import numpy as np
from collections.abc import Callable
from copy import copy
from warnings import warn
import time

from ulmRBM.core import Mu, Vector
from ulmRBM.products import InnerProduct
from ulmRBM.rom import ROM

__all__= [
    'greedy_algorithm',
]

def greedy_algorithm(
    rom: ROM[Mu],
    mu_train: list[Mu],
    Nmax: int,
    tol: float = 1e-5,
    strong: bool | list[Vector] = False,
    ortho: bool | InnerProduct[Mu] | tuple[bool | InnerProduct[Mu], int] = (True,40),
    update_stability: bool = True,
    update_continuity: bool = True,
    callback: Callable[[ROM[Mu], Mu, Vector], None] | None = None
):
    """
    Construct a reduced basis model using a greedy algorithm.

    Args:
        rom: 
            The reduced order model to enrich.
        mu_train: 
            Training set of parameter values.
        Nmax: 
            Maximum number of basis vectors to add (not total number).
        tol: 
            Error tolerance for stopping criterion.
        strong: 
            If True, use true FOM solutions for error computation (strong greedy). If a list, use these FOM solutions directly for a strong greedy. Default is False (use error estimator and thus weak greedy).
        ortho: 
            If True, orthonormalize the basis after each enrichment. If an InnerProduct, use it for orthonormalization. Default is True. For a tuple, a full orthonormalization is performed every specified number of enrichments, e.g. ``(True, 20)`` performs a orthonormalization in every iteration, and a full orthonormalization if ``rom.dim[1] % 20 == 0``, see :meth:`ROM.orthonormalize`.
        update_stability: 
            If True, update the stability constant estimator after each enrichment. Default is True.
        update_continuity: 
            If True, update the continuity constant estimator after each enrichment. Default is True.
        callback: 
            Optional callback called after each enrichment with signature (rom, mu, u_mu), where rom is the reduced model after enrichment, mu the parameter value of the last enrichment, and u_mu the corresponding FOM solution.

    Returns:
        flag: int
			0 if tolerance reached, 1 if Nmax reached, 2 if training set exhausted.
        err_decay: np.ndarray
			Array of maximum errors over the training set at each enrichment step. Note that the last entry might be ``np.nan``, indicating that the error was not computed after the last enrichment (e.g. if ``Nmax`` is reached).
        mu: list
			List of selected parameter values (mu) for each basis enrichment.
        mu_idx: np.ndarray
			Array of selected parameter indices with respect to the training set.
    """
    
    fom = rom.fom
    mu_train = copy(mu_train)
    mu_train = copy(mu_train)
    selected_mu = []
    selected_mu_idx = []
    err_decay = []
    Nstart = rom.dim[1]
    
    try:
        ortho, ortho_interval = ortho
    except TypeError:
        ortho_interval = np.inf
        
    if isinstance(ortho, bool):
        if ortho: U = None
    else:
        U = ortho
        ortho = True
        
    start_time = time.time()
    
    if isinstance(strong, bool):
        if strong: 
            print(f"{time.time() - start_time:6.1f}s: Computing full solutions for strong greedy...")
            u_fom = [fom.solve(mu) for mu in mu_train]
    else:
        if not isinstance(strong, list):
            raise ValueError("If 'strong' is not a boolean, it must be a list of FOM solutions.")
        if len(strong) != len(mu_train):
            raise ValueError("Length of provided FOM solutions does not match length of training set.")
        print(f"{time.time() - start_time:6.1f}s: Using provided full solutions for strong greedy")
        u_fom = copy(strong)
        strong = True
        
        
    if Nmax <= 0:
        raise ValueError("Nmax must be positive integer.")
    
    print(f"{time.time() - start_time:6.1f}s: Starting greedy algorithm: Nstart={Nstart}, Nmax={Nstart+Nmax}, tol={tol:.2e}, training_size={len(mu_train)}")
            
    if Nstart == 0:
        mu_idx = 0
        
        print(f"{time.time() - start_time:6.1f}s: Iteration 0:")
        print(f"{time.time() - start_time:6.1f}s:\t computing initial basis (idx={mu_idx}, N={rom.dim[1]})...")
        
        mu = mu_train.pop(0)
        u_mu = u_fom.pop(0) if strong else fom.solve(mu)
        rom.add_basis(u_mu)
        
        if update_stability:
            print(f"{time.time() - start_time:6.1f}s:\t updating stability estimator (idx={mu_idx})...")
            rom.estimate_fom_stability.update(mu)
        if update_continuity: 
            print(f"{time.time() - start_time:6.1f}s:\t updating continuity estimator (idx={mu_idx})...")
            rom.estimate_fom_continuity.update(mu)
        
        selected_mu.append(mu)
        selected_mu_idx.append(mu_idx)
        err = np.nan
        err_decay.append(np.nan)
        
        if callback is not None:
            print(f"{time.time() - start_time:6.1f}s:\t calling callback...")
            callback(rom, mu, u_mu)
        
    flag = 1 # Nmax reached
    while rom.dim[1] < Nstart+Nmax:
        
        if len(mu_train) == 0:
            flag = 2 # training set exhausted
            break
        
        print(f"{time.time() - start_time:6.1f}s: Iteration {rom.dim[1] - Nstart + 1}: Computing errors for {len(mu_train)} parameters...")
        if strong:
            err = np.array([rom.error(mu, u_fom=u_mu) for mu, u_mu in zip(mu_train, u_fom)])
        else:
            err = np.array([rom.estimate_error(mu) for mu in mu_train])
        
        idx = np.argmax(err)
        err = err[idx]
        err_decay[-1] = err
        
        # Calculate original index
        mu = mu_train[idx]
        mu_idx = idx + sum(selected_mu_idx <= idx)
        
        print(f"{time.time() - start_time:6.1f}s:\t Max error: {err:.2e} at idx={mu_idx}")
        
        if err < tol:
            print(f"{time.time() - start_time:6.1f}s:\t Tolerance reached! Stopping at N={rom.dim[1]}")
            flag = 0 # tolerance reached
            break
        
        print(f"{time.time() - start_time:6.1f}s:\t computing new basis (idx={mu_idx}, N={rom.dim[1]}) ...")
        mu = mu_train.pop(idx)
        u_mu = u_fom.pop(idx) if strong else fom.solve(mu)
        rom.add_basis(u_mu)
        
        if update_stability:
            print(f"{time.time() - start_time:6.1f}s:\t updating stability estimator (idx={mu_idx})...")
            rom.estimate_fom_stability.update(mu)
        if update_continuity: 
            print(f"{time.time() - start_time:6.1f}s:\t updating continuity estimator (idx={mu_idx})...")
            rom.estimate_fom_continuity.update(mu)
        if ortho:
            if rom.dim[1] % ortho_interval == 0:
                print(f"{time.time() - start_time:6.1f}s:\t orthonormalizing basis (full)...")
                rom.orthonormalize(U=U, full=True)
            else:
                print(f"{time.time() - start_time:6.1f}s:\t orthonormalizing basis...")
                rom.orthonormalize(U=U, full=False)
        
        selected_mu.append(mu)
        selected_mu_idx.append(mu_idx)
        err_decay.append(np.nan)
        
        if callback is not None:
            print(f"{time.time() - start_time:6.1f}s:\t calling callback...")
            callback(rom, mu, u_mu)
    
    codes= {0: "tolerance reached", 1: "Nmax reached", 2: "training set exhausted"}
    print(f"{time.time() - start_time:6.1f}s: Greedy algorithm completed: {codes[flag]} (final_error={err:.2e}, final_N={rom.dim[1]})")
    
    if flag:
        warn(f"Greedy algorithm stopped without reaching tolerance ({codes[flag]}). Current tolerance: {err}, target tolerance: {tol}.", UserWarning)
        
    return flag, np.array(err_decay), selected_mu, np.array(selected_mu_idx, dtype=int)