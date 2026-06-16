r"""
Algorithms to construct reduced models.

Functions
-----------------
.. autosummary::
    :toctree: generated/
    
    greedy_rbm
    pod_rbm
    pod_greedy_rbm
"""

from collections.abc import Callable, Sequence
import time

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from ulmRBM.core import NO_MU, Mu, Vector, Matrix
from ulmRBM.products import InnerProduct, MatrixInnerProduct, EuclideanInnerProduct
from ulmRBM.fom import TimeSteppingSolution
from ulmRBM.rom import ROM, StationaryTimeSteppingGalerkinROM

__all__= [
    'greedy_rbm',
    'pod_rbm',
    'pod_greedy_rbm',
]

def greedy_rbm(
    rom: ROM[Mu],
    mu_train: Sequence[Mu],
    N: int,
    tol: float = 1e-5,
    strong: bool | Sequence[Vector] = False,
    ortho: bool | InnerProduct[Mu] | Matrix = True,
    update_stability: bool = False,
    update_continuity: bool = False,
    display: bool = True,
    callback: Callable[[ROM[Mu], Mu, Vector], None] | None = None
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Construct a reduced basis model using a greedy algorithm.

    Args:
        rom: 
            The reduced order model to enrich.
        mu_train: 
            Training set of parameter values.
        N: 
            Maximum number of iterations / of basis vectors to add (not maximum size of the reduced model, which is  ``rom.shape[1] + N``).
        tol: 
            Error tolerance for stopping criterion.
        strong: 
            If True, use the true error (and thus compute the fom solutions for all parameters) (strong greedy). If False, use the online efficent error bound (weak greedy). Alternatively, provide the fom solutions to perform a strong greedy, in this case ``stong[i]`` should be the fom solution for ``mu_train[i]``.
        ortho: 
            If True, orthonormalize the basis after each enrichment. Alternatively, provide a `InnerProduct` or matrix compatible with `ROM.orthonormalize`, to orthonormalize the basis with respect to that product. Default is True.
        update_stability:
            Whether to update the stability estimator of the reduced model in each iteration using `StabilityEstimator.update`. This is useful if the stability estimator can not be setup seperately, e.g. using `greedy_constant_estimator`, or if the stability estimator should be enhanced further.
        update_continuity:
            Whether to update the continuity estimator of the reduced model in each iteration using `ContinuityEstimator.update`. This is useful if the continuity estimator can not be setup seperately, e.g. using `greedy_constant_estimator`, or if the continuity estimator should be enhanced further.
        display:
            Whether to print information during the greedy procedure.
        callback: 
            Optional callback function invoked after each enrichment with signature ``(rom, mu, u_mu)``, where ``rom`` is the reduced model after enrichment, ``mu`` the parameter value of the last enrichment, and ``u_mu`` the corresponding FOM solution.

    Returns
    ---------
    flag:
        0 if tolerance reached, 1 if N reached, 2 if training set exhausted.
    err:
        ``err[i]`` is the maximum error over the training set at iteration ``i``. Note that the last entry might be ``np.nan``, if the error of the last iteration was not computed (e.g. if ``N`` is reached).
    idx: 
        Array of selected parameter indices, i.e. ``mu_train[idx]`` gives the selected parameter values.
    """
    
    ########################
    # initialize
    
    start_time = time.time()
    
    fom = rom.fom
    mu_train = np.array(mu_train)
    
    if isinstance(ortho, bool):
        if ortho: U_ortho = None
    else:
        U_ortho = ortho
        ortho = True
        
    opening_message = lambda strong: print(f"{"Strong" if strong else "Weak"} greedy for basis selection with N={N}, tol={tol:.2e}, training size={len(mu_train)}")
        
    if isinstance(strong, bool):
        if display: opening_message(strong)
        if strong: 
            if display: print(f"{time.time() - start_time:6.1f}s: Computing {len(mu_train)} full solutions for strong greedy...")
            u_fom = np.asarray([fom.solve(mu) for mu in mu_train])
    else:
        if len(strong) != len(mu_train):
            raise ValueError("Length of provided FOM solutions does not match length of training set.")
        if display: opening_message(True)
        u_fom = np.array(strong)
        strong = True
    
    ########################
    # iteration mechanics
    
    err_decay = []
    selected_mu_idx = []
    original_idx = np.arange(len(mu_train))
    
    def _perform_iteration(idx):
        nonlocal rom, mu_train, u_fom, original_idx, selected_mu_idx
        
        if display: print(f"{time.time() - start_time:6.1f}s:\t computing snapshot...")
        u_mu = u_fom[idx] if strong else fom.solve(mu_train[idx])
        
        if display: print(f"{time.time() - start_time:6.1f}s:\t extending rom...")
        rom.add_basis(u_mu)
        
        if ortho:
            if display: print(f"{time.time() - start_time:6.1f}s:\t orthonormalizing...")
            rom.orthonormalize(U_ortho)
            
        if update_stability:
            if display: print(f"{time.time() - start_time:6.1f}s:\t updating stability estimator...")
            rom._fom_stability_estimator.update(mu_train[idx])
            
        if update_continuity:
            if display: print(f"{time.time() - start_time:6.1f}s:\t updating continuity estimator...")
            rom._fom_continuity_estimator.update(mu_train[idx])
            
        if callback is not None:
            callback(rom, mu_train[idx], u_mu)
        
        selected_mu_idx.append(original_idx[idx])
        original_idx = np.delete(original_idx, idx)
        mu_train = np.delete(mu_train, idx, axis=0)
        if strong:
            u_fom = np.delete(u_fom, idx, axis=0)
    
    ########################
    # first iteration

    start = 0
    if rom.shape[1] == 0:
        start = 1
        if display: print(f"{time.time() - start_time:6.1f}s: Iteration {0:3d}:")
        _perform_iteration(0)
    
    ########################
    # main loop
    
    flag = 1
    for n in range(start, N):
        
        if len(mu_train) == 0:
            flag = 2
            break
        
        if display: print(f"{time.time() - start_time:6.1f}s:\t computing errors for {len(mu_train)} parameters...")
        if strong:
            err = np.array([rom.error(mu, u_fom=u_mu) for mu, u_mu in zip(mu_train, u_fom)])
        else:
            err = np.array([rom.error_bound(mu) for mu in mu_train])
            
        if np.isinf(err.max()):
            inf_mask = np.argwhere(np.isinf(err)).flatten()
            idx = inf_mask[np.random.randint(0, len(inf_mask)-1)]
        else:
            idx = np.argmax(err)
        
        err = err[idx]
        err_decay.append(err)
        
        if err < tol:
            flag = 0
            break
        
        if display: print(f"{time.time() - start_time:6.1f}s: Iteration {len(selected_mu_idx):3d}{f" (N={rom.shape[1]:3d})" if not start else ""}: max. error={err:.2e} at mu_train[{original_idx[idx]}]")
        _perform_iteration(idx)
    
    ########################
    # finishing up
    
    codes= {0: "tolerance reached", 1: "Nmax reached", 2: "training set exhausted"}
    
    if display: print(f"{time.time() - start_time:6.1f}s: Iteration {len(selected_mu_idx):3d}{f" (N={rom.shape[1]:3d})" if not start else ""}: stopping due to {codes[flag]}, max. error={err:.2e}\n")
    
    if flag != 0:
        err_decay.append(np.nan)
    
    rom.assemble()
    return flag, np.array(err_decay), np.array(selected_mu_idx, dtype=int)


def pod_rbm(
    rom: ROM[Mu],
    mu_train: Sequence[Mu] | Sequence[Vector],
    N: int,
    tol: float = 1e-5,
    snapshots: bool = False,
    U: InnerProduct[Mu] | Matrix = None,
    display: bool = True
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Construct a reduced basis model using a greedy algorithm.

    Args:
        rom: 
            The reduced order model to enrich.
        mu_train: 
            Training set of parameter values.
        N: 
            Maximum number of basis vectors to add (not maximum size of the reduced model, which is  ``rom.shape[1] + N``).
        tol: 
            Error tolerance for stopping criterion.
        snapshots:
            If True, interpret the entries of ``mu_train`` as snapshots, i.e. a list of FOM solutions, rather than parameter values for which to compute solutions.
        U:
            Inner product to use for correlation matrix in the POD. If None, use ``rom.fom.U`` if it is parameter-independent, otherwise use the Euclidean inner product.
        display:
            Whether to print information.

    Returns
    ---------
    flag:
        0 if tolerance reached, 1 if N reached
    err:
        ``err[i-1]`` is the avarage error over the training set of the POD-ROM using ``i`` basis vectors, with i ranging from 1 to the number of basis vectors added.
    err_full:
        Same as ``err``, but ranging from 1 to ``len(mu_train)``, i.e. the error of the POD-ROM, if more basis vectors would be used.
    """
    
    ########################
    # initialize
    
    start_time = time.time()
    
    if display: print(f"POD for basis selection with N={N}, tol={tol:.2e}, training size={len(mu_train)}")
    
    fom = rom.fom
    
    if U is None:
        if fom.U.is_parametric:
            if display: print("Warning: FOM inner product is parametric, using Euclidean inner product for POD.")
            U = EuclideanInnerProduct(fom.shape[1])
        else:
            U = fom.U
    else:
        if not isinstance(U, InnerProduct):
            U = MatrixInnerProduct(U)
        if U.is_parametric:
            raise ValueError("POD can not be used with parametric inner products, please provide a parameter-independent inner product.")
    
    if snapshots:
        S = np.asarray(mu_train)
    else:
        if display: print(f"{time.time() - start_time:6.1f}s: computing {len(mu_train)} full solutions for POD...")
        S = np.asarray([fom.solve(mu) for mu in mu_train])
        
    N_train = len(mu_train)
    
    if display: print(f"{time.time() - start_time:6.1f}s: computing correlation matrix...")
    C  = 1/N_train * U.inner(NO_MU, S.T, S.T)
    
    if display: print(f"{time.time() - start_time:6.1f}s: solving eigenvalue problem...")
    v, E = eigh(C)
    v = abs(v)
    err = np.r_[0, np.sqrt(np.cumsum(v)), np.inf]
    
    n = np.argmax(err[::-1] < tol)
    
    flag = 0
    if n > N:
        flag = 1
        n = N
    
    if display: print(f"{time.time() - start_time:6.1f}s: extending rom by {n} basis vectors...")
    basis = 1/np.sqrt(v[-n:]) * (S.T @ E[:,-n:])
    rom.add_basis(basis)
    
    if display: print(f"{time.time() - start_time:6.1f}s: orthonormalizing...")
    rom.orthonormalize()
    
    codes= {0: "tolerance reached", 1: "Nmax reached"}
    
    if display: print(f"{time.time() - start_time:6.1f}s: finished ({codes[flag]}); added {n:3d}{f" (i.e. N={rom.shape[1]:3d})" if rom.shape[1] > n else ""} basis functions, avg. error={err[-n-1]:.2e}\n")
    
    return flag, err[-n-1:-1][::-1], err[-2::-1]


def pod_greedy_rbm(
    rom: StationaryTimeSteppingGalerkinROM[Mu],
    mu_train: Sequence[Mu],
    N: int,
    tol: float = 1e-5,
    Npod: int = 1,
    strong: bool | Sequence[TimeSteppingSolution] = False,
    ortho: bool | InnerProduct[Mu] | Matrix = True,
    update_LI_stability: bool = False,
    update_LE_continuity: bool = False,
    W: InnerProduct[Mu] | Matrix = None,
    display: bool = True,
    eig: str = "direct",
    callback: Callable[[StationaryTimeSteppingGalerkinROM[Mu], Mu, TimeSteppingSolution], None] | None = None
) -> tuple[int, np.ndarray, np.ndarray]:
    r"""Construct a reduced basis model for time-stepping problems using a POD-greedy algorithm.
    
    Args:
        rom: 
            The reduced order model to enrich.
        mu_train: 
            Training set of parameter values.
        N: 
            Maximum number of basis vectors to add (not maximum size of the reduced model, which is  ``rom.n + N``).
        tol: 
            Error tolerance for stopping criterion.
        Npod: 
            The number of POD modes added in each iteration.
        strong: 
            If True, use the true error (and thus compute the fom solutions for all parameters) (strong POD-greedy). If False, use the online efficent error bound (weak POD-greedy). Alternatively, provide the fom solutions to perform a strong greedy, in this case ``stong[i]`` should be the fom solution for ``mu_train[i]``.
        ortho: 
            If True, orthonormalize the basis after each enrichment. Alternatively, provide a `InnerProduct` or matrix compatible with `StationaryTimeSteppingGalerkinROM.orthonormalize`, to orthonormalize the basis with respect to that product. Default is True.
        update_LI_stability:
            Whether to update the used stability estimator for `StationaryTimeSteppingGalerkinFOM.LI` in each iteration using `StabilityEstimator.update`. This is useful if the stability estimator can not be setup seperately, e.g. using `greedy_constant_estimator`, or if the stability estimator should be enhanced further.
        update_LE_continuity:
            Whether to update the used continuity estimator for `StationaryTimeSteppingGalerkinFOM.LE` in each iteration using `ContinuityEstimator.update`. This is useful if the continuity estimator can not be setup seperately, e.g. using `greedy_constant_estimator`, or if the continuity estimator should be enhanced further.
        W:
            Inner product used for the POD. If None, uses ``rom.fom.U`` if it is parameter-independent, otherwise the Euclidean inner product.
        display:
            Whether to print information during the greedy procedure.
        eig:
            The method for solving the eigenvalue problem in the POD. Can be either ``'direct'`` for a direct solver, or ``'iterative'`` for an iterative solver. As ``Npod`` is usually small, the iterative solver can be more efficient, especially for large full-order models.
        callback: 
            Optional callback function invoked after each enrichment with signature ``(rom, mu, u_mu)``, where ``rom`` is the reduced model after enrichment, ``mu`` the parameter value of the last enrichment, and ``u_mu`` the corresponding FOM solution.

    Returns
    ---------
    flag:
        0 if tolerance reached, 1 if N reached.
    err:
        ``err[i]`` is the maximum error over the training set and time points at iteration ``i``. Note that the last entry might be ``np.nan``, if the error of the last iteration was not computed (e.g. if ``N`` is reached).
    idx: 
        Array of selected parameter indices, i.e. ``mu_train[idx]`` gives the selected parameter values.
    """
    
    ########################
    # initialize
    
    start_time = time.time()
    
    fom = rom.fom
    
    if eig not in ["direct", "iterative"]:
        raise ValueError("Invalid value for eig, expected 'direct' or 'iterative'.")
    else:
        if eig == "direct":
            eig = lambda C, N: eigh(C, subset_by_index=(C.shape[0]-N, C.shape[0]-1))
        else:
            eig = lambda C, N: eigsh(C, k=N, which='LA')
    
    if W is None:
        if fom.W.is_parametric:
            if display: print("Warning: FOM inner product is parametric, using Euclidean inner product for POD.")
            W = EuclideanInnerProduct(fom.n)
        else:
            W = fom.W
    else:
        if not isinstance(W, InnerProduct):
            W = MatrixInnerProduct(W)
        if W.is_parametric:
            raise ValueError("POD can not be used with parametric inner products, please provide a parameter-independent inner product.")
        
    if isinstance(ortho, bool):
        if ortho: U_ortho = None
    else:
        U_ortho = ortho
        ortho = True
    
    opening_message = lambda strong: print(f"{"Strong" if strong else "Weak"} POD-greedy for basis selection with N={N}, tol={tol:.2e}, training size={len(mu_train)}")
    if isinstance(strong, bool):
        if display: opening_message(strong)
        if strong: 
            print(f"{time.time() - start_time:6.1f}s: Computing {len(mu_train)} full solutions for strong POD-greedy...")
            u_fom = np.asarray([fom.solve(mu) for mu in mu_train], dtype=TimeSteppingSolution)
    else:
        if len(strong) != len(mu_train):
            raise ValueError("Length of provided FOM solutions does not match length of training set.")
        if display: opening_message(True)
        u_fom = np.array(strong, dtype=TimeSteppingSolution)
        strong = True
        
        
    ########################
    # iteration mechanics
    
    err_decay = []
    selected_mu_idx = []
    
    def _perform_iteration(idx, Npod):
        nonlocal rom, mu_train, u_fom, selected_mu_idx
        
        if display: print(f"{time.time() - start_time:6.1f}s:\t computing snapshot...")
        u_mu = u_fom[idx].u if strong else fom.solve(mu_train[idx]).u
        if rom.n > 0:
            E = u_mu - rom.reconstruct(mu_train[idx]).u
        else:
            E = u_mu
        
        if display: print(f"{time.time() - start_time:6.1f}s:\t computing POD modes...")
        C  = 1/(rom.K+1) * W.inner(NO_MU, E, E)
        val, vec = eig(C, Npod)
        basis = 1/np.sqrt(val) * (u_mu @ vec)
        
        if display: print(f"{time.time() - start_time:6.1f}s:\t extending rom by {Npod} basis vectors...")
        rom.add_basis(basis)
        
        if ortho:
            if display: print(f"{time.time() - start_time:6.1f}s:\t orthonormalizing...")
            rom.orthonormalize(U_ortho)
            
        if update_LI_stability:
            if display: print(f"{time.time() - start_time:6.1f}s:\t updating LI stability estimator...")
            rom._LI_stability_estimator.update(mu_train[idx])
            
        if update_LE_continuity:
            if display: print(f"{time.time() - start_time:6.1f}s:\t updating LE continuity estimator...")
            rom._LE_continuity_estimator.update(mu_train[idx])
            
        if callback is not None:
            callback(rom, mu_train[idx], u_mu)
        
        selected_mu_idx.append(idx)
    
    ########################
    # first iteration

    start = 0
    if rom.n == 0:
        start = 1
        print(f"{time.time() - start_time:6.1f}s: Iteration {0:3d}:")
        _perform_iteration(0, np.min([Npod, N]))
    
    ########################
    # main loop
    
    flag = 1
    n = start
    while n*Npod < N:
        
        print(f"{time.time() - start_time:6.1f}s:\t computing errors for {len(mu_train)} parameters...")
        if strong:
            err = np.array([np.max(rom.error(mu, u_fom=u_mu)) for mu, u_mu in zip(mu_train, u_fom)])
        else:
            err = np.array([np.max(rom.error_bound(mu)) for mu in mu_train])
            
        if np.isinf(err.max()):
            inf_mask = np.argwhere(np.isinf(err)).flatten()
            idx = inf_mask[np.random.randint(0, len(inf_mask)-1)]
        else:
            idx = np.argmax(err)
        
        err = err[idx]
        err_decay.append(err)
        
        if err < tol:
            flag = 0
            break
        
        print(f"{time.time() - start_time:6.1f}s: Iteration {n:3d}{f" (N={rom.n:3d})" if not start or Npod != 1 else ""}: max. error={err:.2e} at mu_train[{idx}]")
        _perform_iteration(idx, np.min([Npod, N - n*Npod]))
        n += 1
    
    ########################
    # finishing up
    
    codes= {0: "tolerance reached", 1: "Nmax reached"}
    
    print(f"{time.time() - start_time:6.1f}s: Iteration {len(selected_mu_idx):3d}{f" (N={rom.n:3d})" if not start or Npod != 1 else ""}: stopping due to {codes[flag]}, max. error={err:.2e}\n")
    
    if flag != 0:
        err_decay.append(np.nan)
    
    rom.assemble()
    return flag, np.array(err_decay), np.array(selected_mu_idx, dtype=int)