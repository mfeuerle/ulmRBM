r"""
Algorithms to construct reduced models.

Functions
-----------------
.. autosummary::
    :toctree: generated/
    
    greedy_rbm
    pod_rbm
"""

from collections.abc import Callable, Sequence
import time

import numpy as np
from scipy.linalg import eigh

from ulmRBM.core import NO_MU, Mu, Vector, Matrix
from ulmRBM.products import InnerProduct, MatrixInnerProduct, EuclideanInnerProduct
from ulmRBM.rom import ROM

__all__= [
    'greedy_rbm',
    'pod_rbm',
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
            Maximum number of iterations / of basis vectors to add (not maximum size of the reduced model, which is  ``rom.dim[1] + N``).
        tol: 
            Error tolerance for stopping criterion.
        strong: 
            If True, use the true error (and thus compute the fom solutions for all parameters) (strong greedy). If False, use the online efficent error bound (weak greedy). Alternatively, provide the fom solutions to perform a strong greedy, in this case ``stong[i]`` should be the fom solution for ``mu_train[i]``.
        ortho: 
            If True, orthonormalize the basis after each enrichment. Alternatively, provide a `InnerProduct` or matrix compativle with `ROM:orthonormalize`, to orthonormalize the basis with respect to that product. If an InnerProduct, use it for orthonormalization. Default is True.
        update_stability:
            Whether to update the stability estimator of the reduced model in each iteration using `StabilityEstimator.update`. This is useful if the stability estimator can not be setup seperately, e.g. using `greedy_constant_estimator`, or if the stability estimator should be enhanced further.
        update_continuity:
            Whether to update the continuity estimator of the reduced model in each iteration using `ContinuityEstimator.update`. This is useful if the continuity estimator can not be setup seperately, e.g. using `greedy_constant_estimator`, or if the continuity estimator should be enhanced further.
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
        if ortho: U = None
    else:
        U = ortho
        ortho = True
        
    opening_message = lambda strong: print(f"{"Strong" if strong else "Weak"} greedy for basis selection with N={N}, tol={tol:.2e}, training size={len(mu_train)}")
        
    if isinstance(strong, bool):
        opening_message(strong)
        if strong: 
            print(f"{time.time() - start_time:6.1f}s: Computing {len(mu_train)} full solutions for strong greedy...")
            u_fom = np.asarray([fom.solve(mu) for mu in mu_train])
    else:
        if len(strong) != len(mu_train):
            raise ValueError("Length of provided FOM solutions does not match length of training set.")
        opening_message(True)
        u_fom = np.array(strong)
        strong = True
    
    ########################
    # iteration mechanics
    
    err_decay = []
    selected_mu_idx = []
    original_idx = np.arange(len(mu_train))
    
    def _perform_iteration(idx):
        nonlocal rom, mu_train, u_fom, original_idx, selected_mu_idx
        
        print(f"{time.time() - start_time:6.1f}s:\t computing snapshot...")
        u_mu = u_fom[idx] if strong else fom.solve(mu_train[idx])
        
        print(f"{time.time() - start_time:6.1f}s:\t extending rom...")
        rom.add_basis(u_mu)
        
        if ortho:
            print(f"{time.time() - start_time:6.1f}s:\t orthonormalizing...")
            rom.orthonormalize(U)
            
        if update_stability:
            print(f"{time.time() - start_time:6.1f}s:\t updating stability estimator...")
            rom._fom_stability_estimator.update(mu_train[idx])
            
        if update_continuity:
            print(f"{time.time() - start_time:6.1f}s:\t updating continuity estimator...")
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
    if rom.dim[1] == 0:
        start = 1
        print(f"{time.time() - start_time:6.1f}s: Iteration {0:3d}:")
        _perform_iteration(0)
    
    ########################
    # main loop
    
    flag = 1
    for n in range(start, N):
        
        if len(mu_train) == 0:
            flag = 2
            break
        
        print(f"{time.time() - start_time:6.1f}s:\t computing errors for {len(mu_train)} parameters...")
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
        
        print(f"{time.time() - start_time:6.1f}s: Iteration {rom.dim[1]:3d}{f" (N={rom.dim[1]:3d})" if not start else ""}: max. error={err:.2e} at mu_train[{original_idx[idx]}]")
        _perform_iteration(idx)
    
    ########################
    # finishing up
    
    codes= {0: "tolerance reached", 1: "Nmax reached", 2: "training set exhausted"}
    
    print(f"{time.time() - start_time:6.1f}s: Iteration {len(selected_mu_idx):3d}{f" (N={rom.dim[1]:3d})" if not start else ""}: stopping due to {codes[flag]}, max. error={err:.2e}\n")
    
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
    U: InnerProduct[Mu] | Matrix = None
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Construct a reduced basis model using a greedy algorithm.

    Args:
        rom: 
            The reduced order model to enrich.
        mu_train: 
            Training set of parameter values.
        N: 
            Maximum number of basis vectors to add (not maximum size of the reduced model, which is  ``rom.dim[1] + N``).
        tol: 
            Error tolerance for stopping criterion.
        snapshots:
            If True, interpret the entries of ``mu_train`` as snapshots, i.e. a list of FOM solutions, rather than parameter values for which to compute solutions.
        U:
            Inner product to use for correlation matrix in the POD. If None, use ``rom.fom.U`` if it is parameter-independent, otherwise use the Euclidean inner product.

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
    
    print(f"POD for basis selection with N={N}, tol={tol:.2e}, training size={len(mu_train)}")
    
    fom = rom.fom
    
    if U is None:
        if fom.U.is_parametric:
            print("Warning: FOM inner product is parametric, using Euclidean inner product for POD.")
            U = EuclideanInnerProduct(fom.dim[1])
        else:
            U = fom.U
    else:
        if not isinstance(U, InnerProduct):
            U = MatrixInnerProduct(U)
        if U.is_parametric:
            raise ValueError("POD can not be used with parametric inner products, please provide a parameter-independent inner product.")
    
    if snapshots:
        snapshots = np.asarray(mu_train)
    else:
        print(f"{time.time() - start_time:6.1f}s: computing {len(mu_train)} full solutions for POD...")
        snapshots = np.asarray([fom.solve(mu) for mu in mu_train])
        
    N_train = len(mu_train)
    
    print(f"{time.time() - start_time:6.1f}s: computing correlation matrix...")
    C  = 1/N_train * U.inner(NO_MU, snapshots.T, snapshots.T)
    
    print(f"{time.time() - start_time:6.1f}s: solving eigenvalue problem...")
    v, E = eigh(C)
    v = abs(v)
    err = np.r_[0, np.sqrt(np.cumsum(v)), np.inf]
    
    n = np.argmax(err[::-1] < tol)
    
    flag = 0
    if n > N:
        flag = 1
        n = N
    
    print(f"{time.time() - start_time:6.1f}s: extending rom by {n} basis vectors...")
    basis = 1/np.sqrt(v[-n:]) * (snapshots.T @ E[:,-n:])
    rom.add_basis(basis)
    
    print(f"{time.time() - start_time:6.1f}s: orthonormalizing...")
    rom.orthonormalize()
    
    codes= {0: "tolerance reached", 1: "Nmax reached"}
    
    print(f"{time.time() - start_time:6.1f}s: finished ({codes[flag]}); added {n:3d}{f" (i.e. N={rom.dim[1]:3d})" if rom.dim[1] > n else ""} basis functions, avg. error={err[-n-1]:.2e}\n")
    
    return flag, err[-n-1:-1][::-1], err[-2::-1]