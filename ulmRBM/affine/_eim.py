"""
Empirical Interpolation Method for approximating affine decompositions.
"""

from __future__ import annotations

__all__ = [
    'empirical_interpolation',
]


from collections.abc import Iterable, Callable
import time

import numpy as np
from functools import partial

from ulmRBM.core import Mu
from ulmRBM.affine import AffineLinear, AffineFunction, ScalarComponentList


    
def _scale_func(alpha, func):
    """Scale a function by a scalar factor alpha."""
    return lambda *args, **kwargs: alpha * np.asarray(func(*args, **kwargs))

def _eim_interpolation(G, GQ, TQ):
    """EIM interpolation of [g1,g2,...] (discrete x). """
    GGQ = GQ[:,*(x for x in TQ.T)].T
    GG  = G [:,*(x for x in TQ.T)].T
    alpha = np.linalg.solve(GGQ, GG)
    return (GQ.T @ alpha).T

def _eim_interpolation_continuous(g, GQ, GQ_continuous, TQ):
    """EIM interpolation of g (continous in x). """
    GGQ = GQ[:,*(x for x in TQ.T)].T
    g   = g [*(x for x in TQ.T)]
    alpha = np.linalg.solve(GGQ, g)
    return lambda x: sum(ai * gi(x) for ai, gi in zip(alpha, GQ_continuous))


def _eim_alpha(func, GQ, TQ, points):
    """EIM interpolation coefficients alpha(mu) for func(mu) (continous in mu). """
    GGQ = GQ[:,*(x for x in TQ.T)].T
    pointsQ = points[TQ[:,0]]
    
    def get_alpha(mu):
        g = np.asarray(func(mu, pointsQ))
        if TQ.shape[1] > 1: g = g[range(g.shape[0]),*(x for x in TQ[:,1:].T)]
        return np.linalg.solve(GGQ, g)
    
    return ScalarComponentList(get_alpha, len(GQ))

        
def empirical_interpolation[Point](func: Callable[[Mu, Iterable[Point]], Iterable[float | np.ndarray[float]]],
                                   mus: Iterable[Mu],
                                   points: Iterable[Point],
                                   Nmax: int = 50,
                                   tol: float = 1e-6,
                                   continuous: bool = False,
                                   residual = None) -> AffineLinear[Mu, np.ndarray] | AffineFunction[Mu]:
    r"""Empirical Interpolation Method (EIM) for approximating affine decompositions.
    
    Let :math:`\mathcal{P}` be a parameter space, :math:`\Omega` some domain and :math:`R` a indexable range space, e.g. :math:`R\subset\mathbb{R}`, :math:`R\subset\mathbb{R}^n`, :math:`R\subset\mathbb{R}^{n_1\times n_2}` and so on.
    
    Given a parametric function :math:`f_\mu:\Omega\to R`, :math:`\mu\in\mathcal{P}`, the EIM approximates this function by an affine decomposition w.r.t. the parameter, i.e.
    
    .. math::
        f_\mu \approx f_\mu^{\text{EIM}} = \sum_{q=1}^Q \alpha_i(\mu) g_i
    
    Thereby, :math:`g_i` are either given by :math:`g_i = f_{\mu_i}` (``residual=False``) or by :math:`g_i = f_{\mu_i} - f_{\mu_i}^{\text{EIM},i-1}` (``residual=True``) for some :math:`\mu_i\in\mathcal{P}`, where :math:`f_{\mu_i}^{\text{EIM},i-1}` is the EIM approximation of :math:`f_{\mu_i}` using only :math:`g_1,\dots,g_{i-1}`.
    
    As :math:`\mathcal{P}` and :math:`\Omega` are typically infinite, the EIM approximation is based on finite discretizations :math:`\mathcal{P}_{\text{train}}\subset \mathcal{P}` and :math:`\Omega_{\text{train}}\subset \Omega`. If the approximation :math:`f_\mu^{\text{EIM}}` is only needed on the discrete set :math:`\Omega_{\text{train}}`, use ``continuous=False``. In this case, :math:`f_\mu^{\text{EIM}}` is a discrete vector with :math:`[f_\mu^{\text{EIM}}]_i \approx f_\mu(x_i) \in R` for :math:`\{x_1,x_2,\dots\} = \Omega_{\text{train}}`. If the approximation :math:`f_\mu^{\text{EIM}}` is needed on arbitrary points :math:`x\in\Omega` use ``continuous=True``. In this case, :math:`f_\mu^{\text{EIM}}` is a function on :math:`\Omega` with :math:`f_\mu^{\text{EIM}}(x) \approx f_\mu(x) \in R` for :math:`x\in \Omega`.
    
    The algorithm terminates, if the worst-case approximation on :math:`\mathcal{P}_{\text{train}}\times\Omega_{\text{train}}\times R` is smaller then a tolerance, i.e.
    
    .. math::
        \max_{\mu\in \mathcal{P}_{\text{train}}}\max_{x\in \Omega_{\text{train}}} \|f_\mu(x) - f_\mu^{\text{EIM}}(x)\|_{R;\infty} < \varepsilon_{\text{tol}},
    
    or if the maximum number of affine parts :math:`Q` is reached.
    
    Args:
        func :
            The parametric function :math:`f` to approximate. Should be callable like ``func(mu, points) for mu in mus``, where ``func(mu, points)[i]`` is the function value at ``mu, points[i]``.
        mus :
            The discrete parameter set :math:`\mathcal{P}_{\text{train}}`.
        points :
            The discrete set :math:`\Omega_{\text{train}}`.
        Nmax :
            The maximum number of affine parts :math:`Q` in the EIM approximation.
        tol :
            The tolerance :math:`\varepsilon_{\text{tol}}` for the EIM approximation.
        continuous :
            Whether the returned EIM approximation is a continuous function (`AffineFunction`) on :math:`\Omega` (``True``), or a discrete vector (`AffineLinear`) on :math:`\Omega_{\text{train}}` (``False``). 
        residual :
            Whether to use the residual in the EIM iteration, i.e. :math:`g_i = f_{\mu_i} - f_{\mu_i}^{\text{EIM},i-1}` (``True``) which is more stable or :math:`g_i = f_{\mu_i}` (``False``). For ``continuous=True``, the evaluation of :math:`f_\mu^{\text{EIM}}(x)` is significantly more expensive if ``residual=True``, while there is no difference in speed for ``continuous=False``. Thus, if ``None``, the default is to use the residual if and only if ``continuous=False``. 
    """
    
    if residual is None:
        residual = not continuous
    
    start_time = time.time()
    
    G = np.asarray([func(mu, points) for mu in mus])
    
    if not np.issubdtype(G.dtype, np.number):
        raise ValueError(f"func(mu, points) must return an numpy array of numerical values, but got {G.dtype}")
    
    if G.shape[0] != len(mus) or G.shape[1] != len(points):
        raise ValueError("func(mu,points) must return an iterable of length len(points) for each mu in mus.")
    
    mu_idx, *x_idx = np.unravel_index(np.argmax(abs(G)), G.shape)
    
    q  = G[mu_idx]
    qx = G[mu_idx, *x_idx]
    g  = q/qx
    
    if continuous:
        q_continuous = partial(func, mus[mu_idx])
        g_continuous = _scale_func(1/qx, q_continuous)
        GQ_continuous = [g_continuous]
    
    GQ = g.reshape(1,*g.shape)
    TQ = np.asarray(x_idx).reshape(1,-1)
    
    for i in range(1, Nmax):
        R = G - _eim_interpolation(G, GQ, TQ)
        
        mu_idx, *x_idx = np.unravel_index(np.argmax(abs(R)), R.shape)
        
        if abs(R[mu_idx, *x_idx]) < tol:
            break
        
        if residual:
            r  = R[mu_idx]
            rx = R[mu_idx, *x_idx]
            g  = r/rx
        else:
            q  = G[mu_idx]
            qx = G[mu_idx, *x_idx]
            g  = q/qx
        
        if continuous:
            q_continuous  = partial(func, mus[mu_idx])
            if residual:
                Iq_continuous = _eim_interpolation_continuous(G[mu_idx], GQ, GQ_continuous, TQ)
                r_continuous  = lambda x, q=q_continuous, Iq=Iq_continuous: q(x) - Iq(x)
                g_continuous  = _scale_func(1/rx, r_continuous)
            else:
                g_continuous  = _scale_func(1/qx, q_continuous)
            GQ_continuous += [g_continuous]
        
        GQ = np.vstack([GQ, [g]])
        TQ = np.vstack([TQ, [x_idx]])
        
    if abs(R[mu_idx, *x_idx]) < tol:
        print(f"EIM converged in {time.time() - start_time:5.1f}s after {len(GQ):3d} iterations with error {abs(R[mu_idx, *x_idx]):.2e} < tol={tol:.2e}.")
    else:
        print(f"EIM did not converge in {time.time() - start_time:5.1f}s after {len(GQ):3d} iterations with final error {abs(R[mu_idx, *x_idx]):.2e}.")
    
    alpha = _eim_alpha(func, GQ, TQ, points)
    
    if continuous:
        return AffineFunction(alpha, GQ_continuous)
    else:
        return AffineLinear(alpha, GQ)
