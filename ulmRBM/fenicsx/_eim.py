from collections.abc import Iterable, Callable
from itertools import product

import numpy as np
from dolfinx import fem

from ulmRBM.core import Mu
from ulmRBM.affine import AffineObject, empirical_interpolation
from ulmRBM.fenicsx import utils

__all__ = [
    'interpolate_function_eim',
]

def _get_points(U: fem.FunctionSpace):
    y = [None]
    def __get_points(x):
        y[0] = x.copy()
        dummy = np.zeros(U.value_shape).reshape(-1,1)
        return np.zeros((dummy.shape[0],x.shape[1]))
    fem.Function(U).interpolate(__get_points)
    return y[0]

def interpolate_function_eim(U: fem.FunctionSpace, 
                             func: Callable[[Mu, np.ndarray], np.ndarray], 
                             mus: Iterable[Mu],
                             Nmax: int = 50, 
                             tol: float = 1e-6) -> AffineObject[Mu, fem.Function]:
    r"""Empirical Interpolation (EIM) of a parametric function into a FEniCSx FunctionSpace.
    
    Small wrapper around `empirical_interpolation` to directly interpolate a parametric function into a discrete `dolfinx.fem.Function` defined on the given `dolfinx.fem.FunctionSpace`. To match the FEniCSx convention, the input of the function ``func`` is changed compared to `empirical_interpolation` and now read: ``func(mu, x)[i]`` is the function value at ``(mu, x[:,i])``.
    
    Args:
        U :
            The function space to interpolate into.
        func :
            The parametric function :math:`f_\mu` to approximate. Should be callable like ``func(mu, x) for mu in mus``, where ``func(mu, x)[i]`` is the function value at ``(mu, x[:,i])``.
        mus :
            The discrete parameter set :math:`\mathcal{P}_{\text{train}}`, see `empirical_interpolation`.
        Nmax :
            The maximum number of affine parts :math:`Q` in the EIM approximation, see `empirical_interpolation`.
        tol :
            The tolerance :math:`\varepsilon_{\text{tol}}` for the EIM approximation, see `empirical_interpolation`.
    """
    points = _get_points(U).T
    f = lambda mu, x: func(mu, x.T).T
    func_eim = AffineObject(empirical_interpolation(f, mus, points, Nmax=Nmax, tol=tol, continuous=False, residual=True))
    func_eim = func_eim.apply2data(lambda dataq: lambda x, data=dataq: data.T)
    func_eim = func_eim.apply2data(lambda dataq: utils.interpolate_function(U, dataq))
    return func_eim