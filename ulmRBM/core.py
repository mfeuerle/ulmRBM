"""
Core types and utilities for the ulmRBM package.

This module provides fundamental type aliases, abstract base classes,
and wrapper utilities used throughout the package.

Type Aliases
--------------
.. currentmodule:: ulmRBM.core

.. autodata:: Matrix
.. autodata:: Vector
.. autodata:: Data
.. autodata:: Mu
.. autodata:: NO_MU
    
Abstract Base Classes
----------------------
.. autosummary::
   :toctree: generated/
   
    ParametricObject
    ParametricLinear
    TrivialParametric
    
Wrapper Utilities
-----------------
.. autosummary::
    :toctree: generated/
    
    wrap_linear
    wrap_scalar
    unwrap
"""

from __future__ import annotations

__all__ = [
    'Mu', 'Data', 'Matrix', 'Vector', 'NO_MU',
    'ParametricObject', 'ParametricLinear', 'TrivialParametric',
    'wrap_linear', 'wrap_scalar', 'unwrap', 'KRON_AVAILABLE',
]

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import sparray

from typing import TypeAlias, TypeVar, Generic
from abc import ABC, abstractmethod
from numbers import Number
    
Mu = TypeVar("Mu")
"""TypeAlias variable for parameter types."""

Data = TypeVar("Data")
"""TypeAlias variable for data types."""

Matrix: TypeAlias = np.ndarray | sparray | LinearOperator
"""Type alias for matrices, which can be dense :class:`numpy.ndarray`, sparse :class:`scipy.sparse.sparray`, Kronecker products :class:`kron.kron_base`, or matrix-free :class:`scipy.sparse.linalg.LinearOperator`."""

try:
    import kron
    Matrix: TypeAlias = Matrix | kron.kron_base
    KRON_AVAILABLE = True
except ImportError:
    KRON_AVAILABLE = False

    
Vector: TypeAlias = np.ndarray
"""Type alias for vectors or collections of vectors (always dense numpy arrays)."""

NO_MU = None
"""Special value to be used as parameter when calling a parametric object that is actually parameter-independent to increase readability."""



class ParametricObject(ABC, Generic[Mu, Data]):
    r"""
    A parametric object of type ``Data`` depending on a parameter of type ``Mu``.
    
    Represents an abstact object that can be called using ``object(mu)`` returning a value of type ``Data`` for a parameter value ``mu`` of type ``Mu``.
    """
    
    @abstractmethod
    def __call__(self, mu: Mu) -> Data:
        r"""
        Evaluate the parametric object at the given parameter value.
        
        Args:
            mu :
                Single parameter value at which to evaluate the object.
            
        Returns:
            The value of the parametric object. The type matches the type ``Data``.
        """
        ...
        
class ParametricLinear(ParametricObject[Mu, Data]):
    r"""
    A parametric linear object of type ``Data`` depending on a parameter of type ``Mu``.
    
    For example a parameter dependend matrix or vector.
    """
    
    shape: tuple[int, ...]
    """Shape of the parametric linear object, e.g., (m,n) for a matrix or (n,) for a vector."""
    
    
class TrivialParametric(ParametricObject[Mu, Data]):
    """A parametric object that is constant with respect to the parameter.
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.eye(3)
    >>> wrapped = TrivialParametric(A)
    >>> wrapped(None)  # Returns A regardless of parameter
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    
    def __init__(self, data: Data):
        self.data: Data = data
        """The underlying static data wrapped by this parametric object."""
        
    def __repr__(self):
        return f"<{self.__class__.__name__} wrapping: {self.data.__repr__()}>"
        
    def __call__(self, mu: Mu = NO_MU, *args, **kwargs) -> Data:
        return self.data
    
    def __getattr__(self, name: str):
        r"""Delegate attribute access to the wrapped data. 
        This is only called if the attribute is not found on self.
        Thus, this object behaves like the wrapped object for all attributes
        and just adds __call__ so that it also fits the ParametricObject interface."""
        return getattr(self.data, name)
    
    
def wrap_linear(v: Matrix | Vector | any) -> TrivialParametric[Mu, Matrix | Vector] | any:
    """
    Wrap static linear object (e.g. a matrix or a vector) as :class:`TrivialParametric`. 
    
    Only checks if ``v`` is a Vector or Matrix instance and wraps it. All other inputs are returned unchanged."""
    if isinstance(v, Matrix | Vector):
        v = TrivialParametric(v)
    return v

def wrap_scalar(c: Number | any) -> TrivialParametric[Mu, Number] | any:
    """
    Wrap a static scalar as a :class:`TrivialParametric` if it is a number. 
    
    Only checks if ``c`` is a number and wraps it. All other inputs are returned unchanged. 
    """
    if isinstance(c, Number):
        c = TrivialParametric(c)
    return c

def unwrap(obj):
    """
    Unwrap :class:`TrivialParametric` to get the underlying data. 
    
    Only checks if ``obj`` is a :class:`TrivialParametric` and returns :attr:`~TrivialParametric.data`. All other inputs are returned unchanged.
    """
    return obj.data if isinstance(obj, TrivialParametric) else obj