r""" 
ulmRBM Python Package
=====================================================

This is the ulmRBM Python module for Reduced Basis Methods for parametrized partial differential equations (PDEs).

Modules
--------------
.. autosummary::
   :toctree: generated/
   
    core
    affine
"""

from . import core, affine

__all__ = [
    'affine',
    'core',
]

try:
    from . import fenicsx as _fenicsx
    __all__.append('fenicsx')
    __doc__ += "\tfenicsx"
except ImportError:
    pass


# add the decorators @override and @extend for functions and methods

from ._dirty_patches  import _patch_priority, _patch_MatrixLinearOperator_matmat

_patch_MatrixLinearOperator_matmat()

replaced_operators = (
    "__add__", "__sub__",
    "__eq__", "__ne__", "__ge__", "__gt__", "__le__", "__lt__", 
    "__matmul__", 
    "__mul__", "__div__", "__truediv__",
)

import scipy.sparse as _sparse
from scipy.sparse._base import _spbase

sparse_types = [_spbase] + [type_ for type_ in _sparse.__dict__.values() if isinstance(type_, type) and issubclass(type_, _spbase)]

_patch_priority(sparse_types, "__sparse_priority__", replaced_operators)


import scipy.sparse.linalg as _linalg
from scipy.sparse.linalg import LinearOperator as _LinearOperator

linop_types = [_LinearOperator] + [type_ for type_ in _linalg.__dict__.values() if isinstance(type_, type) and issubclass(type_, _LinearOperator)]

_patch_priority(linop_types, "__linop_priority__", replaced_operators)

