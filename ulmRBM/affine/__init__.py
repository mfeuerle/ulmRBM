"""
Affine decomposition classes for parametric problems.

Affine Classes
--------------
.. autosummary::
   :toctree: generated/
   
    AffineObject
    AffineLinear
    AffineFunction
    
Approximating Affine Decompositions
-----------------------------------
.. autosummary::
    :toctree: generated/
    
    empirical_interpolation
    
Utilities
-----------------
.. autosummary::
    :toctree: generated/
    
    affine_kron
    wrap_affinelinear
    multiply_theta
    ScalarComponentList
"""

from ._affine import *
from ._eim import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]
