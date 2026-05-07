"""
Affine decomposition classes for parametric problems.

Affine Classes
--------------
.. autosummary::
   :toctree: generated/
   
    AffineObject
    AffineLinear
    
Approximating Affine Decompositions
-----------------------------------
    
Wrapper Utilities
-----------------
.. autosummary::
    :toctree: generated/
    
    wrap_affinelinear
"""

from ._affine import *
from ._eim import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]
