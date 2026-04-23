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
    products
    solver
    fom
    rom
    reductors
"""

_submodules = [
    'affine',
    'core',
    'fom',
    'products',
    'rom',
    'solver',
    'reductors'
]

try:
    from . import fenicsx
    __doc__ += "\tfenicsx"
except ImportError:
    pass

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]


from ._dirty_patches import apply_patches
apply_patches()
del apply_patches