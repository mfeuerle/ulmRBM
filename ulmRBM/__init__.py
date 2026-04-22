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

from ._dirty_patches import apply_patches
apply_patches()
del apply_patches

from . import affine, core, fom, products, rom, solver, reductors

__all__ = [
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
    __all__.append('fenicsx')
    __doc__ += "\tfenicsx"
except ImportError:
    pass


# add the decorators @override and @extend for functions and methods
