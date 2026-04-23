"""
Adding some specific FEniCSx related functionality.

Submodules
----------
.. autosummary::
   :toctree: generated/
   
   utils
   problems
   norms

Classes
-------
.. autosummary::
   :toctree: generated/
   
    AffineDirichletBC
    FEniCSxSpaceWithDirichletBCs
    
Functions
----------------
.. autosummary::
   :toctree: generated/
   
    free_dofs
"""

from ._dirichletbcs import *

_submodules = [
    'utils',
    'problems',
    'norms'
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]