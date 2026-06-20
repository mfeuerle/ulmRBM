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
    SpaceTimeKey
    SpaceTimeAffineDirichletBC
    SpaceTimeFEniCSxSpaceWithDirichletBCs
    
Functions
----------------
.. autosummary::
   :toctree: generated/
   
    free_dofs
    interpolate_function_eim
    zero_overlapping_space_time_bcs
"""

from ._dirichletbcs import *
from ._eim import *

_submodules = [
    'utils',
    'problems',
    'norms'
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]