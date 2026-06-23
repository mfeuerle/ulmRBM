"""
Adding some specific FEniCSx related functionality.

Submodules
----------
.. autosummary::
   :toctree: generated/
   
   utils
   problems
   norms
   
Basic Problems
--------------
.. autosummary::
   :toctree: generated/
    
    AffineDirichletBC
    FEniCSxSpaceWithDirichletBCs
    free_dofs
    apply_dirichletbc
    assemble_system
    
Space-Time Problems
-------------------
.. autosummary::
    :toctree: generated/
    
    interpolate_space_time
    SpaceTimeKey
    SpaceTimeAffineDirichletBC
    SpaceTimeFEniCSxSpaceWithDirichletBCs
    zero_overlapping_space_time_bcs
    apply_dirichletbc_space_time
   
Empirical Interpolation Method (EIM)
------------------------------------
.. autosummary::
   :toctree: generated/

    interpolate_function_eim
"""

from ._basic import *
from ._space_time import *
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