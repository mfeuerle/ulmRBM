r""" Collection of some standard problems created with FEniCSx.

Utility Functions
-----------------
.. autosummary::
    :toctree: generated/
    
    assemble_matrix
    assemble_vector
    apply_dirichletbc
    assemble_system
    
Abstract Problems
------------------
.. autosummary::
   :toctree: generated/
   
    weak_problem
    
Elliptic Problems
-----------------
.. autosummary::
    :toctree: generated/
    
    simple_elliptic
    thermal_block
    
Heat Equation Problems
----------------------
.. autosummary::
    :toctree: generated/
    
    simple_timestepping_heat
    simple_heat
    
Wave Equation Problems
----------------------
.. autosummary::
    :toctree: generated/
    
    simple_wave
"""

from ._base import *
from ._elliptic import *
from ._heat import *
from ._wave import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]