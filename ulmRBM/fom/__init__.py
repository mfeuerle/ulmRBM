"""
Full-Order Model (FOM) classes.

Operators
----------
.. autosummary::
    :toctree: generated/
    
    ParametricOperator
    ParametricGalerkinOperator
    
Standard Models
----------------
.. autosummary::
    :toctree: generated/
    
    Model
    FOM
    GalerkinFOM
    
Time-Stepping Models
--------------------
.. autosummary::
    :toctree: generated/
    
    TimeSteppingSolution
    StationaryTimeSteppingGalerkinFOM
    explicit_euler
    implicit_euler
    crank_nicolson
"""

from ._operators import *
from ._basic import *
from ._timestepping import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]

