"""
Classes to estimate inf-sup, coercicity and continuity constants.
"""

from ._interface import *
from ._exact import *
from ._scm import *
from ._theta import *
from ._algorithms import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]
