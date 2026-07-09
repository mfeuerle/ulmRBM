r""" Collection of some standard problems created with FEniCSx.
    
Elliptic Problems
-----------------
.. autosummary::
    :toctree: generated/
    
    weak_problem
    thermal_block
    simple_elliptic
    thermal_block
    
Heat Equation Problems
----------------------
.. autosummary::
    :toctree: generated/
    
    heat_equation
    heat_equation_timestepping
    simple_heat
    simple_heat_timestepping
    
Wave Equation Problems
----------------------
.. autosummary::
    :toctree: generated/
    
    weak_problem
    wave_equation_structured
    wave_equation_hilbert
    simple_wave    
    simple_wave_structured
    simple_wave_hilbert
    
Submodules
-----------------
.. autosummary::
    :toctree: generated/
    
    hilbert_transform
"""

from ._elliptic import *
from ._heat import *
from ._wave import *

_submodules = [
    'hilbert_transform',
]

# for backwards compatibility, we import
from ulmRBM.fenicsx import apply_dirichletbc, assemble_system
from ulmRBM.fenicsx.utils import assemble_matrix, assemble_vector

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]