"""
Reduced-Order Model (ROM) classes.

Reduced-Order Models
--------------------
.. autosummary::
    :toctree: generated/
    
     ROM
     Trial2TestROM
     GalerkinROM

Constant Estimators
-------------------

Options
~~~~~~~
.. autosummary::
    :toctree: generated/
     
     StabilityOptions
     ContinuityOptions
     
Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/
    
    ConstantsEstimator
    StabilityEstimator
    ContinuityEstimator
    
Stability Estimators
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/
    
    StabilityExact
    StabilityMinTheta
    
Continuity Estimators
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated/
    
    ContinuityExact
    ContinuityMaxTheta
    
    
Residual Calculators
--------------------
.. autosummary::
    :toctree: generated/
    
    ResidualOptions
    ResidualCalculator
    DirectResidual
    AffineResidual
"""

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

from ._constants import *
from ._residual import *
from ._rom import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]
