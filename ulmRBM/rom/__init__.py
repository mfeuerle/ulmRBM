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

__all__ = []

from ._constants import *
from ._residual import *
from ._rom import *

from . import _constants, _residual, _rom
__all__ += _constants.__all__ + _residual.__all__ + _rom.__all__
del _constants, _residual, _rom
