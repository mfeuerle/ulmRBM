"""
Reduced-Order Model (ROM) classes.

Reduced-Order Models
--------------------
.. autosummary::
    :toctree: generated/
    
    ROM
    GalerkinROM

Constant Estimators
-------------------

Interface
~~~~~~~~~
.. autosummary::
    :toctree: generated/
     
    ConstantEstimator
    StabilityEstimator
    ContinuityEstimator
    EfficientConstantEstimator
    
Algorithms
~~~~~~~~~~
Setting up any `EfficientConstantEstimator`

.. autosummary::
    :toctree: generated/
    
    greedy_constant_estimator

     
Exact Estimators
~~~~~~~~~~~~~~~~
In general not online-efficient but usefull if the exact stability or continuity constant of the `FOM` can be evaluated fast via custom functions (e.g. if the constants are known analytically).

.. autosummary::
    :toctree: generated/
    
    ExactStability
    ExactContinuity
    
Min/Max-Theta Estimators
~~~~~~~~~~~~~~~~~~~~~~~~
Online-efficient estimatators, under additional assumptions on the system matrices.

.. autosummary::
    :toctree: generated/
    
    ThetaStability
    ThetaContinuity
    
Successive Constraint Method (SCM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Online-efficient estimatators.

.. autosummary::
    :toctree: generated/
    
    SCMStability
    SCMContinuity
"""

from ._constants import *
from ._rom import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]
