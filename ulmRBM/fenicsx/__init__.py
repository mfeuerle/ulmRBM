"""
Adding some specific FEniCSx related functionality.

Submodules
----------
.. autosummary::
   :toctree: generated/
   
   utils
   problems

Classes
-------
.. autosummary::
   :toctree: generated/
   
    AffineDirichletBC
    
Functions
----------------
.. autosummary::
   :toctree: generated/
   
    free_dofs
"""

__all__ = [
    'utils',
    'AffineDirichletBC',
    'free_dofs',
]

from ._dirichletbcs import AffineDirichletBC, free_dofs
from . import utils
from . import problems

