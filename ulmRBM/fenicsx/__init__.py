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

__all__ = []

from ._dirichletbcs import *

from . import _dirichletbcs
__all__ += _dirichletbcs.__all__
del _dirichletbcs


from . import utils, problems, norms

__all__ += [
    'utils',
    'problems',
    'norms'
]