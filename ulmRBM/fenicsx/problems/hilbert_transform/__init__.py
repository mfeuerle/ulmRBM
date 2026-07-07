r"""Collection of functions to assemble matrices involving the modified Hilbert transform.

For a definition of the modified Hilbert transform, see e.g.

    Richard Löscher, Olaf Steinbach and Marco Zank, 2022. "Numerical Results for an Unconditionally Stable Space-Time Finite Element Method for the Wave Equation". Domain Decomposition Methods in Science and Engineering XXVI, p. 625-632. Springer International Publishing.
    
See the above reference for some important properties of the modified Hilbert transform :math:`\mathcal{H}_T`. Jus to mention some:

    - it is an isomorphism between :math:`H^1_{0,}(I)` and :math:`H^1_{,0}(I)`, i.e. :math:`H^1_{,0}(I) = \mathcal{H}_TH^1_{0,}(I)`;
    - the inverse and adjoint coincide, i.e. :math:`\mathcal{H}_T^{-1} = \mathcal{H}_T^*`;
    - it holds :math:`(\partial_t u, \partial_t \mathcal{H}_T v)_{L^2(I)} = -(\mathcal{H}_T \partial_t u, \partial_t v)_{L^2(I)}` for all :math:`u,v \in H^1_{0,}(I)`, thus
    - the Hilbert transform is also isometric between :math:`H^1_{0,}(I)` and :math:`H^1_{,0}(I)` as :math:`(\partial_t u, \partial_t v)_{L^2(I)} = (\mathcal{H}_T \partial_t u, \mathcal{H}_T \partial_t v)_{L^2(I)}`.

.. note::
    The modified Hilbert transform is only defined for 1D domains :math:`I \subset \mathbb{R}` and piecewise linear finite elements.

.. note::
    This module needs a working Matlab installation accessable in terminal / bash via the command ``matlab``, as the original implementation is in Matlab and was not ported to Python.
"""


from ._python_port import *

_submodules = [
]

import importlib as _importlib
for _submodule in _submodules:
    _importlib.import_module(f".{_submodule}", __package__)

__all__ = [s for s in dir() if not s.startswith('_')]