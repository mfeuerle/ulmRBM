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

import subprocess
import pathlib

import numpy as np
from scipy.sparse import csr_array

import basix
from dolfinx import fem

__all__ = [
    'mass_matrix', 
    'stiffness_matrix'
]


def _check_space(U: fem.FunctionSpace):
    element = U.ufl_element().basix_element
    
    if element.degree != 1 or element.family != basix.ElementFamily.P or element.discontinuous:
        raise ValueError("ht_l2 is only defined for continuous piecewise linear elements.")
    
    if U.mesh.geometry.dim != 1:
        raise ValueError("ht_l2 is only defined for 1D domains.")
    

def mass_matrix(U: fem.FunctionSpace) -> csr_array:
    r"""Mass matrix :math:`(u,\mathcal{H}_Tv)_{L^2(\Omega)}` including a modified Hilbert transform.
        
    Args:
        U :
            Function space.
    """
    
    _check_space(U)
    
    n = U.dofmap.index_map.size_global
    t = "[{}]".format(",".join(f"{str(x)}" for x in U.mesh.geometry.x[:,0]))
    
    folder_matlab = str(pathlib.Path(__file__).parent.resolve() / "matlab")
    subprocess.run([f"cd {folder_matlab};\nmatlab -nodisplay -r \"eval('python_assemble_massmatrix_hilbert({t})'); exit\""], shell=True, check=True, stdout=subprocess.DEVNULL)
    M = np.loadtxt(f"{folder_matlab}/.masma.txt")
    subprocess.run([f"rm {folder_matlab}/.masma.txt"], shell=True, check=True, stdout=subprocess.DEVNULL)
    return csr_array((M[:, 2], (M[:,0]-1,M[:,1]-1)), shape=(n,n))


def stiffness_matrix(U: fem.FunctionSpace) -> csr_array:
    r"""Stiffness matrix :math:`(\partial_t u, \partial_t \mathcal{H}_T v)_{L^2(\Omega)}` including a modified Hilbert transform.
    
    This is equal to :math:`-(\mathcal{H}_T \partial_t u, \partial_t v)_{L^2(\Omega)}`.
        
    Args:
        U :
            Function space.
    """
    
    _check_space(U)
    
    n = U.dofmap.index_map.size_global
    t = "[{}]".format(",".join(f"{str(x)}" for x in U.mesh.geometry.x[:,0]))
    
    folder_matlab = str(pathlib.Path(__file__).parent.resolve() / "matlab")
    subprocess.run([f"cd {folder_matlab};\nmatlab -nodisplay -r \"eval('python_assemble_stiffmatrix_hilbert({t})'); exit\""], shell=True, check=True, stdout=subprocess.DEVNULL)
    S = np.loadtxt(f"{folder_matlab}/.stima.txt")
    subprocess.run([f"rm {folder_matlab}/.stima.txt"], shell=True, check=True, stdout=subprocess.DEVNULL)
    return -csr_array((S[:, 2], (S[:,0]-1,S[:,1]-1)), shape=(n,n))