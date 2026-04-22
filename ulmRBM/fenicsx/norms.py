r""" Collection of some standard norms created with FEniCSx.

Norms
-----------------
.. autosummary::
    :toctree: generated/
    
    l2_norm
    h10_norm
    h1_norm
"""

__all__ = [
    'l2',
    'h10',
    'h1',
    ]

from collections.abc import Callable

import ufl

from ulmRBM.core import Matrix, Vector, NO_MU
from ulmRBM.fenicsx import FEniCSxSpaceWithDirichletBCs
from ulmRBM.fenicsx.problems import assemble_matrix
from ulmRBM.solver import Solver
from ulmRBM.products import MatrixInnerProduct


def l2(U: FEniCSxSpaceWithDirichletBCs, 
            bcs: bool = True,
            solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None) -> MatrixInnerProduct:
    r""":math:`L^2` inner product matrix.

    .. math::
        (u,v)_{L^2(\Omega)} = \int_\Omega u\,v\,dx.

    Args:
        U :
            Function space including Dirichlet boundary conditions.
        bcs :
            If ``True``, the inner product is restricted to the free dofs ``U.dofs``.
        solver :
            Optional solver used by ``MatrixInnerProduct`` for dual operations.
    """
    
    u = ufl.TrialFunction(U.space)
    v = ufl.TestFunction(U.space)
    product = assemble_matrix(u*v*ufl.dx)
    if bcs: product = product[U.dofs,:][:,U.dofs]
    return MatrixInnerProduct(product, solver)

def h10(U: FEniCSxSpaceWithDirichletBCs, 
             bcs: bool = True,
             solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None) -> MatrixInnerProduct:
    r""":math:`H^1_0` inner product matrix.

    .. math::
        (u,v)_{H^1_0(\Omega)} = \int_\Omega \nabla u \cdot \nabla v\,dx.

    Args:
        U :
            Function space including Dirichlet boundary conditions.
        bcs :
            If ``True``, the inner product is restricted to the free dofs ``U.dofs``.
        solver :
            Optional solver used by ``MatrixInnerProduct`` for dual operations.
    """
    
    if U.space.ufl_element().basix_element.degree < 1:
        raise ValueError("H1 norm is not defined for elements of degree < 1.")
    
    u = ufl.TrialFunction(U.space)
    v = ufl.TestFunction(U.space)
    product = assemble_matrix(ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
    if bcs: product = product[U.dofs,:][:,U.dofs]
    return MatrixInnerProduct(product, solver)

def h1(U: FEniCSxSpaceWithDirichletBCs, 
            bcs: bool = True,
            solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None) -> MatrixInnerProduct:
    r""":math:`H^1` inner product matrix.

    .. math::
        (u,v)_{H^1(\Omega)} = \int_\Omega u\,v\,dx + \int_\Omega \nabla u \cdot \nabla v\,dx.

    Args:
        U :
            Function space including Dirichlet boundary conditions.
        bcs :
            If ``True``, the inner product is restricted to the free dofs ``U.dofs``.
        solver :
            Optional solver used by ``MatrixInnerProduct`` for dual operations.
    """
    
    return MatrixInnerProduct(l2(U,bcs)(NO_MU) + h10(U,bcs)(NO_MU), solver)