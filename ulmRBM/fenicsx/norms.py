r""" Collection of some standard norms created with FEniCSx.

Norms
-----------------
.. autosummary::
    :toctree: generated/
    
    l2
    h10
    h1
"""

__all__ = [
    'l2',
    'h10',
    'h1',
    ]

from collections.abc import Callable

from scipy.linalg import inv

import ufl

from ulmRBM.core import Matrix, Vector, NO_MU
from ulmRBM.affine import affine_kron
from ulmRBM.fenicsx import FEniCSxSpaceWithDirichletBCs, SpaceTimeKey, SpaceTimeFEniCSxSpaceWithDirichletBCs
from ulmRBM.fenicsx.utils import assemble_matrix
from ulmRBM.solver import Solver, DirectSolver
from ulmRBM.products import MatrixInnerProduct

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME


def l2(U: FEniCSxSpaceWithDirichletBCs,
       solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None, 
       bcs: bool = True) -> MatrixInnerProduct:
    r""":math:`L^2` inner product matrix.

    .. math::
        (u,v)_{L^2(\Omega)} = \int_\Omega u\,v\,dx.

    Args:
        U :
            Function space including Dirichlet boundary conditions.
        solver :
            Optional solver used by ``MatrixInnerProduct`` for dual operations.
        bcs :
            If ``True``, the inner product is restricted to the free dofs ``U.dofs``.
    """
    
    u = ufl.TrialFunction(U.space)
    v = ufl.TestFunction(U.space)
    product = assemble_matrix(u*v*ufl.dx)
    if bcs: product = product[U.dofs,:][:,U.dofs]
    return MatrixInnerProduct(product, solver)

def h10(U: FEniCSxSpaceWithDirichletBCs,
        solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None,
        bcs: bool = True) -> MatrixInnerProduct:
    r""":math:`H^1_0` inner product matrix.

    .. math::
        (u,v)_{H^1_0(\Omega)} = \int_\Omega \nabla u \cdot \nabla v\,dx.

    Args:
        U :
            Function space including Dirichlet boundary conditions.
        solver :
            Optional solver used by ``MatrixInnerProduct`` for dual operations.
        bcs :
            If ``True``, the inner product is restricted to the free dofs ``U.dofs``.
    """
    
    if U.space.ufl_element().basix_element.degree < 1:
        raise ValueError("H1 norm is not defined for elements of degree < 1.")
    
    u = ufl.TrialFunction(U.space)
    v = ufl.TestFunction(U.space)
    product = assemble_matrix(ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
    if bcs: product = product[U.dofs,:][:,U.dofs]
    return MatrixInnerProduct(product, solver)

def h1(U: FEniCSxSpaceWithDirichletBCs,
       solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None, 
       bcs: bool = True) -> MatrixInnerProduct:
    r""":math:`H^1` inner product matrix.

    .. math::
        (u,v)_{H^1(\Omega)} = \int_\Omega u\,v\,dx + \int_\Omega \nabla u \cdot \nabla v\,dx.

    Args:
        U :
            Function space including Dirichlet boundary conditions.
        solver :
            Optional solver used by ``MatrixInnerProduct`` for dual operations.
        bcs :
            If ``True``, the inner product is restricted to the free dofs ``U.dofs``.
    """
    
    if U.space.ufl_element().basix_element.degree < 1:
        raise ValueError("H1 norm is not defined for elements of degree < 1.")
    
    u = ufl.TrialFunction(U.space)
    v = ufl.TestFunction(U.space)
    product = assemble_matrix(u*v*ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
    if bcs: product = product[U.dofs,:][:,U.dofs]
    return MatrixInnerProduct(product, solver)

def space_time(U, norm: list[dict[SpaceTimeKey, str]], solver=None, bcs=True):
    
    M = []
    
    for n in norm:
        Mn = {}
        for KEY in SpaceTimeKey:
            if n[KEY].lower().startswith('l2'):
                Mn[KEY] = l2(FEniCSxSpaceWithDirichletBCs(U.space[KEY], []), bcs=False)._M(NO_MU)
            elif n[KEY].lower().startswith('h10'):
                Mn[KEY] = h10(FEniCSxSpaceWithDirichletBCs(U.space[KEY], []), bcs=False)._M(NO_MU)
            elif n[KEY].lower().startswith('h1'):
                Mn[KEY] = h1(FEniCSxSpaceWithDirichletBCs(U.space[KEY], []), bcs=False)._M(NO_MU)
            else:
                raise ValueError(f"Unknown norm {n[KEY]} for {KEY}.")
        M.append(Mn)
        
    if bcs:
        M = [{KEY: Mi[KEY][:, U.dofs[KEY]][U.dofs[KEY], :] for KEY in SpaceTimeKey} for Mi in M]
        
    for i,n in enumerate(norm):
        for KEY in SpaceTimeKey:
            if n[KEY].lower().endswith('dual'):
                M[i][KEY] = inv(M[i][KEY].toarray(), assume_a='pos')
        
    M = sum([affine_kron(Mi[SPACE], Mi[TIME]) for Mi in M])(NO_MU)
    return MatrixInnerProduct(M, solver)