r""" Collection of some standard norms created with FEniCSx.

Norms
-----------------
.. autosummary::
    :toctree: generated/
    
    l2
    h10
    h1
    space_time
"""

__all__ = [
    'l2',
    'h10',
    'h1',
    'space_time'
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


def space_time(U: SpaceTimeFEniCSxSpaceWithDirichletBCs, norm: list[dict[SpaceTimeKey, str]], solver=None, bcs=True):
    r"""Space-time inner product matrix.
    
    Let :math:`V_i^{SPACE}` and :math:`V_i^{TIME}`, :math:`i=1,\ldots,N`, be a collection of space and time inner product matrices, respectively. Then the space-time inner product matrix is defined
    
    .. math::
        V_{SPACE \otimes TIME} = \sum_{i=1}^N V_i^{SPACE} \otimes V_i^{TIME}.
        
    Args:
        U :
            Space-time function space
        norm :
            List of dictionaries specifying the space and time inner product matrices, with ``norm[i][KEY]`` being one of the norms provided in this module, such as``'l2'``, ``'h10'``, or ``'h1'``, and ``KEY`` being a `SpaceTimeKey`. In addition, the suffix ``'dual'`` can be used to indicate that the dual inner product matrix should be used, e.g., ``'h10 dual'``.
        solver :
            Optional solver used by ``MatrixInnerProduct`` for dual inner product of :math:`V_{SPACE \otimes TIME}`.
        bcs :
            If ``True``, the inner product is restricted to the free dofs of ``U``.
    """
    
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