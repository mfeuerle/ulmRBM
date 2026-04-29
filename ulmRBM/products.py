"""
Inner product classes for parametric spaces.

Abstract Inner Product Classes
------------------------------
.. autosummary::
    :toctree: generated/
    
     InnerProduct
     InverseInnerProduct
     RestrictedInnerProduct
     
Concrete Inner Product Classes
------------------------------
.. autosummary::
    :toctree: generated/
    
     EuclideanInnerProduct
     MatrixInnerProduct
     OperatorInnerProduct
     
Funtions
------------------------------
.. autosummary::
    :toctree: generated/
    
    orthonormalize     
"""

from __future__ import annotations
from typing import Tuple

__all__ = [
    'InnerProduct',
    'InverseInnerProduct',
    'RestrictedInnerProduct',
    'EuclideanInnerProduct',
    'MatrixInnerProduct',
    'OperatorInnerProduct',
    'orthonormalize',
]

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

from abc import abstractmethod
from collections.abc import Callable

from .core import (
    Mu, Matrix, Vector, NO_MU,
    ParametricLinear, TrivialParametric, 
    wrap_linear, unwrap,
)
from ulmRBM.solver import Solver, IterativeSolver, wrap_solver
from ulmRBM.affine import AffineLinear


# maybe add a flag to prevent warnings about default solvers


class InnerProduct(ParametricLinear[Mu, Matrix]):
    r"""
    Abstract base class for parametric inner products on a vector space.
    
    Represents a possibly parameter-dependent inner product :math:`(\cdot, \cdot)_V: V \times V \to \mathbb{R}` on a vector space :math:`V`. 
    An inner product provides the methods :meth:`inner`, :meth:`norm` and :meth:`riesz` for computing the corresponding quantities, the attribute :attr:`dual` for accessing the dual inner product, and the method :meth:`restrict` to restrict the inner product onto a subsace of :math:`V`.
    
    The Riesz map :math:`R_V: V \to V'` associates each vector :math:`u \in V` with its 
    Riesz representative :math:`R_V u \in V'` in the dual space, defined by:
    
    .. math::
        \langle R_V u, v \rangle_{V' \times V} = (u, v)_V \quad \forall v \in V
    
    **Implementation Requirements:**
    
    Subclasses must implement:
    
    - :meth:`riesz`: Compute the Riesz representative
    
    and initialize :attr:`shape` and :attr:`is_parametric` (typically via the base class constructor).
    
    **Optional Overrides:**
    
    Subclasses may optionally override:
    
    - :meth:`inner`: For more efficient inner product computation
    - :meth:`_restrict`: For efficient restriction to subspaces
    - :meth:`_get_inverse`: To provide an efficient inverse inner product
    
    **Note on Performance:**
    
    InnerProduct instances are generally not designed for online efficiency. While some 
    implementations may be efficient, this is not guaranteed.
    """

    dual: InnerProduct[Mu]
    r"""
    The inner product on the dual space :math:`V'`.
    
    For an inner product :math:`(\cdot, \cdot)_V` with matrix representation :math:`M_V`, i.e. :math:`(u, v)_V = u^T M_V v`, the dual inner product on :math:`V'` is typically given by :math:`(f, g)_{V'} = f^T M_V^{-1} g`.
    
    By default, :attr:`dual` returns the inverse of the inner product. However, this field can be set manually to a different inner product if needed (e.g., for handling Dirichlet boundary conditions).
    """
    
    _dual: InnerProduct[Mu] | None = None
    """Private storage for an explicitly set dual inner product."""
    __inverse: InnerProduct[Mu] | None = None
    """Private cached storage for the inverse inner product."""
    
    
    def __init__(self, shape: tuple[int, int], is_parametric: bool):
        r"""
        Args:
            shape: 
                Shape ``(n, n)`` of the inner product operator, where ``n`` is the 
                dimension of the vector space.
            is_parametric: 
                Whether the inner product depends on parameter :math:`\mu`.
        """
        
        if shape[0] != shape[1]:
            raise ValueError("Inner product must be square.")
        
        self.shape: tuple[int, int] = shape
        r"""Shape ``(n, n)`` of the inner product operator, where ``n`` is the dimension of the vector space :math:`V`."""
        
        self.is_parametric: bool = is_parametric
        """Wheter the inner product depends on the parameter (``True``) or is constant w.r.t. the parameter (``False``)."""
    
    
    def __call__(self, mu: Mu) -> LinearOperator:
        """
        Return a LinearOperator representing the inner product matrix :math:`M_V` at parameter value ``mu``.
        
        Args:
            mu: 
                Parameter value at which to evaluate the inner product operator.
        """
        def matmat(v: Vector) -> Vector:
            return self.riesz(mu, v)
        return LinearOperator(shape=self.shape, matvec=matmat, matmat=matmat, rmatmat=matmat, rmatvec=matmat)
    
    
    @abstractmethod
    def riesz(self, mu: Mu, u: Vector) -> Vector:
        r"""
        Compute the Riesz representative(s) of vector(s) at parameter value ``mu``.
        
        The Riesz map :math:`R_V: V \to V'` satisfies:
        
        .. math::
            \langle R_V u, v \rangle_{V' \times V} = (u, v)_V \quad \forall v \in V
        
        For an inner product with matrix representation :math:`M_V`, this is :math:`R_V u = M_V u`.
        
        Args:
            mu: 
                Parameter value at which to compute the Riesz representative.
            u: 
                Vector(s) of shape ``(n,)`` or ``(n, k)`` for which to compute the 
                Riesz representative(s).
        
        Returns:
            The Riesz representative(s), same shape as ``u``.
        """
        ...
        
    def inner(self, mu: Mu, u: Vector, v: Vector = None) -> np.ndarray | float:
        r"""
        Compute the inner product :math:`(u, v)_V` at parameter value ``mu``.
        
        Args:
            mu: 
                Parameter value at which to evaluate the inner product.
            u: 
                First vector(s) of shape ``(n,)`` or ``(n, k)``.
            v: 
                Second vector(s) of shape ``(n,)`` or ``(n, l)``. If ``None``, defaults to ``u``
                (computing self-inner-products).
        
        Returns
        -------
        numpy.ndarray | float:
            - If both ``u`` and ``v`` are 1D: scalar inner product value.
            - If ``u`` is 1D and ``v`` is 2D (or vice versa): 1D array of inner products.
            - If both are 2D: ``(k,l)`` matrix where entry ``[i, j]`` corresponds to :math:`(u_i, v_j)_V`.
        """
        return u.T @ self.riesz(mu, v if v is not None else u)
    
    def norm(self, mu: Mu, u: Vector) -> np.ndarray | float:
        r"""
        Compute the norm(s) :math:`\|u\|_V = \sqrt{(u, u)_V}` at parameter value ``mu``.
        
        Args:
            mu: 
                Parameter value at which to evaluate the norm.
            u: 
                Vector(s) of shape ``(n,)`` or ``(n, k)`` for which to compute the norm(s).
        
        Returns
        -------
        numpy.ndarray | float:
            - If ``u`` is 1D: scalar norm value.
            - If ``u`` is 2D: 1D array of length ``k``.
        """
        inner = self.inner(mu, u)
        return np.sqrt(inner if u.ndim==1 else np.diag(inner).reshape(-1))
    
    
    def restrict(self, 
                 basis: Matrix | Vector | ParametricLinear[Mu, Matrix | Vector]) -> InnerProduct[Mu]:
        r"""
        Restrict the inner product to a subspace spanned by the given basis.
        
        Given a basis :math:`V_N = [v_1, \ldots, v_N] \in \mathbb{R}^{n \times N}`, this returns
        an inner product on the reduced space :math:`\mathbb{R}^N` such that for coefficient
        vectors :math:`\hat{u}, \hat{v} \in \mathbb{R}^N`:
        
        .. math::
            (\hat{u}, \hat{v})_{V_N} = (V_N \hat{u}, V_N \hat{v})_V
        
        Args:
            basis: 
                Basis of shape ``(n, N)`` defining the subspace.
        
        Returns:
            A new inner product on the reduced space :math:`\mathbb{R}^N`.
        """
        if len(basis.shape) != 2:
            raise ValueError("Basis must be a 2D array of shape (n,k).")
        new = self._restrict(basis)
        if self._dual is not None:
            new.dual = self.dual._inverse._restrict(basis)._inverse
        return new
    
    def _restrict(self, basis: Matrix | Vector | ParametricLinear[Mu, Matrix | Vector]) -> InnerProduct[Mu]:
        """
        Internal method for restricting the inner product to a subspace.
        
        This is the method subclasses should override to provide efficient restriction
        implementations. The default implementation creates a :class:`RestrictedInnerProduct`.
        
        Args:
            basis: 
                Basis of shape ``(n, N)`` defining the subspace.
        
        Returns:
            A new inner product on the reduced space.
        """
        return RestrictedInnerProduct(basis, self)
    

    def _get_inverse(self) -> InnerProduct[Mu]:
        """
        Construct the inverse inner product.
        
        Subclasses may override this to provide more efficient inverse implementations, defaults to  :class:`InverseInnerProduct`.
        
        Returns:
            An inner product representing :math:`M_V^{-1}`.
        """
        return InverseInnerProduct(self)
    
    @property
    def _inverse(self) -> InnerProduct[Mu]:
        if self.__inverse is None:
            self.__inverse = self._get_inverse()
            self.__inverse.__inverse = self
        return self.__inverse
    
    
    @property
    def dual(self) -> InnerProduct[Mu]:
        if self._dual is None:
            return self._inverse
        else:
            return self._dual
        
    @dual.setter
    def dual(self, dual: InnerProduct[Mu]):
        if self._dual is not None or dual._dual is not None:
            raise ValueError("Cannot set dual: one of the inner products has already a dual product linked.")
        self._dual = dual
        dual._dual = self


class InverseInnerProduct(InnerProduct[Mu]):
    r"""
    Inner product defined by the inverse of another inner product.
    
    For an given inner product :math:`(\cdot, \cdot)_V` with matrix representation :math:`M_V`, i.e. :math:`(u, v)_V = u^T M_V v`, this represents the inner product :math:`(\cdot, \cdot)_{V^{-1}}` defined by:
    
    .. math::
        (u, v)_{V^{-1}} = u^T M_V^{-1} v
    """
    
    default_solver = IterativeSolver(spd=True)
    """Default solver used when no solver is explicitly specified."""
    
    def __init__(self, 
                 ip: InnerProduct[Mu],
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None):
        """
        Args:
            ip:
                The inner product to invert.
            solver: 
                Solver for computing :math:`M_V^{-1} v`. Can be a :class:`Solver` instance,
                a callable with signature ``(A, b, x0=None) -> x``, or ``None`` to use the default solver.
                If ``None``, a warning is issued when the solver is used accessed.
        """
        
        super().__init__(ip.shape, ip.is_parametric)
        self._ip = ip
        """The original inner product whose inverse is represented."""
        self.solver: Solver = solver
        r"""
        Solver used for computing the inverse.
        
        If no solver was specified manually, the :attr:`default_solver`
        is used and a warning is issued.
        """
        
    def riesz(self, mu: Mu, u: Vector) -> Vector:
        return self.solver(self._ip(mu), u)
    
    def _get_inverse(self) -> InnerProduct[Mu]:
        return self._ip
        
    @property
    def solver(self) -> Solver:
        if self._solver is None:
            from warnings import warn
            warn("No solver specified; using default solver.", UserWarning)
            return self.default_solver
        else:
            return self._solver
    
    @solver.setter
    def solver(self, solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None):
        self._solver = wrap_solver(solver) if solver is not None else None
        
        

class RestrictedInnerProduct(InnerProduct[Mu]):
    r"""
    Inner product restricted to a subspace.
    
    Given an inner product :math:`(\cdot, \cdot)_V` and a basis 
    :math:`V_N = \{v_1, \ldots, v_N\} \subset V`, this represents
    the inner product on :math:`\mathbb{R}^N` defined by:
    
    .. math::
        (\hat{u}, \hat{v})_{V_N} = (V_N \hat{u}, V_N \hat{v})_V = \hat{u}^T (V_N^T M_V V_N) \hat{v}
    
    where :math:`\hat{u}, \hat{v} \in \mathbb{R}^N` are coefficient vectors.
    """
    
    def __init__(self,
                 basis: Matrix | Vector | ParametricLinear[Mu, Matrix | Vector],
                 ip: InnerProduct[Mu]) -> RestrictedInnerProduct[Mu]:
        """
        Args:
            basis: 
                Basis matrix of shape ``(n, N)`` defining the subspace.
            ip: 
                The inner product to restrict to the subspace.
        """
    
        basis = wrap_linear(basis)
        if isinstance(basis, AffineLinear):
            basis = basis.compress()
        n = basis.shape[1]
        is_parametric = not isinstance(basis, TrivialParametric) or ip.is_parametric
        super().__init__((n, n), is_parametric)
        
        self._basis: ParametricLinear[Mu, Matrix | Vector] = basis
        """The basis matrix :math:`V_N` defining the subspace."""
        self._ip: InnerProduct[Mu] = ip
        """The original inner product on the full space."""
        
    def riesz(self, mu: Mu, u: Vector) -> Vector:
        basis = self._basis(mu)
        return basis.T @ self._ip.riesz(mu, basis @ u)
    
    def inner(self, mu: Mu, u: Vector, v: Vector = None) -> np.ndarray | float:
        basis = self._basis(mu)
        if v is not None:
            return self._ip.inner(mu, basis @ u, basis @ v)
        return self._ip.inner(mu, basis @ u)
    
    def _restrict(self, basis: Matrix | Vector | ParametricLinear[Mu, Matrix | Vector]) -> InnerProduct[Mu]:
        basis_new = unwrap(basis)
        basis_old = unwrap(self._basis)
        
        if isinstance(basis_new, Matrix | Vector | AffineLinear) \
            and isinstance(basis_old, Matrix | Vector | AffineLinear):
            return RestrictedInnerProduct(basis_old @ basis_new, self._ip)
        else:
            return super()._restrict(basis_new)
        
        
class EuclideanInnerProduct(InnerProduct[Mu]):
    r"""
    Standard Euclidean inner product.
    
    Represents the inner product :math:`(u, v) = u^T v`.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: 
                Dimension ``n`` of the vector space.
        """
        super().__init__((dim, dim), False)
        
    def __call__(self, mu: Mu) -> sp.sparse.csr_array:
        return sp.sparse.eye_array(self.shape[0], format='csr')
        
    def riesz(self, mu: Mu, u: Vector) -> Vector:
        return u
    
    def _get_inverse(self) -> EuclideanInnerProduct[Mu]:
        return self
    
    def _restrict(self, basis: Matrix | Vector | ParametricLinear[Mu, Matrix | Vector]) -> InnerProduct[Mu]:
        basis = unwrap(basis)
        if isinstance(basis, Matrix | Vector | AffineLinear):
            return MatrixInnerProduct(basis.T @ basis)
        else:
            return super()._restrict(basis)
        

class MatrixInnerProduct(InnerProduct[Mu]):
    r"""
    Inner product defined by a matrix.
    
    Represents the inner product :math:`(u, v)_V = u^T M v` where :math:`M` is a
    symmetric positive definite matrix. The matrix :math:`M` may be parameter-dependent.
    """
    
    def __init__(self, 
                 M: Matrix | ParametricLinear[Mu, Matrix], 
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None):
        """
        Args:
            M:  
                The matrix defining the inner product.
            solver:
                Solver for computing :math:`M^{-1} v`, used for the dual inner product. If `None`, the default solver :attr:`InverseInnerProduct.default_solver` is used.
        """
        M = wrap_linear(M)
        if isinstance(M, AffineLinear):
            M = M.compress()
        super().__init__(M.shape, not isinstance(M, TrivialParametric))
        self._M: ParametricLinear[Mu, Matrix] = M
        """The matrix or linear operator defining the inner product."""
        if solver is not None:
            self._inverse.solver = solver
        
    def __call__(self, mu: Mu) -> LinearOperator:
        return self._M(mu)
    
    def riesz(self, mu: Mu, u: Vector) -> Vector:
        return self._M(mu) @ u
    
    def _restrict(self, basis: Matrix | Vector | ParametricLinear[Mu, Matrix | Vector]) -> InnerProduct[Mu]:
        basis = unwrap(basis)
        M = unwrap(self._M)
        
        if isinstance(basis, Matrix | Vector | AffineLinear) \
            and isinstance(M, Matrix | AffineLinear):
            return MatrixInnerProduct(basis.T @ M @ basis)
        else:
            return super()._restrict(basis)


class OperatorInnerProduct(RestrictedInnerProduct[Mu]):
    r"""
    Inner product defined by :math:`(u, v)_U := (Bu, Bv)_V`.
    
    Represents the inner product :math:`(u, v)_U = (Bu, Bv)_V` where :math:`B` is an
    operator and :math:`(\cdot, \cdot)_V` is an inner product on the range of :math:`B`.
    
    This is implemented as a :class:`RestrictedInnerProduct` with basis :math:`B`, but the inverse inner product is computed via :math:`(u, v)_{U^{-1}} = u^T B^{-T} V^{-1} B^{-1} v` instead of :math:`u^T (B^TVB)^{-1} v`. This is usefull, if the inner product :math:`V` is expensive to compute, but :math:`V^{-1}` is cheap, e.g. if :math:`V` is a :class:`InverseInnerProduct`.
    """
    
    def __init__(self, 
                 B: Matrix | Vector | ParametricLinear[Mu, Matrix | Vector], 
                 V: InnerProduct[Mu],
                 solver: Solver | Callable[[Matrix, Vector, Vector | None], Vector] | None = None):
        """
        Args:
            B: 
                The operator matrix.
            V: 
                The inner product on the range of :math:`B`.
            solver:
                Solver used on :math:`B` and :math:`B^T` for :math:`B^{-1}` and :math:`B^{-T}`. If ``None``, the default solver :attr:`_InverseOperatorInnerProduct.default_solver` is used.
        """
        if B.shape[0] != B.shape[1]:
            raise ValueError("B can not be invertible as it is not square.")
        super().__init__(B, V)
        if solver is not None:
            self._inverse.solver = solver
        
    def _get_inverse(self):
        return _InverseOperatorInnerProduct(self)

class _InverseOperatorInnerProduct(InverseInnerProduct[Mu]):
    
    default_solver = IterativeSolver()
    """Default solver for solving systems with :math:`B` and :math:`B^T`."""
    
    _ip: OperatorInnerProduct[Mu]
    
    def __init__(self, 
             ip: OperatorInnerProduct[Mu], 
             solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None):
        """
        Args:
            ip: The :class:`OperatorInnerProduct` to invert.
            solver: Solver for systems with :math:`B` and :math:`B^T`. If `None`, uses :attr:`default_solver`.
        """

        super().__init__(ip, solver)
        
    def riesz(self, mu: Mu, u: Vector) -> Vector:
        B = self._ip._basis(mu)
        Binv_u = self.solver(B.T, u)
        Vinv = self._ip._ip._inverse
        Vinv_Binv_u = Vinv.riesz(mu, Binv_u)
        return self.solver(B, Vinv_Binv_u)

    def inner(self, mu: Mu, u: Vector, v: Vector = None) -> np.ndarray | float:
        B = self._ip._basis(mu)
        Vinv = self._ip._ip._inverse
        Binv_u = self.solver(B.T, u)
        if v is not None:
            Binv_v = self.solver(B.T, v)
            return Vinv.inner(mu, Binv_u, Binv_v)
        return Vinv.inner(mu, Binv_u)
    
    
def orthonormalize(basis: Vector, ip: InnerProduct[Mu] | Matrix, full: bool = False) -> Tuple[Vector, np.ndarray]:
    r"""
    Orthonormalize a basis with respect to a given inner product.
    
    Args:
        basis:
            Basis matrix :math:`(n, N)` to orthonormalize.
        ip:
            Inner product or linear operator representation of an inner product with respect to which to orthonormalize. Must be parameter-independent.
        full:
            If ``True``, more stable but expensive.
            If ``False``, faster, but might accumulate errors.
            
    Returns:
        - Orthonormalized basis matrix :math:`(n, N)`.
        - Transformation matrix :math:`(N, N)` such that :math:`\text{basis}_\text{orth} = \text{basis} \@ Q`.
    """

    if not isinstance(ip, InnerProduct):
        ip = MatrixInnerProduct(ip)
        
    if ip.is_parametric:
        raise ValueError("Inner product U must be parameter independent.")
            
    inner = ip.inner(NO_MU, basis)
    # othogonalize
    Q = np.linalg.svd(inner)[0]
    basis_orth = basis @ Q
    # normalize
    if full:
        norm = ip.norm(NO_MU, basis_orth)
    else:
        norm = np.sqrt(np.abs(np.diag(Q.T @ inner @ Q))) # fast, but not stable
        for i in range(norm.shape[0]):
            if norm[i] < 1e-5:  # re-compute due to potentiall instability
                norm[i] = ip.norm(NO_MU, basis_orth[:,i])
    basis_orth = basis_orth / norm
    Q = Q / norm
    return basis_orth, Q