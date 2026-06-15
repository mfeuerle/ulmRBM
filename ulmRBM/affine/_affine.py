"""
Affine decomposition classes for parametric problems.
"""


from __future__ import annotations


__all__ = [
    'AffineObject',
    'AffineLinear',
    'AffineFunction',
    'wrap_affinelinear',
    'ScalarComponentList',
    'multiply_theta',
]

from numbers import Number
from collections.abc import Iterable, MutableSequence, Callable, Sequence
from enum import IntEnum
import numpy as np

from ulmRBM.core import Mu, Data, ParametricObject, ParametricLinear, TrivialParametric, wrap_scalar, unwrap


class _ScaledScalar(ParametricObject[Mu, float]):
    def __init__(self, scale: float, scalar: ParametricObject[Mu, float]):
        if isinstance(scalar, _ScaledScalar):
            scale = scale * scalar.scale
            scalar = scalar.scalar
        self.scale = scale
        self.scalar = scalar
        
    def __call__(self, mu: Mu) -> float:
        return self.scale * self.scalar(mu)
    
def multiply_theta(scalar1: ParametricObject[Mu, float] | TrivialParametric[Mu, float] | _ScaledScalar[Mu] | float,
                     scalar2: ParametricObject[Mu, float] | TrivialParametric[Mu, float] | _ScaledScalar[Mu] | float) -> ParametricObject[Mu, float] | TrivialParametric[Mu, float] | _ScaledScalar[Mu]:
    r"""Multiply two """
    
    scalar1 = unwrap(scalar1)
    scalar2 = unwrap(scalar2)
    
    if scalar1 == 0.0 or scalar2 == 0.0:
        return wrap_scalar(0.0)
    if scalar1 == 1.0:
        return wrap_scalar(scalar2)
    if scalar2 == 1.0:
        return wrap_scalar(scalar1)
    
    if isinstance(scalar1, Number):
        if isinstance(scalar2, Number):
            return wrap_scalar(scalar1 * scalar2)
        else:
            return _ScaledScalar(scalar1, scalar2)
    elif isinstance(scalar2, Number):
        return _ScaledScalar(scalar2, scalar1)
    else:
        return lambda mu, s1=scalar1, s2=scalar2: s1(mu) * s2(mu)

class _ConstructNew(IntEnum):
    SAME = 0
    MATMUL = 1
    MUL = 2
    APPLY2DATA = 3


class AffineObject(ParametricObject[Mu, Data], MutableSequence):
    r"""
    An affine decomposition of a parametric object.

    An affine object, consists of a list of parameter dependent scalar functions :math:`\theta_1(\mu),\dots,\theta_Q(\mu)` and parameter independent data terms :math:`\text{data}_1,\dots,\text{data}_Q` and is given
    by an affine decomposition of the form
    
    .. math::
        \text{data}(\mu) = \sum_{q=1}^{Q} \theta_q(\mu) \cdot \text{data}_q,
        
    where :math:`\mu` is a parameter of type ``Mu``, and :math:`\text{data}(\mu)` is an object of type ``Data``.
    """
    r"""
    List of parameter dependent scalar functions and parameter independent data terms.
    
    An affine object, consisting of a list of parameter dependent scalar functions :math:`\theta_1(\mu),\dots,\theta_Q(\mu)` and parameter independent data terms :math:`\text{data}_1,\dots,\text{data}_Q`.
    """
    
    __array_priority__ = 100.0
    """Priority for NumPy to defer to our __r*__ methods."""
    
    __sparse_priority__ = 100.0
    """Priority for scipy.sparse to defer to our ``__r*__`` methods (via patch in :mod:`ulmRBM`)."""
    
    __linop_priority__ = 100.0
    """Priority for :mod:`scipy.sparse.linalg.LinearOperator` to defer to our ``__r*__`` methods (via patch in :mod:`ulmRBM`)."""
    
    __kron_priority__ = 100.0
    """Priority over the https://github.com/mfeuerle/kron module to defer to our ``__r*__`` methods."""
    
    @property
    def is_parametric(self) -> bool:
        """Whether the affine list contains any parameter-dependent terms."""
        return any(not isinstance(theta, TrivialParametric) for theta in self.theta)
    
    def __init__(self, theta: list[ParametricObject[Mu, float]] | AffineObject[Mu, Data] | Iterable[tuple[ParametricObject[Mu, float], any]] = [], data: list = None):
        r"""
        Args:
            theta :
                List of parameter-dependent coefficient functions. Each function should accept a 
                parameter value and return a scalar coefficient. Alternatively, an `AffineObject` can be provided, in which case its ``theta`` and ``data`` attributes are used, or a iterable of ``(theta, data)`` tuples, e.g. ``[(theta1,data1), (theta2,data2)]``.
            data :
                List of parameter-independent data terms, or empty if ``theta`` is an `AffineObject` or iterable.
        """
        
        if data is None:
            if isinstance(theta, AffineObject):
                data  = theta.data
                theta = theta.theta
            else:
                data = [t[1] for t in theta]
                theta = [t[0] for t in theta]
                
        if len(theta) != len(data):
            raise ValueError("Length of theta and data must be the same.")
        
        self.theta: list[ParametricObject[Mu, float]] = [wrap_scalar(t) for t in theta]
        """List of parameter-dependent coefficient functions."""
        self.data: list = [d for d in data]
        """List of parameter-independent data terms."""
        
    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} affine terms>"
    
    def __call__(self, mu: Mu) -> Data:
        r"""
        Evaluate the affine decomposition at a parameter value.
        
        Computes :math:`\sum_{q=1}^{Q} \theta_q(\mu) \cdot \text{data}_q`.
        
        Args:
            mu :
                Single parameter value at which to evaluate the coefficient functions.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> ad(3.0)  # 3*1 + 9*2 = 21
        21.0
        """
        val = self.theta[0](mu) * self.data[0]
        for theta_q, data_q in zip(self.theta[1:], self.data[1:]):
            val += theta_q(mu) * data_q
        return val
        
    def _construct_new(self, theta, data, type: _ConstructNew = _ConstructNew.SAME) -> AffineObject[Mu,Data]:
        """Fine controll construction of new AffineObject objects for operations."""
        return self.__class__(theta, data)
        
        
    def __getitem__(self, key: int | slice) -> tuple[ParametricObject[Mu, float], any] | AffineObject[Mu, Data]:
        r"""
        Get item(s) by index or slice.
        
        Args:
            key :
                Index or slice to retrieve.
            
        Returns:
            If ``key`` is an integer, returns a tuple ``(theta[key], data[key])``.
            If ``key`` is a slice, returns a new ``AffineObject`` with the sliced terms.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2, lambda mu: 1.0]
        >>> data = [1.0, 2.0, 3.0]
        >>> ad = AffineObject(theta, data)
        >>> theta_0, data_0 = ad[0]  # Get first term
        >>> sliced = ad[1:]  # Slice returns new AffineObject
        >>> len(sliced)
        2
        """
        if isinstance(key, int):
            return (self.theta[key], self.data[key])
        return self._construct_new(self.theta[key], self.data[key])
    
    
    def __setitem__(self, key: int | slice, value: AffineObject[Mu, Data] | tuple[ParametricObject[Mu, float], Data] | Iterable[tuple[ParametricObject[Mu, float], Data]]):
        r"""
        Set item(s) by index or slice.
        
        Args:
            key :
                Index or slice to set.
            value :
                For integer index: a tuple ``(theta, data)``, a list ``[(theta, data)]`` or an ``AffineObject``of length 1. 
                For slice: an ``AffineObject`` or a list of ``(theta, data)`` tuples.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> ad[0] = (lambda mu: 2*mu, 5.0)  # Set single term
        >>> ad[0:2] = [(lambda mu: mu**3, 10.0), (lambda mu: 1.0, 20.0)]  # Set slice with list of tuples
        """
        
        if isinstance(value, AffineObject):
            theta = value.theta
            data = value.data
        else:
            if isinstance(key, int) and len(value) == 2:
                theta = [value[0]]
                data = [value[1]]
            else:
                theta = [v[0] for v in value]
                data = [v[1] for v in value]
                
        if isinstance(key, int):
            if len(theta) != 1:
                raise ValueError("When setting a single element, value must be a single (theta, data) tuple or AffineObject of length 1.")
            self.theta[key] = theta[0]
            self.data[key] = data[0]
        else:                
            self.theta[key] = theta
            self.data[key] = data
        
    def __delitem__(self, key: int | slice):
        r"""
        Delete item(s) by index or slice.
        
        Args:
            key :
                Index or slice to delete.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2, lambda mu: 1.0]
        >>> data = [1.0, 2.0, 3.0]
        >>> ad = AffineObject(theta, data)
        >>> del ad[1]  # Delete second term
        >>> len(ad)
        2
        """
        del self.theta[key]
        del self.data[key]
        
    def __len__(self) -> int:
        r"""
        Return the number of terms in the affine list.
        
        Returns:
            Number of terms :math:`Q` in the list.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> len(ad)
        2
        """
        return len(self.data)
    
    def __iter__(self) -> zip[ParametricObject[Mu, float], any]:
        r"""
        Iterate over ``(theta, data)`` pairs.
        
        Yields
        ------
            Pairs of ``(theta_q, data_q)`` for each term in the list.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> for theta_q, data_q in ad:
        ...     print(theta_q(2.0), data_q)
        2.0 1.0
        4.0 2.0
        """
        return zip(self.theta, self.data)
    
    def insert(self, index: int, value: tuple[ParametricObject[Mu, float], any]):
        r"""
        Insert a new term at a given position.
        
        Args:
            index :
                Position at which to insert the new term.
            value :
                A tuple ``(theta, data)`` representing the term to insert.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> ad.insert(1, (lambda mu: 2*mu, 5.0))
        >>> len(ad)
        3
        >>> ad[1][1]  # Data of inserted term
        5.0
        """
        self.theta.insert(index, wrap_scalar(value[0]))
        self.data.insert(index, value[1])
    
    def __add__(self, other: AffineObject[Mu, Data] | Iterable[tuple[ParametricObject[Mu, float], Data]]) -> AffineObject[Mu, Data]:
        r"""
        Add object to this affine decomposition.
        
        Concatenates two decompositions:
        
        .. math::
            \sum_{q=1}^{Q_1} \theta_q(\mu) \cdot \text{data}_q + 
            \sum_{q=1}^{Q_2} \tilde{\theta}_q(\mu) \cdot \tilde{\text{data}}_q
        
        
        Args:
            other :
                Either another `AffineObject` or a iterable of ``(theta,data)`` tuples.
        
        Returns:
            A new `AffineObject` with the combined terms.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> ad2 = ad + 5.0
        >>> len(ad2)
        3
        >>> ad2(1.0)  # 1*1 + 1*2 + 5 = 8
        8.0
        """
        if isinstance(other, AffineObject):
            theta = other.theta
            data  = other.data
        elif other == 0:
            theta = []
            data = []
        else:
            theta = [d[0] for d in other]
            data  = [d[1] for d in other]
        return self._construct_new(self.theta + theta, self.data + data)
    
    __radd__ = __add__
    __radd__.__doc__ = __add__.__doc__
    
    add = __add__
    add.__doc__ = __add__.__doc__
    
    def __neg__(self) -> AffineObject[Mu, Data]:
        r"""
        Negation of the affine decomposition.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> ad_neg = -ad
        >>> ad_neg(2.0)  # -(2*1 + 4*2) = -10
        -10.0
        """
            
        neg_theta = []
        for theta_i in self.theta:
            neg_theta.append(multiply_theta(-1.0, theta_i))
        return self._construct_new(neg_theta, self.data)
    
    def __sub__(self, other: AffineObject[Mu, Data] | Iterable[tuple[ParametricObject[Mu, float], Data]]):    # self - other
        return -( (-self) + other )

    def __rsub__(self, other: AffineObject[Mu, Data] | Iterable[tuple[ParametricObject[Mu, float], any]]):   # other - self
        return (-self) + other
        
    
    def compress(self) -> AffineObject[Mu, Data]:
        r"""
        Compress the affine decomposition by combining constant terms.
        
        All terms with  constant coefficients are combined into a single term, 
        reducing the total number of terms in the decomposition. 
        Further, terms with constant zero coefficients are removed, unless all terms are zero, 
        in which case it returns a single term with zero coefficient, 
        to avoid returning an empty decomposition.
        
        Returns:
            A new `AffineObject` with constant terms merged.
            
        Examples
        --------
        >>> theta = [2.0, lambda mu: mu, 3.0]
        >>> data = [1.0, 2.0, 1.0]
        >>> ad = AffineObject(theta, data)
        >>> len(ad)
        3
        >>> len(ad.compress()) # Two constant terms combined
        2
        """
        constant_data = None
        zero_data = None
        new_theta = []
        new_data = []
        
        for theta_q, data_q in self:
            if isinstance(theta_q, TrivialParametric):
                if theta_q.data == 0.0:
                    if zero_data is None:
                        zero_data = data_q
                    else:
                        zero_data += data_q
                else:
                    if constant_data is None:
                        constant_data = theta_q.data * data_q
                    else:
                        constant_data += theta_q.data * data_q
            else:
                new_theta.append(theta_q)
                new_data.append(data_q)
        
        if constant_data is not None:
            new_theta.insert(0, 1.0)
            new_data.insert(0, constant_data)
        
        if len(new_data) == 0:
            new_theta.insert(0, 0.0)
            new_data.insert(0, zero_data)
            
        return self._construct_new(new_theta, new_data)
    
    def apply2data(self, func: Callable[[any], any]) -> AffineObject[Mu, Data]:
        r"""
        Apply a function to each data term in the decomposition and return
        a new affine object with the transformed data but same theta.
        
        Args:
            func :
                Function to apply to each data term.
        
        Returns:
            A new `AffineObject` with the transformed data terms.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [1.0, 2.0]
        >>> ad = AffineObject(theta, data)
        >>> ad_squared = ad.apply2data(lambda d: d**2)
        >>> ad_squared(2.0)  # 2*1^2 + 4*2^2 = 18
        18.0
        """
        new_data = [func(d) for d in self.data]
        return self._construct_new(self.theta, new_data, _ConstructNew.APPLY2DATA)
    
    def apply2theta[Mu2](self, func: Callable[[ParametricObject[Mu, float]], ParametricObject[Mu2, float]]) -> AffineObject[Mu2, Data]:
        r"""
        Apply a function to each theta term in the decomposition and return
        a new affine object with the transformed theta but same data.
        
        Args:
            func :
                Function to apply to each theta term.
        
        Returns:
            A new `AffineObject` with the transformed theta terms.
        """
        new_theta = [func(t) for t in self.theta]
        return self._construct_new(new_theta, self.data, _ConstructNew.SAME)   

    

class AffineLinear(AffineObject[Mu, Data], ParametricLinear[Mu, Data]):
    r"""
    An affine decomposition of a parametric linear object (e.g. a matrix or vector).
    
    Can be evaluated at a parameter value :math:`\mu` to compute the value of the affine decomposition at that parameter, i.e.,
    
    .. math::
        \text{data}(\mu) = \sum_{q=1}^{Q} \theta_q(\mu) \cdot \text{data}_q,
        
    where :math:`\mu` is a parameter of type ``Mu``, and :math:`\text{data}(\mu)` is a linear object of type ``Data``, e.g. a matrix or vector.
    
    Supports basic linear algebra operations, such as matrix-matrix or matrix-vector multiplication.
    """
    
    _T: AffineLinear[Mu, Data] | None = None
    """Cached transpose."""
    
    
    def __init__(self, theta: list[ParametricObject[Mu, float]] | AffineObject[Mu, Data] | Iterable = [], data: list = None):
        r"""
        Args:
            theta :
                List of parameter-dependent coefficient functions. Each function should accept a 
                parameter value and return a scalar coefficient. Alternatively, an `AffineObject` can be provided, in which case its ``theta`` and ``data`` attributes are used. Alternatively, an iterable of alternating ``(theta, data)`` entries can be provided, e.g. ``[(theta1,data1), (theta2,data2)]``.
            data :
                List of parameter-independent data terms. Or empty if ``theta`` is an `AffineObject` or iterable. Do not have to be of the same type, but, if added up, must return an object of type ``Data``. Should support ``*``- and ``@``-multiplication for element-wise- and matrix-multiplication, as well as the ``.T`` attribute for the transposed.
        """
        super().__init__(theta, data)
        if len(self) != 0:
            shape = self.data[0].shape
            for d in self.data:
                if d.shape != shape:
                    raise ValueError("All data must have the same shape.")
            self.shape = shape
        else:
            self.shape = None
            
    def __repr__(self):
        if self.shape is not None:        
            shape = f"({self.shape[0]},"
            shape += "".join([f" {s}," for s in self.shape[1:]])[:-1] + ")"
        else:
            shape = "(unknown shape)"            
        return f"<{self.__class__.__name__} with {len(self)} affine terms and shape {shape}>"
            
    def insert(self, index: int, value: tuple[ParametricObject[Mu, float], Data]):
        super().insert(index, value)
        if self.shape is None:
            self.shape = value[1].shape
        elif value[1].shape != self.shape:
            raise ValueError("Inserted data must have the same shape.")
        self._T = None  # Invalidate cached transpose

        
    @property
    def T(self) -> AffineLinear[Mu, Data]:
        r"""
        Transpose of the affine linear object.
        """
        if self._T is None:
            transposed_data = [d.T for d in self.data]
            self._T = self._construct_new(self.theta, transposed_data)
        return self._T
    
    def transpose(self) -> AffineLinear[Mu, Data]:
        return self.T
    transpose.__doc__ = T.__doc__  
    
    def __mul__[Data2, Data3](self, other: AffineLinear[Mu, Data2] | Data2) -> AffineLinear[Mu, Data3]:
        r"""
        ``*``-multiplication with another object.
        
        If ``other`` is an :class:`AffineLinear`, this creates a new
        decomposition with :math:`Q_1 \cdot Q_2` terms by combining each term
        of ``self`` with each term of ``other``:
        
        .. math::
            \sum_{q_1=1}^{Q_1} \sum_{q_2=1}^{Q_2} 
            \theta_{q_1}(\mu) \cdot \theta_{q_2}(\mu) \cdot 
            \text{data}_{q_1} * \text{data}_{q_2}
        
        If ``other`` is not an :class:`AffineDecomposition`, it applies
        ``*``-multiplication to each data term:
        
        .. math::
            \sum_{q=1}^{Q} \theta_q(\mu) \cdot (\text{data}_q * \text{other})
        
        Args:
            other :
                Either another :class:`AffineLinear` or a compatible
                data object for ``*``-multiplication.
        
        Returns:
            A new :class:`AffineLinear` with the multiplied terms. Thereby, the type ``Data3`` is the result type of the ``*``-multiplication between ``Data`` and ``Data2``.
            
        Examples
        --------
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [2.0, 3.0]
        >>> ad = AffineLinear(theta, data)
        >>> ad2 = ad * 2.0
        >>> ad2(1.0)  # (1*4.0) + (1*6.0) = 10.0
        10.0
        """
        if isinstance(other, AffineLinear):
            new_theta = []
            new_data = []
            for theta_i, data_i in self:
                for theta_j, data_j in other:
                    new_theta.append(multiply_theta(theta_i, theta_j))
                    new_data.append(data_i * data_j)
            return self._construct_new(new_theta, new_data, _ConstructNew.MUL)
        else:
            new_data = [d * other for d in self.data]
            return self._construct_new(self.theta, new_data, _ConstructNew.MUL)
    
    def __rmul__[Data2, Data3](self, other: Data2) -> AffineLinear[Mu, Data3]:
        r"""
        Right ``*``-multiplication with a non-parametric object.
        
        Applies ``*``-multiplication from the left to each data term:
        
        .. math::
            \sum_{q=1}^{Q} \theta_q(\mu) \cdot (\text{other} * \text{data}_q)
        
        Args:
            other :
                A compatible data object for ``*``-multiplication.
        
        Returns:
            A new :class:`AffineLinear` with the multiplied terms. Thereby, the type ``Data3`` is the result type of the ``*``-multiplication between ``Data`` and ``Data2``.
        """
        new_data = [other * d for d in self.data]
        return self._construct_new(self.theta, new_data, _ConstructNew.MUL)
    
    multiply = __mul__
    multiply.__doc__ = __mul__.__doc__
    
    def __matmul__[Data2, Data3](self, other: AffineLinear[Mu, Data2] | Data2) -> AffineLinear[Mu, Data3]:
        r"""
        ``@``-multiplication with another object.
        
        If ``other`` is an :class:`AffineLinear`, this creates a new
        decomposition with :math:`Q_1 \cdot Q_2` terms by combining each term
        of ``self`` with each term of ``other``:
        
        .. math::
            \sum_{q_1=1}^{Q_1} \sum_{q_2=1}^{Q_2} 
            \theta_{q_1}(\mu) \cdot \theta_{q_2}(\mu) \cdot 
            \text{data}_{q_1} @ \text{data}_{q_2}
        
        If ``other`` is not an :class:`AffineLinear`, it applies
        ``@``-multiplication to each data term:
        
        .. math::
            \sum_{q=1}^{Q} \theta_q(\mu) \cdot (\text{data}_q @ \text{other})
        
        Args:
            other :
                Either another :class:`AffineLinear` or a compatible
                data object for ``@``-multiplication.
        
        Returns:
            A new :class:`AffineLinear` with the multiplied terms. Thereby, the type ``Data3`` is the result type of the ``@``-multiplication between ``Data`` and ``Data2``.
            
        Examples
        --------
        >>> import numpy as np
        >>> theta = [lambda mu: mu, lambda mu: mu**2]
        >>> data = [np.eye(2), 2*np.eye(2)]
        >>> ad = AffineLinear(theta, data)
        >>> v = np.array([1.0, 2.0])
        >>> result = ad @ v
        >>> np.allclose(result(1.0), 3*v)  # (1*I + 1*2I) @ v = 3*v
        True
        """
        if isinstance(other, AffineLinear):
            new_theta = []
            new_data = []
            for theta_i, data_i in self:
                for theta_j, data_j in other:
                    new_theta.append(multiply_theta(theta_i, theta_j))
                    new_data.append(data_i @ data_j)
            return self._construct_new(new_theta, new_data, _ConstructNew.MATMUL)
        else:
            new_data = [d @ other for d in self.data]
            return self._construct_new(self.theta, new_data, _ConstructNew.MATMUL)
    
    def __rmatmul__[Data2, Data3](self, other: Data2) -> AffineLinear[Mu, Data3]:
        r"""
        Right ``@``-multiplication with a non-parametric object.
        
        Applies ``@``-multiplication from the left to each data term:
        
        .. math::
            \sum_{q=1}^{Q} \theta_q(\mu) \cdot (\text{other} @ \text{data}_q)
        
        Args:
            other :
                A compatible data object for ``@``-multiplication.
        
        Returns:
            A new :class:`AffineLinear` with the multiplied terms. Thereby, the type ``Data3`` is the result type of the ``@``-multiplication between ``Data`` and ``Data2``.
        """
        new_data = [other @ d for d in self.data]
        return self._construct_new(self.theta, new_data, _ConstructNew.MATMUL)
    
    matmul = __matmul__
    matmul.__doc__ = __matmul__.__doc__
    
    rmatmul = __rmatmul__
    rmatmul.__doc__ = __rmatmul__.__doc__
    
    
class AffineFunction(AffineObject[Mu,Callable]):
    r"""
    An affine decomposition of a parametric function.
    
    Can be evaluated at a parameter value :math:`\mu` to assemble a function,
    
    .. math::
        \text{data}(\mu) = \sum_{q=1}^{Q} \theta_q(\mu) \cdot \text{data}_q,
        
    where :math:`\mu` is a parameter of type ``Mu``, and :math:`\text{data}(\mu)` is a callable object, i.e. a function, e.g. use it like :math:`\text{data}(\mu)(x)`.
    """
    
    def __call__(self, mu: Mu) -> Callable:
        theta = [t(mu) for t in self.theta]
        def func(*args, **kwargs):
            val = theta[0] * self.data[0](*args, **kwargs)
            for theta_q, data_q in zip(theta[1:], self.data[1:]):
                val += theta_q * data_q(*args, **kwargs)
            return val
        return func
    
    
def wrap_affinelinear(data: Data | AffineLinear[Mu, Data]) -> AffineLinear[Mu, Data]:
    r"""
    Wrap a data object into an :class:`AffineLinear` with a single constant term.
    
    If the input is already an :class:`AffineLinear`, it is returned unchanged.
    
    Args:
        data :
            A data object or an :class:`AffineLinear`.
    
    Returns:
        An :class:`AffineLinear` representing the input data.
        
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> aff_A = wrap_affinelinear(A)
    >>> aff_A(0.5)  # Constant affine linear returns the same matrix
    array([[1., 2.],
           [3., 4.]])
    """
    if isinstance(data, AffineLinear):
        return data
    elif isinstance(data, AffineObject):
        return AffineLinear(data)
    else:
        return AffineLinear([1.0], [data])
    
    
class ScalarComponentList(Sequence):
    r"""Wrapper for a vector valued function into a list of scalar component functions.
    
    Given a vector-valued function :math:`f(\mu) = (f_1(\mu), \dots, f_n(\mu))`, this class provides a list-like interface to access the individual scalar component functions :math:`f_i(\mu)`, so that this class 
    can be used as the ``theta`` argument in an :class:`AffineObject`.
    
    It provides a simple caching mechanism that caches :math:`f(\mu)` for the last parameter value :math:`\mu` to avoid redundant evaluations of the vector-valued function :math:`f` when accessing multiple components :math:`f_1,\dots,f_n` for the same parameter value.
    """
    
    _last_mu   = None
    _last_func = None
    
    def __init__(self, func, n):
        self._func = func
        self._thetas = np.array([lambda mu, i=i: self._eval_theta(mu)[i] for i in range(n)])
        
    def _eval_theta(self, mu):
        if mu is not self._last_mu:
            self._last_mu   = mu
            self._last_func = self._func(mu)
        return self._last_func
        
    def __getitem__(self, key):
        return self._thetas[key]
    
    def __len__(self):
        return len(self._thetas)