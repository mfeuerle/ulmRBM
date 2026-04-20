"""
Comprehensive test suite for AffineLinear.

Tests cover:
- AffineLinear with dense matrices, sparse matrices, LinearOperator, and vectors
- All operations: call, add, mul, matmul, compress, transpose, etc.
- MutableSequence protocol: getitem, setitem, delitem, insert, len, iter
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from ulmRBM.affine import AffineLinear, AffineObject
from ulmRBM.core import TrivialParametric


# =============================================================================
# Fixtures for different data types
# =============================================================================

@pytest.fixture
def theta_funcs():
    """Standard theta functions for testing."""
    return [lambda mu: mu, lambda mu: mu**2, lambda mu: 1.0]


@pytest.fixture
def theta_funcs_2():
    """Another set of theta functions for binary operations."""
    return [lambda mu: 2*mu, lambda mu: 0.5]


# -----------------------------------------------------------------------------
# Data factories for different types
# -----------------------------------------------------------------------------

def make_vectors_1d(n=5):
    """Create 1D vectors of shape (n,)."""
    return [np.arange(n, dtype=float), 
            np.ones(n), 
            np.arange(n, 0, -1, dtype=float)]


def make_matrices_dense(n=4):
    """Create dense numpy matrices."""
    return [np.eye(n), 
            np.diag(np.arange(1, n+1, dtype=float)), 
            np.ones((n, n))]


def make_matrices_sparse(n=4):
    """Create scipy sparse arrays (sparray format)."""
    return [sp.csr_array(sp.eye(n)),
            sp.csr_array(sp.diags(np.arange(1, n+1, dtype=float))),
            sp.csr_array(np.ones((n, n)))]


def make_matrices_linop(n=4):
    """Create LinearOperator objects."""
    matrices = make_matrices_dense(n)
    return [aslinearoperator(m) for m in matrices]


def make_matrices_rectangular_dense(m=3, n=5):
    """Create non-square dense numpy matrices of shape (m, n)."""
    return [np.ones((m, n)),
            np.arange(m * n, dtype=float).reshape(m, n),
            np.eye(m, n)]


def make_matrices_rectangular_sparse(m=3, n=5):
    """Create non-square sparse arrays of shape (m, n)."""
    return [sp.csr_array(np.ones((m, n))),
            sp.csr_array(np.arange(m * n, dtype=float).reshape(m, n)),
            sp.csr_array(np.eye(m, n))]


# =============================================================================
# Parametrized test data
# =============================================================================

LINEAR_DATA_CASES = [
    pytest.param(make_vectors_1d, "vector", (5,), id="vector_1d"),
    pytest.param(make_matrices_dense, "dense", (4, 4), id="matrix_dense"),
    pytest.param(make_matrices_sparse, "sparse", (4, 4), id="matrix_sparse"),
    pytest.param(make_matrices_linop, "linop", (4, 4), id="matrix_linop"),
]

MATRIX_CASES = [
    pytest.param(make_matrices_dense, "dense", (4, 4), id="matrix_dense"),
    pytest.param(make_matrices_sparse, "sparse", (4, 4), id="matrix_sparse"),
    pytest.param(make_matrices_linop, "linop", (4, 4), id="matrix_linop"),
]

RECTANGULAR_MATRIX_CASES = [
    pytest.param(make_matrices_rectangular_dense, "dense", (3, 5), id="rect_dense"),
    pytest.param(make_matrices_rectangular_sparse, "sparse", (3, 5), id="rect_sparse"),
]


# =============================================================================
# Test Classes
# =============================================================================

class TestAffineLinearInitialization:
    """Test AffineLinear initialization with different data types."""
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", LINEAR_DATA_CASES)
    def test_initialization(self, theta_funcs, make_data, data_type, expected_shape):
        """Test AffineLinear can be created with different data types."""
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        assert len(al) == 3
        assert al.shape == expected_shape
    
    def test_shape_mismatch(self, theta_funcs):
        """Test that mismatched shapes raise error."""
        data = [np.ones(5), np.ones(6)]  # Different sizes
        
        with pytest.raises(ValueError, match="same shape"):
            AffineLinear(theta_funcs[:2], data)
    
    def test_theta_data_length_mismatch(self):
        """Test that theta/data length mismatch raises error."""
        theta = [lambda mu: mu]
        data = [np.ones(5), np.ones(5)]
        
        with pytest.raises(ValueError, match="Length"):
            AffineLinear(theta, data)


class TestAffineLinearCall:
    """Test evaluation of AffineLinear at parameter values."""
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", LINEAR_DATA_CASES)
    def test_call(self, theta_funcs, make_data, data_type, expected_shape):
        """Test evaluation at parameter value."""
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        mu = 2.0
        result = al(mu)
        expected = mu * data[0] + mu**2 * data[1] + 1.0 * data[2]
        
        # Convert to dense for comparison if needed
        if sp.issparse(result):
            result = result.toarray()
        if sp.issparse(expected):
            expected = expected.toarray()
        if isinstance(result, LinearOperator):
            # For LinearOperator, test by applying to vectors
            if len(expected_shape) == 2:
                test_vec = np.ones(expected_shape[1])
                result = result @ test_vec
                expected = expected @ test_vec
        
        np.testing.assert_allclose(result, expected)


class TestAffineLinearTranspose:
    """Test transpose property."""
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", MATRIX_CASES)
    def test_transpose(self, theta_funcs, make_data, data_type, expected_shape):
        """Test transpose property."""
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        alT = al.T
        assert alT.shape == expected_shape  # Square matrix
        
        # Transpose of transpose should give original shape
        alTT = alT.T
        assert alTT.shape == al.shape
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", RECTANGULAR_MATRIX_CASES)
    def test_transpose_rectangular(self, theta_funcs, make_data, data_type, expected_shape):
        """Test transpose property for rectangular matrices."""
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        alT = al.T
        assert alT.shape == expected_shape[::-1]  # (3, 5) -> (5, 3)
        
        mu = 1.5
        result = alT(mu)
        expected = al(mu).T
        
        if sp.issparse(result):
            result = result.toarray()
        if sp.issparse(expected):
            expected = expected.toarray()
            
        np.testing.assert_allclose(result, expected)
    
    def test_transpose_cached(self, theta_funcs):
        """Test that transpose is cached."""
        data = make_matrices_dense()
        al = AffineLinear(theta_funcs, data)
        
        alT1 = al.T
        alT2 = al.T
        
        assert alT1 is alT2  # Same object
    
    def test_transpose_method_same_as_property(self, theta_funcs):
        """Test that transpose() method gives same result as T property."""
        data = make_matrices_dense()
        al = AffineLinear(theta_funcs, data)
        
        mu = 2.0
        np.testing.assert_allclose(al.T(mu), al.transpose()(mu))


class TestAffineLinearArithmetic:
    """Test arithmetic operations."""
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", LINEAR_DATA_CASES)
    def test_scalar_multiply(self, theta_funcs, make_data, data_type, expected_shape):
        """Test scalar multiplication."""
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        result = al * 3.0
        mu = 1.5
        
        res_eval = result(mu)
        exp_eval = 3.0 * al(mu)
        
        if sp.issparse(res_eval):
            res_eval = res_eval.toarray()
        if sp.issparse(exp_eval):
            exp_eval = exp_eval.toarray()
        if isinstance(res_eval, LinearOperator) and len(expected_shape) == 2:
            test_vec = np.ones(expected_shape[1])
            res_eval = res_eval @ test_vec
            exp_eval = exp_eval @ test_vec
            
        np.testing.assert_allclose(res_eval, exp_eval)
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", LINEAR_DATA_CASES)
    def test_scalar_rmultiply(self, theta_funcs, make_data, data_type, expected_shape):
        """Test right scalar multiplication."""
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        result = 3.0 * al
        mu = 1.5
        
        res_eval = result(mu)
        exp_eval = 3.0 * al(mu)
        
        if sp.issparse(res_eval):
            res_eval = res_eval.toarray()
        if sp.issparse(exp_eval):
            exp_eval = exp_eval.toarray()
        if isinstance(res_eval, LinearOperator) and len(expected_shape) == 2:
            test_vec = np.ones(expected_shape[1])
            res_eval = res_eval @ test_vec
            exp_eval = exp_eval @ test_vec
            
        np.testing.assert_allclose(res_eval, exp_eval)
    
    def test_add_affine_linear(self, theta_funcs, theta_funcs_2):
        """Test adding two AffineLinear objects."""
        data1 = make_vectors_1d()
        data2 = [np.ones(5), np.arange(5, dtype=float)]
        
        al1 = AffineLinear(theta_funcs, data1)
        al2 = AffineLinear(theta_funcs_2, data2)
        
        result = al1 + al2
        
        assert len(result) == len(al1) + len(al2)
        
        mu = 2.0
        np.testing.assert_allclose(result(mu), al1(mu) + al2(mu))
    
    def test_mul_affine_linear(self, theta_funcs, theta_funcs_2):
        """Test element-wise multiplication of two AffineLinear objects."""
        data1 = make_vectors_1d()
        data2 = [np.ones(5), np.arange(5, dtype=float)]
        
        al1 = AffineLinear(theta_funcs, data1)
        al2 = AffineLinear(theta_funcs_2, data2)
        
        result = al1 * al2
        
        assert len(result) == len(al1) * len(al2)
        
        mu = 2.0
        np.testing.assert_allclose(result(mu), al1(mu) * al2(mu))


class TestAffineLinearMatmul:
    """Test matrix multiplication operations."""
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", MATRIX_CASES)
    def test_matmul_vector(self, theta_funcs, make_data, data_type, expected_shape):
        """Test matrix @ vector multiplication."""
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        v = np.ones(expected_shape[1])
        result = al @ v
        
        assert isinstance(result, AffineObject)
        
        mu = 2.0
        expected = al(mu) @ v
        
        if sp.issparse(expected):
            expected = expected.toarray().ravel()
        if isinstance(expected, LinearOperator):
            # Should not happen for matrix @ vector
            pass
            
        np.testing.assert_allclose(result(mu), expected)
    
    @pytest.mark.parametrize("make_data,data_type,expected_shape", RECTANGULAR_MATRIX_CASES)
    def test_matmul_vector_rectangular(self, theta_funcs, make_data, data_type, expected_shape):
        """Test matrix @ vector multiplication with rectangular matrix."""
        m, n = expected_shape
        data = make_data()
        al = AffineLinear(theta_funcs, data)
        
        v = np.ones(n)  # Vector size must match columns
        result = al @ v
        
        assert isinstance(result, AffineObject)
        assert result.shape == (m,)  # Result size matches rows
        
        mu = 2.0
        expected = al(mu) @ v
        
        if sp.issparse(expected):
            expected = expected.toarray().ravel()
            
        np.testing.assert_allclose(result(mu), expected)
    
    def test_matmul_affine_linear(self, theta_funcs, theta_funcs_2):
        """Test matrix @ matrix operation between AffineLinear objects."""
        data_mat = make_matrices_dense()
        data_vec = [np.ones(4), np.arange(4, dtype=float)]
        
        al_mat = AffineLinear(theta_funcs, data_mat)
        al_vec = AffineLinear(theta_funcs_2, data_vec)
        
        result = al_mat @ al_vec
        
        assert len(result) == len(al_mat) * len(al_vec)
        
        mu = 1.5
        np.testing.assert_allclose(result(mu), al_mat(mu) @ al_vec(mu))
    
    def test_rmatmul_dense_matrix(self, theta_funcs):
        """Test right matrix multiplication with dense array."""
        data = make_matrices_dense()
        al = AffineLinear(theta_funcs, data)
        
        left_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 2x4
        result = left_matrix @ al
        
        mu = 1.5
        np.testing.assert_allclose(result(mu), left_matrix @ al(mu))
    
    def test_rmatmul_sparse_matrix(self, theta_funcs):
        """Test right matrix multiplication with sparse array."""
        data = make_matrices_dense()
        al = AffineLinear(theta_funcs, data)
        
        left_matrix = sp.csr_array(np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))  # 2x4 sparse
        result = left_matrix @ al
        
        mu = 1.5
        expected = left_matrix @ al(mu)
        np.testing.assert_allclose(result(mu), expected)


class TestMutableSequenceProtocol:
    """Test MutableSequence protocol methods."""
    
    @pytest.fixture
    def al(self, theta_funcs):
        return AffineLinear(theta_funcs, make_vectors_1d())
    
    def test_getitem_int(self, al):
        """Test integer indexing."""
        theta, data = al[0]
        assert callable(theta)
        np.testing.assert_array_equal(data, np.arange(5))
    
    def test_getitem_slice(self, al):
        """Test slice indexing."""
        sliced = al[1:]
        assert len(sliced) == 2
        assert isinstance(sliced, AffineLinear)
    
    def test_setitem(self, al):
        """Test setting an item."""
        new_theta = lambda mu: 5.0 * mu
        new_data = np.zeros(5)
        
        al[1] = (new_theta, new_data)
        
        assert al.theta[1](1.0) == 5.0
        np.testing.assert_array_equal(al.data[1], np.zeros(5))
    
    def test_setitem_with_affine_linear(self, al, theta_funcs):
        """Test setting an item with an AffineLinear of length 1."""
        new_al = AffineLinear([lambda mu: 7.0 * mu], [np.full(5, 7.0)])
        al[1] = new_al
        
        assert al.theta[1](1.0) == 7.0
        np.testing.assert_array_equal(al.data[1], np.full(5, 7.0))
    
    def test_setitem_with_multi_element_affine_raises(self, al, theta_funcs):
        """Test setting a single index with multi-element AffineLinear raises."""
        multi_al = AffineLinear([lambda mu: mu, lambda mu: mu**2], 
                                [np.ones(5), np.zeros(5)])
        with pytest.raises(ValueError, match="When setting a single element"):
            al[1] = multi_al
    
    def test_setitem_slice_with_list_of_tuples(self, al):
        """Test setting a slice with a list of (theta, data) tuples."""
        al[0:2] = [
            (lambda mu: 10.0 * mu, np.ones(5)),
            (lambda mu: 20.0 * mu, np.full(5, 2.0)),
        ]
        
        assert al.theta[0](1.0) == 10.0
        assert al.theta[1](1.0) == 20.0
        np.testing.assert_array_equal(al.data[0], np.ones(5))
        np.testing.assert_array_equal(al.data[1], np.full(5, 2.0))
    
    def test_setitem_slice_with_affine_linear(self, al, theta_funcs):
        """Test setting a slice with an AffineLinear."""
        new_al = AffineLinear([lambda mu: 100.0, lambda mu: 200.0], 
                              [np.full(5, 100.0), np.full(5, 200.0)])
        al[0:2] = new_al
        
        assert al.theta[0](1.0) == 100.0
        assert al.theta[1](1.0) == 200.0
        np.testing.assert_array_equal(al.data[0], np.full(5, 100.0))
        np.testing.assert_array_equal(al.data[1], np.full(5, 200.0))
    
    def test_delitem(self, al):
        """Test deleting an item."""
        original_len = len(al)
        del al[1]
        assert len(al) == original_len - 1
    
    def test_insert(self, al):
        """Test inserting an item."""
        original_len = len(al)
        new_theta = lambda mu: 10.0
        new_data = np.full(5, 42.0)
        
        al.insert(1, (new_theta, new_data))
        
        assert len(al) == original_len + 1
        np.testing.assert_array_equal(al.data[1], np.full(5, 42.0))
    
    def test_iter(self, al):
        """Test iteration."""
        items = list(al)
        assert len(items) == 3
        for theta, data in items:
            assert callable(theta)
            assert isinstance(data, np.ndarray)
    
    def test_len(self, al):
        """Test length."""
        assert len(al) == 3


class TestCompress:
    """Test the compress method for combining constant terms."""
    
    def test_compress_with_trivial_parametric(self):
        """Test compressing TrivialParametric theta terms."""
        theta = [TrivialParametric(2.0), lambda mu: mu, TrivialParametric(3.0)]
        data = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]
        
        al = AffineLinear(theta, data)
        compressed = al.compress()
        
        assert len(compressed) == 2  # Two constants merged into one
        
        # Verify the result is the same
        mu = 2.5
        np.testing.assert_allclose(compressed(mu), al(mu))
    
    def test_compress_with_wrapped_scalars(self):
        """Test compressing scalar theta values (auto-wrapped to TrivialParametric)."""
        theta = [2.0, lambda mu: mu, 3.0]
        data = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        
        al = AffineLinear(theta, data)
        compressed = al.compress()
        
        assert len(compressed) == 2
        
        mu = 1.5
        np.testing.assert_allclose(compressed(mu), al(mu))
    
    def test_compress_no_constants(self):
        """Test compress with no constant terms."""
        theta = [lambda mu: mu, lambda mu: mu**2]
        data = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        
        al = AffineLinear(theta, data)
        compressed = al.compress()
        
        assert len(compressed) == 2  # No change
    
    def test_compress_all_constants(self):
        """Test compress when all terms are constant."""
        theta = [1.0, 2.0, 3.0]
        data = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        
        al = AffineLinear(theta, data)
        compressed = al.compress()
        
        assert len(compressed) == 1
        
        # Result should be constant for any mu
        np.testing.assert_allclose(compressed(0.0), compressed(100.0))


class TestAliases:
    """Test method aliases (add, multiply, apply, matmul)."""
    
    @pytest.fixture
    def al_vec(self, theta_funcs):
        return AffineLinear(theta_funcs[:2], make_vectors_1d()[:2])
    
    @pytest.fixture
    def al_mat(self, theta_funcs):
        return AffineLinear(theta_funcs[:2], make_matrices_dense()[:2])
    
    def test_add_alias(self, al_vec):
        """Test that add is an alias for __add__."""
        constant = np.ones(5)
        
        result1 = al_vec + [(1.0, constant)]
        result2 = al_vec.add([(1.0, constant)])
        
        mu = 2.0
        np.testing.assert_allclose(result1(mu), result2(mu))
    
    def test_multiply_alias(self, al_vec):
        """Test that multiply is an alias for __mul__."""
        result1 = al_vec * 2.0
        result2 = al_vec.multiply(2.0)
        
        mu = 2.0
        np.testing.assert_allclose(result1(mu), result2(mu))
    
    def test_matmul_alias(self, al_mat):
        """Test that matmul is an alias for __matmul__."""
        v = np.ones(4)
        
        result1 = al_mat @ v
        result2 = al_mat.matmul(v)
        
        mu = 2.0
        np.testing.assert_allclose(result1(mu), result2(mu))


class TestEdgeCases:
    """Test edge cases and error handling."""


class TestRightOperations:
    """Test right-hand operations (__radd__, __rmul__, __rmatmul__)."""
    
    @pytest.fixture
    def al_vec(self, theta_funcs):
        return AffineLinear(theta_funcs[:2], make_vectors_1d()[:2])
    
    @pytest.fixture
    def al_mat(self, theta_funcs):
        return AffineLinear(theta_funcs[:2], make_matrices_dense()[:2])
    
    def test_radd_scalar(self, theta_funcs):
        """Test right addition with scalar."""
        al = AffineObject(theta_funcs[:2], [1.0, 2.0])
        result = [(1.0, 5.0)] + al
        
        assert len(result) == 3
        mu = 2.0
        assert result(mu) == pytest.approx(5.0 + al(mu))
    
    def test_radd_array(self, al_vec):
        """Test right addition with numpy array."""
        constant = np.ones(5) * 10
        result = [(1.0, constant)] + al_vec
        
        assert len(result) == 3
        mu = 2.0
        np.testing.assert_allclose(result(mu), constant + al_vec(mu))
    
    def test_rmul(self, al_vec):
        """Test right scalar multiplication."""
        result = 3.0 * al_vec
        
        mu = 2.0
        np.testing.assert_allclose(result(mu), 3.0 * al_vec(mu))


class TestLinearOperatorSpecific:
    """Test LinearOperator-specific behavior."""
    
    def test_linop_initialization(self, theta_funcs):
        """Test AffineLinear with LinearOperator."""
        data = make_matrices_linop()
        al = AffineLinear(theta_funcs, data)
        
        assert len(al) == 3
        assert al.shape == (4, 4)
        
        # Verify all data are LinearOperators
        for d in al.data:
            assert isinstance(d, LinearOperator)
    
    def test_linop_call_with_matmul(self, theta_funcs):
        """Test LinearOperator evaluation by matrix-vector product."""
        data = make_matrices_linop()
        al = AffineLinear(theta_funcs, data)
        
        mu = 2.0
        result = al(mu)
        
        # LinearOperator, test by applying to vector
        v = np.ones(4)
        result_v = result @ v
        
        # Expected: sum of theta(mu) * data @ v
        expected_v = sum(theta(mu) * (d @ v) for theta, d in al)
        
        np.testing.assert_allclose(result_v, expected_v)
    
    def test_linop_transpose(self, theta_funcs):
        """Test transpose of LinearOperator-based AffineLinear."""
        data = make_matrices_linop()
        al = AffineLinear(theta_funcs, data)
        
        alT = al.T
        assert alT.shape == (4, 4)
        
        # Test by applying to vector
        v = np.ones(4)
        mu = 1.5
        
        result = alT(mu) @ v
        expected = al(mu).T @ v
        
        np.testing.assert_allclose(result, expected)