"""
Comprehensive test suite for the products module.

Tests cover:
- EuclideanInnerProduct
- MatrixInnerProduct (dense, sparse, parametric)
- InverseInnerProduct
- RestrictedInnerProduct
- OperatorInnerProduct
- Nested operations (multiple restrictions, inverse of restriction, restriction of inverse)
- Dual inner products
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ulmRBM.products import (
    EuclideanInnerProduct,
    MatrixInnerProduct,
    InverseInnerProduct,
    RestrictedInnerProduct,
    OperatorInnerProduct,
    orthonormalize,
)
from ulmRBM.solver import DirectSolver
from ulmRBM.affine import AffineLinear


# =============================================================================
# Fixtures: Test Vectors
# =============================================================================

@pytest.fixture
def vector_1d():
    """Single vector of dimension 5."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def vector_2d():
    """Two vectors of dimension 5 (shape 5x2)."""
    return np.array([[1.0, 2.0],
                     [2.0, 3.0],
                     [3.0, 4.0],
                     [4.0, 5.0],
                     [5.0, 6.0]])


@pytest.fixture
def basis_full_rank():
    """Full rank basis matrix 5x5."""
    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.rand(5, 5))
    return Q


@pytest.fixture
def basis_reduced():
    """Reduced basis matrix 5x3."""
    np.random.seed(43)
    Q, _ = np.linalg.qr(np.random.rand(5, 5))
    return Q[:, :3]


@pytest.fixture
def basis_further_reduced():
    """Further reduced basis matrix 3x2."""
    np.random.seed(44)
    Q, _ = np.linalg.qr(np.random.rand(3, 3))
    return Q[:, :2]


# =============================================================================
# Fixtures: Matrices
# =============================================================================

@pytest.fixture
def matrix_diagonal():
    """Diagonal SPD matrix with entries 1, 2, ..., 5."""
    return np.diag(np.arange(1.0, 6.0))


@pytest.fixture
def matrix_spd_dense():
    """Dense SPD matrix via A^T A + I."""
    np.random.seed(45)
    A = np.random.rand(5, 5)
    return A @ A.T + 5.0 * np.eye(5)


@pytest.fixture
def matrix_spd_sparse():
    """Sparse SPD tridiagonal matrix."""
    diag = 4.0 * np.ones(5)
    off_diag = -1.0 * np.ones(4)
    return sp.csr_array(sp.diags([off_diag, diag, off_diag], [-1, 0, 1]))


@pytest.fixture
def matrix_rectangular():
    """Rectangular matrix 5x3."""
    np.random.seed(46)
    return np.random.rand(5, 3)


@pytest.fixture
def affine_matrix():
    """Affine parametric matrix: M(mu) = mu * diag(1,...,5) + (1-mu) * I."""
    M1 = np.diag(np.arange(1.0, 6.0))
    M2 = np.eye(5)
    theta = [lambda mu: mu, lambda mu: 1.0 - mu]
    return AffineLinear(theta, [M1, M2])


@pytest.fixture
def mu_value():
    """Test parameter value."""
    return 0.6


@pytest.fixture
def matrix_spd_linearoperator():
    """SPD matrix as LinearOperator."""
    np.random.seed(45)
    A = np.random.rand(5, 5)
    M = A @ A.T + 5.0 * np.eye(5)
    
    def matvec(v):
        return M @ v
    
    return LinearOperator((5, 5), matvec=matvec)


@pytest.fixture
def non_orthogonal_basis():
    """Non-orthogonal basis vectors for orthonormalization tests."""
    np.random.seed(50)
    return np.random.rand(5, 3)


# =============================================================================
# Test EuclideanInnerProduct
# =============================================================================

class TestEuclideanInnerProduct:
    """Tests for EuclideanInnerProduct."""
    
    def test_initialization(self):
        """Test initialization and basic properties."""
        ip = EuclideanInnerProduct(5)
        assert ip.shape == (5, 5)
        assert ip.is_parametric is False
    
    def test_call(self, mu_value):
        """Test __call__ returns identity matrix."""
        ip = EuclideanInnerProduct(5)
        M = ip(mu_value)
        expected = sp.eye(5, format='csr')
        assert sp.issparse(M)
        assert np.allclose(M.toarray(), expected.toarray())
    
    def test_riesz_identity(self, vector_1d, vector_2d, mu_value):
        """Test Riesz map is identity for Euclidean inner product."""
        ip = EuclideanInnerProduct(5)
        
        # 1D vector
        riesz_1d = ip.riesz(mu_value, vector_1d)
        assert np.allclose(riesz_1d, vector_1d)
        
        # 2D vectors
        riesz_2d = ip.riesz(mu_value, vector_2d)
        assert np.allclose(riesz_2d, vector_2d)
    
    def test_inner_product(self, vector_1d, vector_2d, mu_value):
        """Test inner product computation."""
        ip = EuclideanInnerProduct(5)
        
        # Self inner product (1D)
        inner_self = ip.inner(mu_value, vector_1d)
        expected_self = np.dot(vector_1d, vector_1d)
        assert np.isclose(inner_self, expected_self)
        
        # Different vectors
        u = vector_1d
        v = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        inner_diff = ip.inner(mu_value, u, v)
        expected_diff = np.dot(u, v)
        assert np.isclose(inner_diff, expected_diff)
        
        # 2D vectors (self)
        inner_2d = ip.inner(mu_value, vector_2d)
        # inner() returns full matrix for 2D inputs
        expected_2d = vector_2d.T @ vector_2d
        assert np.allclose(inner_2d, expected_2d)
    
    def test_norm(self, vector_1d, vector_2d, mu_value):
        """Test norm computation."""
        ip = EuclideanInnerProduct(5)
        
        # 1D vector
        norm_1d = ip.norm(mu_value, vector_1d)
        expected_1d = np.linalg.norm(vector_1d)
        assert np.isclose(norm_1d, expected_1d)
        
        # 2D vectors
        norm_2d = ip.norm(mu_value, vector_2d)
        expected_2d = np.linalg.norm(vector_2d, axis=0)
        assert np.allclose(norm_2d, expected_2d)
    
    def test_inverse_is_self(self):
        """Test that inverse of Euclidean inner product is itself."""
        ip = EuclideanInnerProduct(5)
        ip_inv = ip._inverse
        assert ip_inv is ip
    
    def test_restrict(self, basis_reduced, mu_value):
        """Test restriction creates MatrixInnerProduct with V^T V."""
        ip = EuclideanInnerProduct(5)
        ip_restricted = ip.restrict(basis_reduced)
        
        # Should create MatrixInnerProduct (optimization)
        assert isinstance(ip_restricted, MatrixInnerProduct)
        assert ip_restricted.shape == (3, 3)
        
        # Matrix should be V^T V
        expected_matrix = basis_reduced.T @ basis_reduced
        actual_matrix = ip_restricted(mu_value)
        assert np.allclose(actual_matrix.toarray() if sp.issparse(actual_matrix) else actual_matrix, 
                          expected_matrix)
    
    def test_double_restriction(self, basis_reduced, basis_further_reduced, mu_value):
        """Test restricting twice."""
        ip = EuclideanInnerProduct(5)
        
        # First restriction: 5 -> 3
        ip_r1 = ip.restrict(basis_reduced)
        assert ip_r1.shape == (3, 3)
        
        # Second restriction: 3 -> 2
        ip_r2 = ip_r1.restrict(basis_further_reduced)
        assert ip_r2.shape == (2, 2)
        
        # Verify consistency
        u = np.array([1.0, 0.5])
        inner = ip_r2.inner(mu_value, u)
        
        # Manual computation: u -> V2 @ V1 @ u
        u_in_3d = basis_further_reduced @ u
        u_in_5d = basis_reduced @ u_in_3d
        expected = u_in_5d @ u_in_5d  # Euclidean inner product
        assert np.isclose(inner, expected)
    
    def test_inverse_of_restricted(self, basis_reduced, mu_value):
        """Test taking inverse of restricted Euclidean inner product."""
        ip = EuclideanInnerProduct(5)
        ip_restricted = ip.restrict(basis_reduced)
        ip_restricted_inv = ip_restricted._inverse
        
        assert isinstance(ip_restricted_inv, InverseInnerProduct)
        assert ip_restricted_inv.shape == (3, 3)
        
        # Test round-trip: M^{-1} M = I
        u = np.array([1.0, 2.0, 3.0])
        riesz = ip_restricted.riesz(mu_value, u)
        riesz_inv = ip_restricted_inv.riesz(mu_value, riesz)
        assert np.allclose(riesz_inv, u, rtol=1e-10)
    
    def test_triple_restriction(self, mu_value):
        """Test three levels of restriction."""
        np.random.seed(47)
        
        # Create nested bases
        Q1, _ = np.linalg.qr(np.random.rand(5, 5))
        V1 = Q1[:, :4]  # 5 -> 4
        
        Q2, _ = np.linalg.qr(np.random.rand(4, 4))
        V2 = Q2[:, :3]  # 4 -> 3
        
        Q3, _ = np.linalg.qr(np.random.rand(3, 3))
        V3 = Q3[:, :2]  # 3 -> 2
        
        ip = EuclideanInnerProduct(5)
        ip_r1 = ip.restrict(V1)
        ip_r2 = ip_r1.restrict(V2)
        ip_r3 = ip_r2.restrict(V3)
        
        assert ip_r3.shape == (2, 2)
        
        # Verify consistency
        u = np.array([1.0, -1.0])
        inner = ip_r3.inner(mu_value, u)
        
        # Full transformation (Euclidean inner product)
        u_full = V1 @ V2 @ V3 @ u
        expected = u_full @ u_full
        assert np.isclose(inner, expected, rtol=1e-10)


# =============================================================================
# Test MatrixInnerProduct
# =============================================================================

class TestMatrixInnerProduct:
    """Tests for MatrixInnerProduct with different matrix types."""
    
    @pytest.mark.parametrize("matrix_fixture", [
        "matrix_diagonal",
        "matrix_spd_dense",
        "matrix_spd_sparse",
    ])
    def test_initialization(self, matrix_fixture, request):
        """Test initialization with different matrix types."""
        M = request.getfixturevalue(matrix_fixture)
        ip = MatrixInnerProduct(M)
        assert ip.shape == (5, 5)
        assert ip.is_parametric is False  # Wrapped in TrivialParametric
    
    def test_initialization_linearoperator(self, matrix_spd_linearoperator):
        """Test initialization with LinearOperator."""
        ip = MatrixInnerProduct(matrix_spd_linearoperator)
        assert ip.shape == (5, 5)
        assert ip.is_parametric is False  # Wrapped in TrivialParametric
    
    def test_initialization_affine(self, affine_matrix):
        """Test initialization with affine parametric matrix."""
        ip = MatrixInnerProduct(affine_matrix)
        assert ip.shape == (5, 5)
        assert ip.is_parametric is True  # AffineLinear is not TrivialParametric
    
    def test_call(self, matrix_diagonal, mu_value):
        """Test __call__ returns the matrix."""
        ip = MatrixInnerProduct(matrix_diagonal)
        M = ip(mu_value)
        assert np.allclose(M.toarray() if sp.issparse(M) else M, matrix_diagonal)
    
    def test_riesz(self, matrix_diagonal, vector_1d, vector_2d, mu_value):
        """Test Riesz map computation."""
        ip = MatrixInnerProduct(matrix_diagonal)
        
        # 1D vector
        riesz_1d = ip.riesz(mu_value, vector_1d)
        expected_1d = matrix_diagonal @ vector_1d
        assert np.allclose(riesz_1d, expected_1d)
        
        # 2D vectors
        riesz_2d = ip.riesz(mu_value, vector_2d)
        expected_2d = matrix_diagonal @ vector_2d
        assert np.allclose(riesz_2d, expected_2d)
    
    def test_inner_product(self, matrix_diagonal, vector_1d, mu_value):
        """Test inner product computation."""
        ip = MatrixInnerProduct(matrix_diagonal)
        
        # Self inner product
        inner_self = ip.inner(mu_value, vector_1d)
        expected_self = vector_1d @ matrix_diagonal @ vector_1d
        assert np.isclose(inner_self, expected_self)
        
        # Different vectors
        v = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        inner_diff = ip.inner(mu_value, vector_1d, v)
        expected_diff = vector_1d @ matrix_diagonal @ v
        assert np.isclose(inner_diff, expected_diff)
    
    def test_norm(self, matrix_diagonal, vector_1d, mu_value):
        """Test norm computation."""
        ip = MatrixInnerProduct(matrix_diagonal)
        norm = ip.norm(mu_value, vector_1d)
        expected = np.sqrt(vector_1d @ matrix_diagonal @ vector_1d)
        assert np.isclose(norm, expected)
    
    def test_restrict(self, matrix_diagonal, basis_reduced, mu_value):
        """Test restriction to subspace (optimized to precompute V^T M V)."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = ip.restrict(basis_reduced)
        
        # Should create MatrixInnerProduct with V^T M V (optimization)
        assert isinstance(ip_restricted, MatrixInnerProduct)
        assert not isinstance(ip_restricted, RestrictedInnerProduct)
        assert ip_restricted.shape == (3, 3)
        
        # Check restricted matrix
        expected_matrix = basis_reduced.T @ matrix_diagonal @ basis_reduced
        actual_matrix = ip_restricted(mu_value)
        assert np.allclose(actual_matrix.toarray() if sp.issparse(actual_matrix) else actual_matrix,
                          expected_matrix)
    
    def test_parametric_affine(self, affine_matrix, vector_1d):
        """Test parametric behavior with affine matrix."""
        ip = MatrixInnerProduct(affine_matrix)
        
        mu1 = 0.0
        mu2 = 1.0
        
        # At mu=0, should use I
        M0 = affine_matrix(mu1)
        inner0 = ip.inner(mu1, vector_1d)
        expected0 = vector_1d @ M0 @ vector_1d
        assert np.isclose(inner0, expected0)
        
        # At mu=1, should use diag(1,...,5)
        M1 = affine_matrix(mu2)
        inner1 = ip.inner(mu2, vector_1d)
        expected1 = vector_1d @ M1 @ vector_1d
        assert np.isclose(inner1, expected1)
        
        # Should be different
        assert not np.isclose(inner0, inner1)
    
    def test_inverse(self, matrix_diagonal, vector_1d, mu_value):
        """Test getting the inverse inner product."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = ip._inverse
        
        assert isinstance(ip_inv, InverseInnerProduct)
        assert ip_inv.shape == (5, 5)
        
        # Test inverse computes M^{-1} v
        riesz_inv = ip_inv.riesz(mu_value, vector_1d)
        expected = np.linalg.solve(matrix_diagonal, vector_1d)
        assert np.allclose(riesz_inv, expected)
    
    def test_double_restriction(self, matrix_diagonal, basis_reduced, basis_further_reduced, mu_value):
        """Test restricting twice."""
        ip = MatrixInnerProduct(matrix_diagonal)
        
        # First restriction: 5 -> 3
        ip_r1 = ip.restrict(basis_reduced)
        assert ip_r1.shape == (3, 3)
        
        # Second restriction: 3 -> 2
        ip_r2 = ip_r1.restrict(basis_further_reduced)
        assert ip_r2.shape == (2, 2)
        
        # Test inner product consistency
        u = np.array([1.0, 0.5])
        inner = ip_r2.inner(mu_value, u)
        
        # Manual computation
        u_in_3d = basis_further_reduced @ u
        u_in_5d = basis_reduced @ u_in_3d
        expected = u_in_5d @ matrix_diagonal @ u_in_5d
        assert np.isclose(inner, expected)
    
    def test_riesz_linearoperator(self, vector_1d, mu_value):
        """Test Riesz map with LinearOperator inner product."""
        # Create reference dense matrix
        np.random.seed(45)
        A = np.random.rand(5, 5)
        M_dense = A @ A.T + 5.0 * np.eye(5)
        
        # Create as LinearOperator
        def matvec(v):
            return M_dense @ v
        
        M_op = LinearOperator((5, 5), matvec=matvec)
        
        ip_op = MatrixInnerProduct(M_op)
        ip_dense = MatrixInnerProduct(M_dense)
        
        # Compare Riesz maps
        riesz_op = ip_op.riesz(mu_value, vector_1d)
        riesz_dense = ip_dense.riesz(mu_value, vector_1d)
        assert np.allclose(riesz_op, riesz_dense, atol=1e-10)
    
    def test_inner_linearoperator(self, vector_1d, vector_2d, mu_value):
        """Test inner product computation with LinearOperator."""
        # Create reference dense matrix
        np.random.seed(45)
        A = np.random.rand(5, 5)
        M_dense = A @ A.T + 5.0 * np.eye(5)
        
        # Create as LinearOperator
        def matvec(v):
            return M_dense @ v
        
        M_op = LinearOperator((5, 5), matvec=matvec)
        
        ip_op = MatrixInnerProduct(M_op)
        ip_dense = MatrixInnerProduct(M_dense)
        
        # Compare inner products (1D)
        inner_op = ip_op.inner(mu_value, vector_1d)
        inner_dense = ip_dense.inner(mu_value, vector_1d)
        assert np.isclose(inner_op, inner_dense, atol=1e-10)
        
        # Compare inner products (2D)
        inner_2d_op = ip_op.inner(mu_value, vector_2d)
        inner_2d_dense = ip_dense.inner(mu_value, vector_2d)
        assert np.allclose(inner_2d_op, inner_2d_dense, atol=1e-10)
    
    def test_norm_linearoperator(self, vector_1d, mu_value):
        """Test norm computation with LinearOperator."""
        # Create reference dense matrix
        np.random.seed(45)
        A = np.random.rand(5, 5)
        M_dense = A @ A.T + 5.0 * np.eye(5)
        
        # Create as LinearOperator
        def matvec(v):
            return M_dense @ v
        
        M_op = LinearOperator((5, 5), matvec=matvec)
        
        ip_op = MatrixInnerProduct(M_op)
        ip_dense = MatrixInnerProduct(M_dense)
        
        # Compare norms
        norm_op = ip_op.norm(mu_value, vector_1d)
        norm_dense = ip_dense.norm(mu_value, vector_1d)
        assert np.isclose(norm_op, norm_dense, atol=1e-10)
    
    def test_restrict_linearoperator(self, basis_reduced, mu_value):
        """Test restriction with LinearOperator inner product."""
        # Create reference dense matrix
        np.random.seed(45)
        A = np.random.rand(5, 5)
        M_dense = A @ A.T + 5.0 * np.eye(5)
        
        # Create as LinearOperator
        def matvec(v):
            return M_dense @ v
        def rmatvec(v):
            return v.T @ M_dense
        
        M_op = LinearOperator((5, 5), matvec=matvec, rmatvec=rmatvec)
        
        ip_op = MatrixInnerProduct(M_op)
        ip_dense = MatrixInnerProduct(M_dense)
        
        # Restrict both
        ip_restricted_op = ip_op.restrict(basis_reduced)
        ip_restricted_dense = ip_dense.restrict(basis_reduced)
        
        # Compare on test vector
        u_reduced = np.array([1.0, 2.0, 3.0])
        
        riesz_op = ip_restricted_op.riesz(mu_value, u_reduced)
        riesz_dense = ip_restricted_dense.riesz(mu_value, u_reduced)
        assert np.allclose(riesz_op, riesz_dense, atol=1e-10)
        
        inner_op = ip_restricted_op.inner(mu_value, u_reduced)
        inner_dense = ip_restricted_dense.inner(mu_value, u_reduced)
        assert np.isclose(inner_op, inner_dense, atol=1e-10)
    
    def test_inverse_of_restricted(self, matrix_diagonal, basis_reduced, mu_value):
        """Test taking inverse of a restricted MatrixInnerProduct."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = ip.restrict(basis_reduced)
        ip_restricted_inv = ip_restricted._inverse
        
        assert isinstance(ip_restricted_inv, InverseInnerProduct)
        assert ip_restricted_inv.shape == (3, 3)
        
        # Test round-trip
        u = np.array([1.0, 2.0, 3.0])
        riesz = ip_restricted.riesz(mu_value, u)
        riesz_inv = ip_restricted_inv.riesz(mu_value, riesz)
        assert np.allclose(riesz_inv, u, rtol=1e-10)
    
    def test_restriction_of_inverse(self, matrix_diagonal, basis_reduced, mu_value):
        """Test restricting an inverse inner product."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        ip_inv_restricted = ip_inv.restrict(basis_reduced)
        
        assert ip_inv_restricted.shape == (3, 3)
        
        # Test inner product computation with M^{-1}
        u = np.array([1.0, 2.0, 3.0])
        inner = ip_inv_restricted.inner(mu_value, u)
        
        # Manual computation: (V u)^T M^{-1} (V u)
        u_full = basis_reduced @ u
        M_inv = np.linalg.inv(matrix_diagonal)
        expected = u_full @ M_inv @ u_full
        assert np.isclose(inner, expected)
    
    def test_triple_restriction(self, matrix_spd_dense, mu_value):
        """Test three levels of restriction."""
        np.random.seed(47)
        
        # Create nested bases
        Q1, _ = np.linalg.qr(np.random.rand(5, 5))
        V1 = Q1[:, :4]  # 5 -> 4
        
        Q2, _ = np.linalg.qr(np.random.rand(4, 4))
        V2 = Q2[:, :3]  # 4 -> 3
        
        Q3, _ = np.linalg.qr(np.random.rand(3, 3))
        V3 = Q3[:, :2]  # 3 -> 2
        
        ip = MatrixInnerProduct(matrix_spd_dense)
        ip_r1 = ip.restrict(V1)
        ip_r2 = ip_r1.restrict(V2)
        ip_r3 = ip_r2.restrict(V3)
        
        assert ip_r3.shape == (2, 2)
        
        # Verify consistency
        u = np.array([1.0, -1.0])
        inner = ip_r3.inner(mu_value, u)
        
        # Full transformation
        u_full = V1 @ V2 @ V3 @ u
        expected = u_full @ matrix_spd_dense @ u_full
        assert np.isclose(inner, expected, rtol=1e-10)
    
    def test_inverse_of_restricted(self, matrix_diagonal, basis_reduced, mu_value):
        """Test (M|_V)^{-1} operation with reference solution."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = ip.restrict(basis_reduced)
        ip_restricted_inv = ip_restricted._inverse
        u = np.array([1.0, 2.0, 3.0])
        inner = ip_restricted_inv.inner(mu_value, u)
        M_restricted = basis_reduced.T @ matrix_diagonal @ basis_reduced
        expected = u @ np.linalg.inv(M_restricted) @ u
        assert np.isclose(inner, expected, rtol=1e-10)

    def test_restricted_inverse(self, matrix_diagonal, basis_reduced, mu_value):
        """Test M^{-1}|_V operation with reference solution."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        ip_inv_restricted = ip_inv.restrict(basis_reduced)
        u = np.array([1.0, 2.0, 3.0])
        inner = ip_inv_restricted.inner(mu_value, u)
        u_full = basis_reduced @ u
        M_inv = np.linalg.inv(matrix_diagonal)
        expected = u_full @ M_inv @ u_full
        assert np.isclose(inner, expected, rtol=1e-10)


# =============================================================================
# Test InverseInnerProduct
# =============================================================================

class TestInverseInnerProduct:
    """Tests for InverseInnerProduct."""
    
    def test_initialization(self, matrix_diagonal):
        """Test initialization."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        
        assert ip_inv.shape == (5, 5)
        assert ip_inv._ip is ip
    
    def test_riesz_inverse(self, matrix_diagonal, vector_1d, mu_value):
        """Test Riesz map computes M^{-1} v."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        
        riesz = ip_inv.riesz(mu_value, vector_1d)
        expected = np.linalg.solve(matrix_diagonal, vector_1d)
        assert np.allclose(riesz, expected)
    
    def test_inner_product_inverse(self, matrix_diagonal, vector_1d, mu_value):
        """Test inner product with M^{-1}."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        
        inner = ip_inv.inner(mu_value, vector_1d)
        M_inv = np.linalg.inv(matrix_diagonal)
        expected = vector_1d @ M_inv @ vector_1d
        assert np.isclose(inner, expected)
    
    def test_inverse_of_inverse(self, matrix_diagonal):
        """Test that (M^{-1})^{-1} = M."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        ip_inv_inv = ip_inv._inverse
        
        assert ip_inv_inv is ip
    
    def test_default_solver_warning(self, matrix_diagonal):
        """Test warning when no solver is specified."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, solver=None)
        
        with pytest.warns(UserWarning, match="No solver specified"):
            _ = ip_inv.solver
    
    def test_restrict(self, matrix_diagonal, basis_reduced, mu_value):
        """Test restriction of inverse inner product."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        ip_inv_restricted = ip_inv.restrict(basis_reduced)
        
        assert ip_inv_restricted.shape == (3, 3)
        
        # Test inner product computation
        u = np.array([1.0, 2.0, 3.0])
        inner = ip_inv_restricted.inner(mu_value, u)
        
        # Manual: (V u)^T M^{-1} (V u)
        u_full = basis_reduced @ u
        M_inv = np.linalg.inv(matrix_diagonal)
        expected = u_full @ M_inv @ u_full
        assert np.isclose(inner, expected)
    
    def test_inverse_of_restricted_inverse(self, matrix_diagonal, basis_reduced, mu_value):
        """Test (M^{-1}|_V)^{-1}."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        ip_inv_restricted = ip_inv.restrict(basis_reduced)
        ip_inv_restricted_inv = ip_inv_restricted._inverse
        
        assert ip_inv_restricted_inv.shape == (3, 3)
        
        # Round-trip test
        u = np.array([1.0, 2.0, 3.0])
        riesz = ip_inv_restricted.riesz(mu_value, u)
        riesz_inv = ip_inv_restricted_inv.riesz(mu_value, riesz)
        assert np.allclose(riesz_inv, u, rtol=1e-10)
    
    def test_triple_restriction(self, matrix_diagonal, mu_value):
        """Test three levels of restriction on inverse inner product."""
        np.random.seed(48)
        
        # Create nested bases
        Q1, _ = np.linalg.qr(np.random.rand(5, 5))
        V1 = Q1[:, :4]  # 5 -> 4
        
        Q2, _ = np.linalg.qr(np.random.rand(4, 4))
        V2 = Q2[:, :3]  # 4 -> 3
        
        Q3, _ = np.linalg.qr(np.random.rand(3, 3))
        V3 = Q3[:, :2]  # 3 -> 2
        
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_inv = InverseInnerProduct(ip, DirectSolver())
        ip_inv_r1 = ip_inv.restrict(V1)
        ip_inv_r2 = ip_inv_r1.restrict(V2)
        ip_inv_r3 = ip_inv_r2.restrict(V3)
        
        assert ip_inv_r3.shape == (2, 2)
        
        # Test inner product computation
        u = np.array([1.0, -1.0])
        inner = ip_inv_r3.inner(mu_value, u)
        
        # Manual computation with M^{-1}
        u_full = V1 @ V2 @ V3 @ u
        M_inv = np.linalg.inv(matrix_diagonal)
        expected = u_full @ M_inv @ u_full
        assert np.isclose(inner, expected, rtol=1e-10)


# =============================================================================
# Test RestrictedInnerProduct
# =============================================================================

class TestRestrictedInnerProduct:
    """Tests for RestrictedInnerProduct."""
    
    def test_initialization(self, matrix_diagonal, basis_reduced):
        """Test initialization."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = RestrictedInnerProduct(basis_reduced, ip)
        
        assert ip_restricted.shape == (3, 3)
        assert ip_restricted._ip is ip
        assert np.allclose(ip_restricted._basis(None), basis_reduced)
    
    def test_riesz(self, matrix_diagonal, basis_reduced, mu_value):
        """Test Riesz map via basis transformation."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = RestrictedInnerProduct(basis_reduced, ip)
        
        u_reduced = np.array([1.0, 2.0, 3.0])
        riesz = ip_restricted.riesz(mu_value, u_reduced)
        
        # Should compute V^T M (V u)
        u_full = basis_reduced @ u_reduced
        expected = basis_reduced.T @ (matrix_diagonal @ u_full)
        assert np.allclose(riesz, expected)
    
    def test_inner_product(self, matrix_diagonal, basis_reduced, mu_value):
        """Test inner product via basis transformation."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = RestrictedInnerProduct(basis_reduced, ip)
        
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, 1.5, 2.5])
        
        inner = ip_restricted.inner(mu_value, u, v)
        
        # Should compute (V u)^T M (V v)
        u_full = basis_reduced @ u
        v_full = basis_reduced @ v
        expected = u_full @ matrix_diagonal @ v_full
        assert np.isclose(inner, expected)
    
    def test_no_simplification(self, basis_reduced):
        """Test that RestrictedInnerProduct does not simplify even for concrete matrices."""
        # Use Euclidean inner product (identity)
        ip = EuclideanInnerProduct(5)
        
        # Manually create RestrictedInnerProduct (bypassing the optimized restrict())
        ip_restricted = RestrictedInnerProduct(basis_reduced, ip)
        
        # Should remain RestrictedInnerProduct, not simplified to MatrixInnerProduct
        assert isinstance(ip_restricted, RestrictedInnerProduct)
        assert not isinstance(ip_restricted, MatrixInnerProduct)
    
    def test_double_restriction(self, matrix_diagonal, basis_reduced, basis_further_reduced, mu_value):
        """Test restricting a RestrictedInnerProduct again."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = RestrictedInnerProduct(basis_reduced, ip)
        
        # Second restriction: 3 -> 2
        ip_double_restricted = ip_restricted.restrict(basis_further_reduced)
        assert ip_double_restricted.shape == (2, 2)
        
        # Test consistency
        u = np.array([1.0, 0.5])
        inner = ip_double_restricted.inner(mu_value, u)
        
        # Manual computation through both bases
        u_in_3d = basis_further_reduced @ u
        u_in_5d = basis_reduced @ u_in_3d
        expected = u_in_5d @ matrix_diagonal @ u_in_5d
        assert np.isclose(inner, expected)
    
    def test_inverse_of_restricted(self, matrix_diagonal, basis_reduced, mu_value):
        """Test taking inverse of RestrictedInnerProduct."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_restricted = RestrictedInnerProduct(basis_reduced, ip)
        ip_restricted_inv = ip_restricted._inverse
        
        assert isinstance(ip_restricted_inv, InverseInnerProduct)
        assert ip_restricted_inv.shape == (3, 3)
        
        # Round-trip test
        u = np.array([1.0, 2.0, 3.0])
        riesz = ip_restricted.riesz(mu_value, u)
        riesz_inv = ip_restricted_inv.riesz(mu_value, riesz)
        assert np.allclose(riesz_inv, u, rtol=1e-10)
    
    def test_triple_restriction(self, matrix_diagonal, mu_value):
        """Test three levels of manual RestrictedInnerProduct nesting."""
        np.random.seed(49)
        
        # Create nested bases
        Q1, _ = np.linalg.qr(np.random.rand(5, 5))
        V1 = Q1[:, :4]  # 5 -> 4
        
        Q2, _ = np.linalg.qr(np.random.rand(4, 4))
        V2 = Q2[:, :3]  # 4 -> 3
        
        Q3, _ = np.linalg.qr(np.random.rand(3, 3))
        V3 = Q3[:, :2]  # 3 -> 2
        
        ip = MatrixInnerProduct(matrix_diagonal)
        # Use RestrictedInnerProduct directly (not the optimized restrict() method)
        ip_r1 = RestrictedInnerProduct(V1, ip)
        ip_r2 = RestrictedInnerProduct(V2, ip_r1)
        ip_r3 = RestrictedInnerProduct(V3, ip_r2)
        
        assert ip_r3.shape == (2, 2)
        
        # Test consistency
        u = np.array([1.0, -1.0])
        inner = ip_r3.inner(mu_value, u)
        
        # Manual computation through all bases
        u_in_3d = V3 @ u
        u_in_4d = V2 @ u_in_3d
        u_in_5d = V1 @ u_in_4d
        expected = u_in_5d @ matrix_diagonal @ u_in_5d
        assert np.isclose(inner, expected, rtol=1e-10)


# =============================================================================
# Test OperatorInnerProduct
# =============================================================================

class TestOperatorInnerProduct:
    """Tests for OperatorInnerProduct."""
    
    def test_subclass(self):
        """Most funktionality comes from the RestrictedInnerProduct base class."""
        # if that is no longer true, the inherited functionality must be re-tested
        assert issubclass(OperatorInnerProduct, RestrictedInnerProduct)
    
    def test_initialization_invertible(self, basis_full_rank):
        """Test initialization with invertible operator."""
        V = EuclideanInnerProduct(5)
        ip = OperatorInnerProduct(basis_full_rank, V)
        
        from ulmRBM.products import _InverseOperatorInnerProduct
        assert isinstance(ip._inverse, _InverseOperatorInnerProduct)
        
        assert ip.shape == (5, 5)
    
    def test_initialization_error(self, matrix_rectangular):
        """Test error when claiming non-square matrix is invertible."""
        V = EuclideanInnerProduct(3)
        
        with pytest.raises(ValueError, match="B can not be invertible"):
            OperatorInnerProduct(matrix_rectangular, V)
    
    def test_inverse(self, basis_full_rank, mu_value):
        """Test inverse computation."""
        V = EuclideanInnerProduct(5)
        ip = OperatorInnerProduct(basis_full_rank, V, solver=DirectSolver())
        # Get the inverse - should use specialized class
        ip_inv = ip._inverse
        from ulmRBM.products import _InverseOperatorInnerProduct
        assert isinstance(ip_inv, _InverseOperatorInnerProduct)
        # Test Riesz computation: should compute B^{-T} V^{-1} B^{-1} u
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        riesz = ip_inv.riesz(mu_value, u)
        # the inverse Riesz should satisfy:
        # riesz = B^{-T} V^{-1} B^{-1} u
        # Since V is Euclidean (identity), this simplifies to B^{-T} B^{-1} u
        B_inv = np.linalg.inv(basis_full_rank)
        expected = B_inv.T @ B_inv @ u
        assert np.allclose(riesz, expected, rtol=1e-10)
        # Test inner product: u^T (B^T V B)^{-1} u = u^T B^{-T} V^{-1} B^{-1} u
        inner = ip_inv.inner(mu_value, u)
        expected_inner = u @ B_inv.T @ B_inv @ u
        assert np.isclose(inner, expected_inner, rtol=1e-10)


# =============================================================================
# Test Dual Inner Products
# =============================================================================

class TestDualInnerProducts:
    """Tests for dual inner product functionality."""
    
    def test_dual_default_is_inverse(self, matrix_diagonal):
        """Test that default dual is the inverse."""
        ip = MatrixInnerProduct(matrix_diagonal)
        dual = ip.dual
        
        assert dual is ip._inverse
        assert isinstance(dual, InverseInnerProduct)
    
    def test_dual_setter(self, matrix_diagonal, matrix_spd_dense):
        """Test explicitly setting a dual inner product."""
        ip1 = MatrixInnerProduct(matrix_diagonal)
        ip2 = MatrixInnerProduct(matrix_spd_dense)
        
        # Set dual relationship
        ip1.dual = ip2
        
        assert ip1.dual is ip2
        assert ip2.dual is ip1
        assert ip1._dual is ip2
        assert ip2._dual is ip1
    
    def test_dual_setter_error(self, matrix_diagonal, matrix_spd_dense):
        """Test error when trying to set dual when already set."""
        ip1 = MatrixInnerProduct(matrix_diagonal)
        ip2 = MatrixInnerProduct(matrix_spd_dense)
        ip3 = MatrixInnerProduct(np.eye(5))
        
        # Set dual relationship
        ip1.dual = ip2
        
        # Try to set again - should fail
        with pytest.raises(ValueError, match="already a dual product"):
            ip1.dual = ip3
        
        with pytest.raises(ValueError, match="already a dual product"):
            ip2.dual = ip3
    
    def test_dual_restriction_propagation(self, matrix_diagonal, matrix_spd_dense, basis_reduced):
        """Test that dual relationship propagates through restriction."""
        ip = MatrixInnerProduct(matrix_diagonal)
        ip_dual = MatrixInnerProduct(matrix_spd_dense)
        
        # Set dual
        ip.dual = ip_dual
        
        # Restrict primal
        ip_restricted = ip.restrict(basis_reduced)
        
        # Dual should be automatically restricted
        ip_dual_restricted = ip_restricted.dual
        assert ip_dual_restricted.shape == (3, 3)
        
        # Should be the inverse of the restricted dual
        assert isinstance(ip_dual_restricted, InverseInnerProduct)

        u = np.array([1.0, 2.0, 3.0])
        result = ip_dual_restricted.riesz(None, u)
        expect = np.linalg.solve(basis_reduced.T @ np.linalg.solve(matrix_spd_dense, basis_reduced), u)
        assert np.allclose(result, expect, rtol=1e-5)


# =============================================================================
# Test Edge Cases and Errors
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_restrict_invalid_shape(self, matrix_diagonal):
        """Test error when restricting with 1D array."""
        ip = MatrixInnerProduct(matrix_diagonal)
        
        with pytest.raises(ValueError, match="must be a 2D array"):
            ip.restrict(np.array([1.0, 2.0, 3.0]))
    
    def test_shape_mismatch(self, matrix_diagonal):
        """Test operations with mismatched dimensions."""
        ip = MatrixInnerProduct(matrix_diagonal)
        
        # Wrong size vector
        u = np.array([1.0, 2.0, 3.0])  # Size 3, should be 5
        
        with pytest.raises((ValueError, IndexError)):
            ip.riesz(0.0, u)
    
    def test_empty_inner_product(self):
        """Test with empty/zero-dimensional spaces."""
        ip = EuclideanInnerProduct(0)
        assert ip.shape == (0, 0)
        
        u = np.array([])
        inner = ip.inner(0.0, u)
        assert inner == 0.0
    
    def test_operator_inner_product_inverse_computation(self, basis_full_rank, mu_value):
        """Test inverse computation for OperatorInnerProduct with reference solution."""
        V = EuclideanInnerProduct(5)
        ip = OperatorInnerProduct(basis_full_rank, V)
        # Get the inverse
        ip_inv = ip._inverse
        # Set a solver
        ip_inv.solver = DirectSolver()
        # Test Riesz computation: should compute B^{-T} V^{-1} B^{-1} u
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        riesz = ip_inv.riesz(mu_value, u)
        # For Euclidean V, riesz = B^{-T} B^{-1} u = (BB^T)^{-1} u
        B_inv = np.linalg.inv(basis_full_rank)
        expected_riesz = B_inv.T @ B_inv @ u
        assert np.allclose(riesz, expected_riesz, rtol=1e-10)
        # Test inner product: u^T (B^T V B)^{-1} u
        inner = ip_inv.inner(mu_value, u)
        expected_inner = u @ B_inv.T @ B_inv @ u
        assert np.isclose(inner, expected_inner, rtol=1e-10)
        # Verify consistency: inner(u, u) should equal u^T riesz(u)
        assert np.isclose(inner, u @ riesz, rtol=1e-10)


# =============================================================================
# Test orthonormalize function
# =============================================================================

class TestOrthonormalize:
    """Tests for the orthonormalize function."""
    
    def test_orthonormalize_euclidean(self, non_orthogonal_basis):
        """Test orthonormalization with Euclidean inner product."""
        ip = EuclideanInnerProduct(5)
        
        # Orthonormalize
        basis_orth, Q = orthonormalize(non_orthogonal_basis, ip, full=False)
        
        # Check orthogonality: B^T B should be identity
        gram = basis_orth.T @ basis_orth
        assert np.allclose(gram, np.eye(3), atol=1e-10)
        
        # Check normalization
        norms = ip.norm(None, basis_orth)
        assert np.allclose(norms, np.ones(3), atol=1e-10)
        
        # Check relationship: basis_orth = non_orthogonal_basis @ Q
        assert np.allclose(basis_orth, non_orthogonal_basis @ Q, atol=1e-10)
    
    def test_orthonormalize_euclidean_full(self, non_orthogonal_basis):
        """Test orthonormalization with full=True for stability."""
        ip = EuclideanInnerProduct(5)
        
        # Orthonormalize with full=True
        basis_orth, Q = orthonormalize(non_orthogonal_basis, ip, full=True)
        
        # Check orthogonality
        gram = basis_orth.T @ basis_orth
        assert np.allclose(gram, np.eye(3), atol=1e-10)
        
        # Check normalization
        norms = ip.norm(None, basis_orth)
        assert np.allclose(norms, np.ones(3), atol=1e-12)
    
    def test_orthonormalize_matrix_ip(self, non_orthogonal_basis, matrix_spd_dense):
        """Test orthonormalization with custom matrix inner product."""
        ip = MatrixInnerProduct(matrix_spd_dense)
        
        # Orthonormalize
        basis_orth, Q = orthonormalize(non_orthogonal_basis, ip, full=False)
        
        # Check orthonormality w.r.t. the custom inner product
        # (u_i, u_j)_M = u_i^T M u_j should be delta_ij
        gram = ip.inner(None, basis_orth)
        assert np.allclose(gram, np.eye(3), atol=1e-10)
        
        # Check normalization
        norms = ip.norm(None, basis_orth)
        assert np.allclose(norms, np.ones(3), atol=1e-10)
        
        # Check relationship
        assert np.allclose(basis_orth, non_orthogonal_basis @ Q, atol=1e-10)
    
    def test_orthonormalize_linearoperator(self, non_orthogonal_basis):
        """Test orthonormalization with LinearOperator directly."""
        # Create a LinearOperator for the Euclidean inner product
        def matvec(v):
            return v
        
        op = LinearOperator((5, 5), matvec=matvec)
        
        # Orthonormalize
        basis_orth, Q = orthonormalize(non_orthogonal_basis, op, full=False)
        
        # Check orthogonality
        gram = basis_orth.T @ basis_orth
        assert np.allclose(gram, np.eye(3), atol=1e-10)
        
        # Check normalization
        norms = np.linalg.norm(basis_orth, axis=0)
        assert np.allclose(norms, np.ones(3), atol=1e-10)
    
    def test_orthonormalize_preserves_span(self, non_orthogonal_basis):
        """Test that orthonormalization preserves the span of the basis."""
        ip = EuclideanInnerProduct(5)
        
        basis_orth, Q = orthonormalize(non_orthogonal_basis, ip, full=False)
        
        # Original and orthonormalized bases should span the same space
        # Check: every original basis vector can be expressed as linear combination
        for i in range(non_orthogonal_basis.shape[1]):
            coeff = basis_orth.T @ non_orthogonal_basis[:, i]
            reconstructed = basis_orth @ coeff
            assert np.allclose(reconstructed, non_orthogonal_basis[:, i], atol=1e-10)
    
    def test_orthonormalize_parametric_ip_raises(self, non_orthogonal_basis, affine_matrix):
        """Test that parametric inner products raise an error."""
        ip = MatrixInnerProduct(affine_matrix)
        
        with pytest.raises(ValueError, match="parameter independent"):
            orthonormalize(non_orthogonal_basis, ip, full=False)
    
    def test_orthonormalize_single_vector(self, matrix_spd_dense):
        """Test orthonormalization of a single vector."""
        ip = MatrixInnerProduct(matrix_spd_dense)
        
        v = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        basis_orth, Q = orthonormalize(v, ip, full=False)
        
        # Should have unit norm
        norm = ip.norm(None, basis_orth)
        assert np.isclose(norm, 1.0, atol=1e-10)
        
        # Should be parallel to original
        assert basis_orth.shape == (5, 1)
