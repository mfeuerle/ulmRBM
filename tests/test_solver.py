"""
Tests for solver module.

Tests cover:
- DirectSolver with dense and sparse matrices (square and rectangular)
- IterativeSolver with CG, GMRES, and LSMR
- LinearOperator support for iterative solvers
- wrap_solver utility
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ulmRBM.solver import DirectSolver, IterativeSolver, wrap_solver, Solver


# =============================================================================
# Matrix Factory Functions
# =============================================================================

def make_spd_dense(n=5):
    """Create a dense symmetric positive definite matrix."""
    np.random.seed(42)
    A = np.random.rand(n, n)
    return A @ A.T + n * np.eye(n)


def make_spd_sparse(n=5):
    """Create a sparse symmetric positive definite matrix (tridiagonal)."""
    diag = 4.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    return sp.csr_array(sp.diags([off_diag, diag, off_diag], [-1, 0, 1]))


def make_square_dense(n=5):
    """Create a square non-symmetric dense matrix (diagonally dominant)."""
    np.random.seed(42)
    return np.random.rand(n, n) + n * np.eye(n)


def make_square_sparse(n=5):
    """Create a square non-symmetric sparse matrix (diagonally dominant)."""
    np.random.seed(42)
    A = sp.random(n, n, density=0.5, format='csr', random_state=42)
    return sp.csr_array(A + n * sp.eye(n, format='csr'))


def make_tall_dense(m=8, n=5):
    """Create a tall (overdetermined) dense matrix."""
    np.random.seed(42)
    return np.random.rand(m, n)


def make_tall_sparse(m=8, n=5):
    """Create a tall (overdetermined) sparse matrix."""
    np.random.seed(42)
    return sp.csr_array(sp.random(m, n, density=0.5, format='csr', random_state=42))


def make_linear_operator_spd(n=5):
    """Create a LinearOperator from an SPD matrix."""
    A = make_spd_dense(n)
    return LinearOperator(shape=A.shape, matvec=lambda x: A @ x, rmatvec=lambda x: A.T @ x), A


def make_linear_operator_square(n=5):
    """Create a LinearOperator from a square matrix."""
    A = make_square_dense(n)
    return LinearOperator(shape=A.shape, matvec=lambda x: A @ x, rmatvec=lambda x: A.T @ x), A


def make_linear_operator_tall(m=8, n=5):
    """Create a LinearOperator from a tall matrix."""
    A = make_tall_dense(m, n)
    return LinearOperator(shape=A.shape, matvec=lambda x: A @ x, rmatvec=lambda x: A.T @ x), A


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def vector_1d():
    """1D test vector of size 5."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def vector_2d():
    """2D test vectors (multiple RHS) of shape (5, 3)."""
    return np.array([[1.0, 0.5, 0.2],
                     [2.0, 1.0, 0.4],
                     [3.0, 1.5, 0.6],
                     [4.0, 2.0, 0.8],
                     [5.0, 2.5, 1.0]])


@pytest.fixture
def vector_tall_1d():
    """1D test vector for tall matrices (size 8)."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@pytest.fixture
def vector_tall_2d():
    """2D test vectors for tall matrices (shape 8, 2)."""
    return np.array([[1.0, 0.5],
                     [2.0, 1.0],
                     [3.0, 1.5],
                     [4.0, 2.0],
                     [5.0, 2.5],
                     [6.0, 3.0],
                     [7.0, 3.5],
                     [8.0, 4.0]])


# =============================================================================
# Parametrized Test Cases
# =============================================================================

# Square matrices (SPD and general)
SQUARE_MATRICES = [
    pytest.param(make_spd_dense, id="spd_dense"),
    pytest.param(make_spd_sparse, id="spd_sparse"),
    pytest.param(make_square_dense, id="square_dense"),
    pytest.param(make_square_sparse, id="square_sparse"),
]

# SPD matrices only (for CG solver)
SPD_MATRICES = [
    pytest.param(make_spd_dense, id="spd_dense"),
    pytest.param(make_spd_sparse, id="spd_sparse"),
]

# Rectangular (tall) matrices
TALL_MATRICES = [
    pytest.param(make_tall_dense, id="tall_dense"),
    pytest.param(make_tall_sparse, id="tall_sparse"),
]


# =============================================================================
# DirectSolver Tests
# =============================================================================

class TestDirectSolver:
    """Tests for DirectSolver with various matrix types."""
    
    # -------------------------------------------------------------------------
    # Square systems (exact solve)
    # -------------------------------------------------------------------------
    
    @pytest.mark.parametrize("make_matrix", SQUARE_MATRICES)
    def test_square_1d(self, make_matrix, vector_1d):
        """Test exact solve with square matrix and 1D RHS."""
        A = make_matrix()
        x_true = vector_1d
        b = A @ x_true
        
        solver = DirectSolver()
        x = solver(A, b)
        
        assert x.shape == x_true.shape
        np.testing.assert_allclose(x, x_true, rtol=1e-10)
    
    @pytest.mark.parametrize("make_matrix", SQUARE_MATRICES)
    def test_square_2d(self, make_matrix, vector_2d):
        """Test exact solve with square matrix and multiple RHS."""
        A = make_matrix()
        X_true = vector_2d
        B = A @ X_true
        
        solver = DirectSolver()
        X = solver(A, B)
        
        assert X.shape == X_true.shape
        np.testing.assert_allclose(X, X_true, rtol=1e-8)
    
    # -------------------------------------------------------------------------
    # Rectangular systems (least squares)
    # -------------------------------------------------------------------------
    
    @pytest.mark.parametrize("make_matrix", TALL_MATRICES)
    def test_rectangular_1d(self, make_matrix, vector_tall_1d):
        """Test least squares with tall matrix and 1D RHS."""
        A = make_matrix()
        b = vector_tall_1d
        
        solver = DirectSolver()
        
        # Sparse matrices trigger a SparseEfficiencyWarning for least squares
        if sp.issparse(A):
            with pytest.warns(sp.SparseEfficiencyWarning):
                x = solver(A, b)
        else:
            x = solver(A, b)
        
        # Verify normal equations: A^T A x = A^T b
        assert x.shape == (A.shape[1],)
        AtA = A.T @ A
        Atb = A.T @ b
        if sp.issparse(AtA):
            AtA = AtA.toarray()
        if sp.issparse(Atb):
            Atb = Atb.toarray().flatten()
        np.testing.assert_allclose(AtA @ x, Atb, rtol=1e-8)
    
    @pytest.mark.parametrize("make_matrix", TALL_MATRICES)
    def test_rectangular_2d(self, make_matrix, vector_tall_2d):
        """Test least squares with tall matrix and multiple RHS."""
        A = make_matrix()
        B = vector_tall_2d
        
        solver = DirectSolver()
        
        # Sparse matrices trigger a SparseEfficiencyWarning for least squares
        if sp.issparse(A):
            with pytest.warns(sp.SparseEfficiencyWarning):
                X = solver(A, B)
        else:
            X = solver(A, B)
        
        # Verify normal equations for each column
        assert X.shape == (A.shape[1], B.shape[1])
        AtA = A.T @ A
        AtB = A.T @ B
        if sp.issparse(AtA):
            AtA = AtA.toarray()
        if sp.issparse(AtB):
            AtB = AtB.toarray()
        np.testing.assert_allclose(AtA @ X, AtB, rtol=1e-8)
    
    # -------------------------------------------------------------------------
    # Error cases
    # -------------------------------------------------------------------------
    
    def test_underdetermined_raises(self):
        """Test that underdetermined systems raise ValueError."""
        A = np.array([[1.0, 0.0, 1.0],
                      [0.0, 1.0, 1.0]])  # 2x3 matrix
        b = np.array([1.0, 2.0])
        
        solver = DirectSolver()
        with pytest.raises(ValueError, match="[Uu]nderdetermined"):
            solver(A, b)
    
    def test_unsupported_type_raises(self):
        """Test that unsupported matrix types raise error."""
        A = [[1.0, 0.0], [0.0, 1.0]]  # list, not array
        b = np.array([1.0, 2.0])
        
        solver = DirectSolver()
        with pytest.raises((TypeError, AttributeError)):
            solver(A, b)


# =============================================================================
# IterativeSolver Tests
# =============================================================================

class TestIterativeSolverCG:
    """Tests for IterativeSolver with CG (spd=True)."""
    
    @pytest.mark.parametrize("make_matrix", SPD_MATRICES)
    def test_cg_1d(self, make_matrix, vector_1d):
        """Test CG with SPD matrix and 1D RHS."""
        A = make_matrix()
        x_true = vector_1d
        b = A @ x_true
        
        solver = IterativeSolver(spd=True, rtol=1e-10, atol=1e-12)
        x = solver(A, b)
        
        assert x.shape == x_true.shape
        np.testing.assert_allclose(x, x_true, rtol=1e-6)
    
    @pytest.mark.parametrize("make_matrix", SPD_MATRICES)
    def test_cg_2d(self, make_matrix, vector_2d):
        """Test CG with SPD matrix and multiple RHS."""
        A = make_matrix()
        X_true = vector_2d
        B = A @ X_true
        
        solver = IterativeSolver(spd=True, rtol=1e-10, atol=1e-12)
        X = solver(A, B)
        
        assert X.shape == X_true.shape
        np.testing.assert_allclose(X, X_true, rtol=1e-6)
    
    def test_cg_linear_operator_1d(self, vector_1d):
        """Test CG with LinearOperator and 1D RHS."""
        A_op, A = make_linear_operator_spd()
        x_true = vector_1d
        b = A @ x_true
        
        solver = IterativeSolver(spd=True, rtol=1e-10, atol=1e-12)
        x = solver(A_op, b)
        
        assert x.shape == x_true.shape
        np.testing.assert_allclose(x, x_true, rtol=1e-6)
    
    def test_cg_linear_operator_2d(self, vector_2d):
        """Test CG with LinearOperator and multiple RHS."""
        A_op, A = make_linear_operator_spd()
        X_true = vector_2d
        B = A @ X_true
        
        solver = IterativeSolver(spd=True, rtol=1e-10, atol=1e-12)
        X = solver(A_op, B)
        
        assert X.shape == X_true.shape
        np.testing.assert_allclose(X, X_true, rtol=1e-6)
    
    def test_cg_requires_square(self, vector_tall_1d):
        """Test that CG raises error for non-square matrices."""
        A = make_tall_dense()
        
        solver = IterativeSolver(spd=True)
        with pytest.raises(ValueError, match="[Ss]quare"):
            solver(A, vector_tall_1d)


class TestIterativeSolverGMRES:
    """Tests for IterativeSolver with GMRES (spd=False, square)."""
    
    @pytest.mark.parametrize("make_matrix", SQUARE_MATRICES)
    def test_gmres_1d(self, make_matrix, vector_1d):
        """Test GMRES with square matrix and 1D RHS."""
        A = make_matrix()
        x_true = vector_1d
        b = A @ x_true
        
        solver = IterativeSolver(spd=False, rtol=1e-10, atol=1e-12)
        x = solver(A, b)
        
        assert x.shape == x_true.shape
        np.testing.assert_allclose(x, x_true, rtol=1e-6)
    
    @pytest.mark.parametrize("make_matrix", SQUARE_MATRICES)
    def test_gmres_2d(self, make_matrix, vector_2d):
        """Test GMRES with square matrix and multiple RHS."""
        A = make_matrix()
        X_true = vector_2d
        B = A @ X_true
        
        solver = IterativeSolver(spd=False, rtol=1e-10, atol=1e-12)
        X = solver(A, B)
        
        assert X.shape == X_true.shape
        np.testing.assert_allclose(X, X_true, rtol=1e-6)
    
    def test_gmres_linear_operator_1d(self, vector_1d):
        """Test GMRES with LinearOperator and 1D RHS."""
        A_op, A = make_linear_operator_square()
        x_true = vector_1d
        b = A @ x_true
        
        solver = IterativeSolver(spd=False, rtol=1e-10, atol=1e-12)
        x = solver(A_op, b)
        
        assert x.shape == x_true.shape
        np.testing.assert_allclose(x, x_true, rtol=1e-6)


class TestIterativeSolverLSMR:
    """Tests for IterativeSolver with LSMR (rectangular systems)."""
    
    @pytest.mark.parametrize("make_matrix", TALL_MATRICES)
    def test_lsmr_1d(self, make_matrix, vector_tall_1d):
        """Test LSMR with tall matrix and 1D RHS."""
        A = make_matrix()
        b = vector_tall_1d
        
        solver = IterativeSolver(spd=False, atol=1e-6, btol=1e-6, conlim=1e8)
        x = solver(A, b)
        
        # Verify approximate normal equations
        assert x.shape == (A.shape[1],)
        AtA = A.T @ A
        Atb = A.T @ b
        if sp.issparse(AtA):
            AtA = AtA.toarray()
        if sp.issparse(Atb):
            Atb = Atb.toarray().flatten()
        np.testing.assert_allclose(AtA @ x, Atb, rtol=1e-4)
    
    @pytest.mark.parametrize("make_matrix", TALL_MATRICES)
    def test_lsmr_2d(self, make_matrix, vector_tall_2d):
        """Test LSMR with tall matrix and multiple RHS."""
        A = make_matrix()
        B = vector_tall_2d
        
        solver = IterativeSolver(spd=False, atol=1e-6, btol=1e-6, conlim=1e8)
        X = solver(A, B)
        
        assert X.shape == (A.shape[1], B.shape[1])
    
    def test_lsmr_linear_operator_1d(self, vector_tall_1d):
        """Test LSMR with LinearOperator and 1D RHS."""
        A_op, A = make_linear_operator_tall()
        b = vector_tall_1d
        
        solver = IterativeSolver(spd=False, atol=1e-6, btol=1e-6, conlim=1e8)
        x = solver(A_op, b)
        
        assert x.shape == (A.shape[1],)
        # Verify approximate normal equations
        np.testing.assert_allclose(A.T @ A @ x, A.T @ b, rtol=1e-4)


# =============================================================================
# IterativeSolver Default Options Tests
# =============================================================================

class TestIterativeSolverDefaults:
    """Tests for IterativeSolver with default options (no arguments)."""
    
    def test_default_square_1d(self, vector_1d):
        """Test default solver with square matrix and 1D RHS (uses GMRES)."""
        A = make_square_dense()
        b = vector_1d
        
        solver = IterativeSolver()  # No arguments - uses defaults
        x = solver(A, b)
        
        assert x.shape == (A.shape[0],)
        np.testing.assert_allclose(A @ x, b, rtol=1e-4)
    
    def test_default_square_2d(self, vector_2d):
        """Test default solver with square matrix and multiple RHS (uses GMRES)."""
        A = make_square_dense()
        B = vector_2d
        
        solver = IterativeSolver()
        X = solver(A, B)
        
        assert X.shape == (A.shape[0], B.shape[1])
        np.testing.assert_allclose(A @ X, B, rtol=1e-4)
    
    def test_default_square_sparse(self, vector_1d):
        """Test default solver with sparse square matrix (uses GMRES)."""
        A = make_square_sparse()
        b = vector_1d
        
        solver = IterativeSolver()
        x = solver(A, b)
        
        assert x.shape == (A.shape[0],)
        np.testing.assert_allclose(A @ x, b, rtol=1e-4)
    
    def test_default_spd_matrix(self, vector_1d):
        """Test default solver with SPD matrix (uses GMRES since spd=False by default)."""
        A = make_spd_dense()
        b = vector_1d
        
        solver = IterativeSolver()  # spd=False by default, so uses GMRES
        x = solver(A, b)
        
        assert x.shape == (A.shape[0],)
        np.testing.assert_allclose(A @ x, b, rtol=1e-4)
    
    def test_default_rectangular_1d(self, vector_tall_1d):
        """Test default solver with rectangular matrix and 1D RHS (uses LSMR)."""
        A = make_tall_dense()
        b = vector_tall_1d
        
        solver = IterativeSolver()  # Rectangular, so uses LSMR
        x = solver(A, b)
        
        assert x.shape == (A.shape[1],)
        # Verify normal equations are approximately satisfied
        np.testing.assert_allclose(A.T @ A @ x, A.T @ b, rtol=1e-4)
    
    def test_default_rectangular_2d(self, vector_tall_2d):
        """Test default solver with rectangular matrix and multiple RHS (uses LSMR)."""
        A = make_tall_dense()
        B = vector_tall_2d
        
        solver = IterativeSolver()
        X = solver(A, B)
        
        assert X.shape == (A.shape[1], B.shape[1])
    
    def test_default_rectangular_sparse(self, vector_tall_1d):
        """Test default solver with sparse rectangular matrix (uses LSMR)."""
        A = make_tall_sparse()
        b = vector_tall_1d
        
        solver = IterativeSolver()
        x = solver(A, b)
        
        assert x.shape == (A.shape[1],)
    
    def test_default_linear_operator_square(self, vector_1d):
        """Test default solver with square LinearOperator (uses GMRES)."""
        A_op, A = make_linear_operator_square()
        b = vector_1d
        
        solver = IterativeSolver()
        x = solver(A_op, b)
        
        assert x.shape == (A.shape[0],)
        np.testing.assert_allclose(A @ x, b, rtol=1e-4)
    
    def test_default_linear_operator_tall(self, vector_tall_1d):
        """Test default solver with tall LinearOperator (uses LSMR)."""
        A_op, A = make_linear_operator_tall()
        b = vector_tall_1d
        
        solver = IterativeSolver()
        x = solver(A_op, b)
        
        assert x.shape == (A.shape[1],)
        np.testing.assert_allclose(A.T @ A @ x, A.T @ b, rtol=1e-4)


# =============================================================================
# wrap_solver Tests
# =============================================================================

class TestWrapSolver:
    """Tests for wrap_solver utility."""
    
    def test_passthrough_solver(self):
        """Test that Solver instances are returned unchanged."""
        original = DirectSolver()
        wrapped = wrap_solver(original)
        
        assert wrapped is original
        assert isinstance(wrapped, Solver)
    
    def test_wrap_callable_1d(self, vector_1d):
        """Test wrapping a callable that solves Ax = b."""
        def my_solver(A, b, x0=None):
            return np.linalg.solve(A, b)
        
        A = make_square_dense()
        x_true = vector_1d
        b = A @ x_true
        
        solver = wrap_solver(my_solver)
        x = solver(A, b)
        
        assert isinstance(solver, Solver)
        np.testing.assert_allclose(x, x_true, rtol=1e-10)
    
    def test_wrap_callable_2d(self, vector_2d):
        """Test wrapped callable handles multiple RHS correctly."""
        def my_solver(A, b, x0=None):
            return np.linalg.solve(A, b)
        
        A = make_square_dense()
        X_true = vector_2d
        B = A @ X_true
        
        solver = wrap_solver(my_solver)
        X = solver(A, B)
        
        assert X.shape == X_true.shape
        np.testing.assert_allclose(X, X_true, rtol=1e-10)
    
    def test_wrap_callable_1d_only(self, vector_2d):
        """Test wrapped callable that only handles 1D vectors."""
        def my_1d_solver(A, b, x0=None):
            if b.ndim != 1:
                raise ValueError("Only 1D supported")
            return np.linalg.solve(A, b)
        
        A = make_square_dense()
        X_true = vector_2d
        B = A @ X_true
        
        solver = wrap_solver(my_1d_solver)
        X = solver(A, B)
        
        assert X.shape == X_true.shape
        np.testing.assert_allclose(X, X_true, rtol=1e-10)


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
