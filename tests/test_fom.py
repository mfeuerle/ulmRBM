"""
Test suite for the fom (Full-Order Model) module.

Tests cover:
- FOM and GalerkinFOM initialization and properties
- solve method correctness
- stability and continuity constants (explicit and computed)
- supremizer functionality
- error handling for dimension mismatches
- solver property management
"""

import pytest
import numpy as np
import scipy as sp
from scipy.sparse import csr_array
from scipy.sparse.linalg import aslinearoperator

from ulmRBM.fom import FOM, GalerkinFOM
from ulmRBM.affine import AffineLinear
from ulmRBM.products import EuclideanInnerProduct, MatrixInnerProduct
from ulmRBM.solver import Solver, DirectSolver, IterativeSolver

# =============================================================================
# Fixtures for Test Data
# =============================================================================

matrix_classes = [np.array, csr_array, aslinearoperator]
U_dim = 2
V_dim = 2

@pytest.fixture(params=matrix_classes)
def affine_matrix(request):
    """Affine matrix B(mu) = mu * A + (1-mu) * I."""
    matrix_class = request.param
    A = matrix_class(np.array([[2.0, 1.0], [1.0, 3.0]]))
    I = matrix_class(np.eye(2))
    return  AffineLinear([lambda mu: mu, lambda mu: 1-mu], [A, I])

@pytest.fixture(params=matrix_classes)
def simple_matrix(request):
    """Simple 2x2 matrix for basic tests."""
    matrix_class = request.param
    return matrix_class(np.array([[2.0, 1.0], [1.0, 3.0]]))


@pytest.fixture
def affine_vector():
    """Affine vector f(mu) = mu * b1 + (1-mu) * b2."""
    b1 = np.array([1.0, 2.0])
    b2 = np.array([0.5, 1.5])
    return AffineLinear([lambda mu: mu, lambda mu: 1-mu], [b1, b2])

@pytest.fixture
def simple_vector():
    """Simple 2D vector for basic tests."""
    return np.array([1.0, 2.0])


@pytest.fixture(params=matrix_classes)
def param_product_U(request):
    """Standard inner product matrix."""
    matrix_class = request.param
    A1 = matrix_class(np.array([[1.0, 0.0], [0.0, 0.0]]))
    A2 = matrix_class(np.array([[0.0, 0.0], [0.0, 1.0]]))
    A =AffineLinear([lambda mu: abs(mu)+1.0, lambda mu: mu**2+1.0], [A1, A2])
    return MatrixInnerProduct(A)

@pytest.fixture(params=matrix_classes)
def noparam_product_U(request):
    """Standard inner product matrix."""
    matrix_class = request.param
    return MatrixInnerProduct(matrix_class(np.array([[2.0, 1.0], [1.0, 3.0]])))

@pytest.fixture
def identity_product_U():
    """Identity inner product."""
    return EuclideanInnerProduct(2)


@pytest.fixture(params=matrix_classes)
def param_product_V(request):
    """Standard inner product matrix."""
    matrix_class = request.param
    A1 = matrix_class(np.array([[1.0, 0.0], [0.0, 0.0]]))
    A2 = matrix_class(np.array([[0.0, 0.0], [0.0, 1.0]]))
    A =AffineLinear([lambda mu: mu**2+1.0, lambda mu: abs(mu)+1.0], [A1, A2])
    return MatrixInnerProduct(A)

@pytest.fixture(params=matrix_classes)
def noparam_product_V(request):
    """Standard inner product matrix."""
    matrix_class = request.param
    return MatrixInnerProduct(matrix_class(np.array([[3.0, 1.0], [1.0, 2.0]])))

@pytest.fixture
def identity_product_V():
    """Identity inner product."""
    return EuclideanInnerProduct(2)


@pytest.fixture
def pg_fom(affine_matrix, affine_vector, param_product_U, param_product_V):
    """Petrov-Galerkin FOM with different trial and test spaces."""
    return FOM(affine_matrix, affine_vector, param_product_U, param_product_V)

@pytest.fixture
def pg_fom_noparam_V(affine_matrix, affine_vector, param_product_U, noparam_product_V):
    """Petrov-Galerkin FOM with different trial and test spaces."""
    return FOM(affine_matrix, affine_vector, param_product_U, noparam_product_V)

@pytest.fixture
def g_fom(affine_matrix, affine_vector, param_product_U):
    """GalerkinFOM instance."""
    return GalerkinFOM(affine_matrix, affine_vector, param_product_U)



test_parameters = [0.0, 0.3, 0.5, 0.8, 1.0]


# =============================================================================
# Test FOM Initialization and Properties
# =============================================================================

class TestFOMInitialization:
    """Tests for FOM initialization and basic properties."""

    def test_basic_initialization(self, affine_matrix, affine_vector, param_product_U, param_product_V):
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_V)
        assert fom.dim == (U_dim, V_dim)
        assert fom.U is param_product_U
        assert fom.V is param_product_V
        assert isinstance(fom.solver, Solver)

    def test_matrix_vector_compatibility(self, affine_matrix, affine_vector, param_product_U, param_product_V):
        """Test that matrix-vector operations work consistently."""
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_V)
        E = np.eye(U_dim)
        
        for mu in test_parameters:
            B_mu = fom.B(mu)
            f_mu = fom.f(mu)
            assert B_mu.shape == (U_dim, V_dim)
            assert f_mu.shape == (V_dim,)
            assert np.allclose(B_mu@E, affine_matrix(mu)@E)
            assert np.allclose(f_mu, affine_vector(mu))

    def test_custom_solver(self, affine_matrix, affine_vector, param_product_U, param_product_V):
        """Test initialization with custom solver."""
        solver = DirectSolver()
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_V, solver=solver)
        assert fom.solver is solver

    def test_callable_solver_wrapping(self, affine_matrix, affine_vector, param_product_U, param_product_V):
        """Test that callable solvers are properly wrapped."""
        def custom_solve(A, b):
            return np.linalg.solve(A, b)
        
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_V, solver=custom_solve)
        assert isinstance(fom.solver, Solver)

    def test_dimension_validation(self, affine_matrix, affine_vector, param_product_U, param_product_V):
        """Test dimension mismatch error handling."""
        
        f_wrong = np.ones(V_dim+1)
        with pytest.raises(ValueError, match="B and f must have compatible dimensions"):
            FOM(affine_matrix, f_wrong, param_product_U, param_product_V)
            
        U_wrong = MatrixInnerProduct(np.eye(U_dim+1))
        with pytest.raises(ValueError, match="B and U must have compatible dimensions"):
            FOM(affine_matrix, affine_vector, U_wrong, param_product_V)
            
        V_wrong = MatrixInnerProduct(np.eye(V_dim+1))
        with pytest.raises(ValueError, match="B and V must have compatible dimensions"):
            FOM(affine_matrix, affine_vector, param_product_U, V_wrong)
            
    def test_non_affine_to_affine_conversion(self, simple_matrix, simple_vector, param_product_U, param_product_V):
        """Test that non-affine inputs are converted to affine."""
        fom = FOM(simple_matrix, simple_vector, param_product_U, param_product_V)
        E = np.eye(U_dim)
        # Should be wrapped as AffineLinear
        assert isinstance(fom.B, AffineLinear)
        assert isinstance(fom.f, AffineLinear)
        
        # Should give same results as original
        mu = 0.5  # Parameter shouldn't matter for non-parametric case
        assert np.allclose(fom.B(mu)@E, simple_matrix@E)
        assert np.allclose(fom.f(mu), simple_vector)


# =============================================================================
# Test GalerkinFOM Specific Features
# =============================================================================

class TestGalerkinFOM:
    """Tests for GalerkinFOM-specific functionality."""
    
    def test_galerkin_initialization_equivalence(self, affine_matrix, affine_vector, param_product_U):
        """Test that GalerkinFOM is equivalent to FOM with U=V."""
        galerkin = GalerkinFOM(affine_matrix, affine_vector, param_product_U)
        petrov = FOM(affine_matrix, affine_vector, param_product_U, param_product_U)
        
        assert galerkin.U is petrov.U
        assert galerkin.V is petrov.V
        assert galerkin.dim == petrov.dim
        
        E = np.eye(galerkin.dim[0])
        for mu in test_parameters:
            assert np.allclose(galerkin.B(mu)@E, petrov.B(mu)@E)
            assert np.allclose(galerkin.f(mu), petrov.f(mu))

    def test_galerkin_v_property(self, g_fom):
        """Test that V property always returns U for Galerkin models."""
        assert g_fom.V is g_fom.U

    def test_galerkin_v_setter_restriction(self, g_fom, param_product_V):
        """Test that V cannot be set to different value than U."""
        with pytest.raises(AttributeError, match="Cannot set V for Galerkin models"):
            g_fom.V = param_product_V



# =============================================================================
# Test solve Method
# =============================================================================

class TestSolveMethod:
    """Tests for the solve method."""

    def test_solve_correctness(self, pg_fom):
        """Test that solve method produces correct solutions."""
        for mu in test_parameters:
            u = pg_fom.solve(mu)
            B_mu = pg_fom.B(mu)
            f_mu = pg_fom.f(mu)
            
            # Check that B(mu) * u = f(mu)
            residual = B_mu @ u - f_mu
            assert np.allclose(residual, 0, atol=1e-12)

    def test_solve_galerkin_vs_petrov_galerkin(self, affine_matrix, affine_vector, param_product_U):
        """Test solve consistency between Galerkin and equivalent Petrov-Galerkin."""
        galerkin = GalerkinFOM(affine_matrix, affine_vector, param_product_U)
        petrov = FOM(affine_matrix, affine_vector, param_product_U, param_product_U)
        
        for mu in [0.2, 0.7]:
            u_galerkin = galerkin.solve(mu)
            u_petrov = petrov.solve(mu)
            assert np.allclose(u_galerkin, u_petrov)

    def test_solve_different_solvers(self, affine_matrix, affine_vector, param_product_U):
        """Test solve with different solver types."""
        fom_direct = FOM(affine_matrix, affine_vector, param_product_U, param_product_U, 
                        solver=lambda A, b, x0: sp.sparse.linalg.lsmr(A, b, atol=1e-12, btol=1e-12)[0])
        fom_iter = FOM(affine_matrix, affine_vector, param_product_U, param_product_U, 
                      solver=IterativeSolver(spd=False, rtol=1e-12, atol=1e-12, btol=1e-12))
        
        mu = 0.5
        u_direct = fom_direct.solve(mu)
        u_iter = fom_iter.solve(mu)
        
        assert np.allclose(u_direct, u_iter, rtol=1e-10)
        
    def test_solver_setter_solver_object(self, pg_fom):
        """Test setting solver with Solver object."""
        new_solver = DirectSolver()
        pg_fom.solver = new_solver
        assert pg_fom.solver is new_solver

    def test_solver_setter_callable_wrapping(self, pg_fom):
        """Test setting solver with callable (should be wrapped)."""
        def custom_solve(A, b):
            return np.linalg.solve(A, b)
        
        pg_fom.solver = custom_solve
        assert isinstance(pg_fom.solver, Solver)


# =============================================================================
# Test Stability and Continuity Constants
# =============================================================================

class TestStabilityContinuity:
    """Tests for stability and continuity constant computation."""

    def test_explicit_stability_function(self, affine_matrix, affine_vector, param_product_U):
        """Test stability with explicit function."""
        def stability_func(mu, fom):
            return mu + 1.0
            
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_U,
                 stability=stability_func)
        assert fom.stability(0.5) == 1.5
        assert fom.stability(2.0) == 3.0

    def test_explicit_continuity_function(self, affine_matrix, affine_vector, param_product_U):
        """Test continuity with explicit function."""
        def continuity_func(mu, fom):
            return mu * 2.0
            
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_U,
                 continuity=continuity_func)
        assert fom.continuity(0.5) == 1.0
        assert fom.continuity(2.0) == 4.0

    def test_explicit_stability_values(self, affine_matrix, affine_vector, param_product_U):
        """Test stability/continuity with constant values."""
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_U, stability=2.5)
        assert fom.stability(0.1) == 2.5
        assert fom.stability(1.0) == 2.5
        
    def test_explicit_stability_values(self, affine_matrix, affine_vector, param_product_U):
        """Test stability/continuity with constant values."""
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_U, continuity=3.7)
        assert fom.continuity(0.1) == 3.7
        assert fom.continuity(1.0) == 3.7

    def test_computed_constants_diagonal_case(self, affine_vector, identity_product_U):
        """Test computed constants for simple diagonal case."""
        # Diagonal matrix with eigenvalues mu and 1/mu
        B = AffineLinear([lambda mu: mu**2, lambda mu: 1/mu], 
                        [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
        
        petrov = FOM(B, affine_vector, identity_product_U, identity_product_U)
        galerkin = GalerkinFOM(B, affine_vector, identity_product_U)

        # For diagonal matrices, stability is min eigenvalue, continuity is max eigenvalue     
        assert np.isclose(petrov.stability(0.1), 0.01)
        assert np.isclose(petrov.continuity(0.1), 10.0)
        assert np.isclose(galerkin.stability(0.1), 0.01)
        assert np.isclose(galerkin.continuity(0.1), 10.0)
        
        assert np.isclose(petrov.stability(1.0), 1.0)
        assert np.isclose(petrov.continuity(1.0), 1.0)
        assert np.isclose(galerkin.stability(1.0), 1.0)
        assert np.isclose(galerkin.continuity(1.0), 1.0)
        
        assert np.isclose(petrov.stability(10.0), 0.1)
        assert np.isclose(petrov.continuity(10.0), 100.0)
        assert np.isclose(galerkin.stability(10.0), 0.1)
        assert np.isclose(galerkin.continuity(10.0), 100.0)

    def test_computed_constants_optimal_petrov(self, affine_matrix, affine_vector, param_product_U):
        """Test computed constant for Petrov-Galerkin."""
        inner_product_U = param_product_U.dual.restrict(affine_matrix)
        fom = FOM(affine_matrix, affine_vector, U=inner_product_U, V=param_product_U)
        
        assert np.isclose(fom.stability(0.1), 1)
        assert np.isclose(fom.continuity(0.1), 1)
        
        assert np.isclose(fom.stability(1), 1)
        assert np.isclose(fom.continuity(1), 1)
        
        assert np.isclose(fom.stability(10), 1)
        assert np.isclose(fom.continuity(10), 1)

# =============================================================================
# Test Supremizer Functionality
# =============================================================================

class TestSupremizer:
    """Tests for supremizer functionality."""

    def test_custom_supremizer_function(self, affine_matrix, affine_vector, param_product_U):
        """Test supremizer with custom function."""
        def custom_supremizer(u, fom):
            return lambda mu: 2 * u  # Simple custom implementation
            
        fom = FOM(affine_matrix, affine_vector, param_product_U, param_product_U,
                 supremizer=custom_supremizer)
        
        u = np.array([1.0, 2.0])
        result = fom.supremizer(u)
        
        # Should use custom function
        assert np.allclose(result(0.5), 2 * u)

    def test_default_supremizer_parameter_independent(self, affine_matrix, affine_vector, noparam_product_U):
        """Test default supremizer with parameter-independent inner product."""
        fom = FOM(affine_matrix, affine_vector, noparam_product_U, noparam_product_U)
        
        for u in [np.array([1.0, 0.0]), np.array([[1.0, 0.0], [2.0, 1.0]])]:
            supremizer = fom.supremizer(u)
            assert isinstance(supremizer, AffineLinear) # Should return AffineLinear for parameter-independent case
            for mu in test_parameters:
                expected = noparam_product_U.dual(mu) @ (affine_matrix(mu) @ u)
                assert np.allclose(supremizer(mu), expected)
                
    def test_default_supremizer_parameter_dependent(self, affine_matrix, affine_vector, param_product_U):
        inner_product_V = param_product_U.restrict(affine_matrix)
        fom = FOM(affine_matrix, affine_vector, param_product_U, inner_product_V)
        
        for u in [np.array([1.0, 0.0]), np.array([[1.0, 0.0], [2.0, 1.0]])]:
            supremizer = fom.supremizer(u)
            for mu in test_parameters:
                expected = inner_product_V.dual(mu) @ (affine_matrix(mu) @ u)
                assert np.allclose(supremizer(mu), expected)


if __name__ == "__main__":
    pytest.main([__file__])