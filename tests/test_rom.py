"""
Test suite for ROM module constants and residual classes.

Tests cover:
- Enum classes and their get() methods
- Constants estimators (StabilityEstimator, ContinuityEstimator)
- ResidualCalculator implementations (DirectResidual, AffineResidual)

Excludes ROM classes and their subclasses.
"""

import pytest
import numpy as np

from scipy.sparse import csr_array
from scipy.sparse.linalg import aslinearoperator

from ulmRBM.rom import (
    StabilityOptions, ContinuityOptions, ResidualOptions,
    StabilityExact, StabilityMinTheta,
    ContinuityExact, ContinuityMaxTheta,
    DirectResidual, AffineResidual,
    ROM, Trial2TestROM, GalerkinROM
)
from ulmRBM.fom import FOM, GalerkinFOM
from ulmRBM.affine import AffineLinear
from ulmRBM.products import MatrixInnerProduct, EuclideanInnerProduct
from ulmRBM.solver import DirectSolver


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

@pytest.fixture
def affine_vector():
    """Affine vector f(mu) = mu * b1 + (1-mu) * b2."""
    b1 = np.array([1.0, 2.0])
    b2 = np.array([0.5, 1.5])
    return AffineLinear([lambda mu: mu, lambda mu: 1-mu], [b1, b2])


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


@pytest.fixture
def simple_basis_U():
    """Simple basis matrix."""
    return np.array([[1.0, 0.0], [0.0, 1.0]])

@pytest.fixture
def simple_basis_V():
    """Simple basis matrix."""
    return np.array([[1.0, 0.5], [0.5, 1.0]])


test_parameters = [0.1, 0.3, 0.5, 0.8, 0.9]


# =============================================================================
# Test Enum Classes
# =============================================================================

class TestEnumOptions:
    """Tests for enum option classes and their get() methods."""

    def test_stability_options(self):
        """Test StabilityOptions.get() method."""
        assert StabilityOptions.MIN_THETA == 'min-theta'
        assert StabilityOptions.EXACT == 'exact'
        
        assert isinstance(StabilityOptions.EXACT.get(), StabilityExact)
        assert isinstance(StabilityOptions.MIN_THETA.get(), StabilityMinTheta)

    def test_continuity_options(self):
        """Test ContinuityOptions.get() method."""
        assert ContinuityOptions.MAX_THETA == 'max-theta'
        assert ContinuityOptions.EXACT == 'exact'
        
        assert isinstance(ContinuityOptions.EXACT.get(), ContinuityExact)
        assert isinstance(ContinuityOptions.MAX_THETA.get(), ContinuityMaxTheta)

    def test_residual_options(self):
        """Test ResidualOptions.get() method."""
        assert ResidualOptions.DIRECT == 'direct'
        assert ResidualOptions.AFFINE == 'affine'
        
        assert isinstance(ResidualOptions.DIRECT.get(), DirectResidual)
        assert isinstance(ResidualOptions.AFFINE.get(), AffineResidual)
        
        
# =============================================================================
# Test Estimators
# =============================================================================

@pytest.mark.parametrize("estimator_class", [StabilityExact, StabilityMinTheta, ContinuityExact, ContinuityMaxTheta])
class TestConstantsEstimators:
    """Tests for stability and continuity constant estimators."""
    
    def test_empty_initialization(self, estimator_class, pg_fom):
        estimator = estimator_class()
        with pytest.raises(ValueError, match="FOM has not been set"):
            _ = estimator.fom
        # Set FOM
        estimator.fom = pg_fom
        assert estimator.fom is pg_fom
        
    def test_initialization(self, estimator_class, pg_fom):
        estimator = estimator_class(pg_fom)
        assert estimator.fom is pg_fom
    
    def test_estimator_fom_property_management(self, estimator_class, pg_fom):
        estimator = estimator_class(pg_fom)
        # set with same fom
        estimator.fom = pg_fom
        # set with different fom
        with pytest.raises(ValueError, match="cannot be changed"):
            estimator.fom = pg_fom_noparam_V


# =============================================================================
# Test Stability Estimators

@pytest.mark.parametrize("estimator_class", [StabilityExact, StabilityMinTheta])
class TestStabilityEstimators:
    """Tests for stability constant estimators."""
    
    def test_update(self, estimator_class, pg_fom):
        estimator = estimator_class(pg_fom)
        # Update with some parameters
        for mu in test_parameters[:3]:
            estimator.update(mu)
        # reproduce added parameters
        for mu in test_parameters[:3]:
            assert np.isclose(estimator(mu), estimator.fom.stability(mu))
        # lower bound for the rest
        for mu in test_parameters:
            assert 0 < estimator(mu) <= estimator.fom.stability(mu) + 1e-10
            
# =============================================================================
# Test Continuity Estimators

@pytest.mark.parametrize("estimator_class", [ContinuityExact, ContinuityMaxTheta])
class TestContinuityEstimators:
    """Tests for continuity constant estimators."""
    
    def test_update(self, estimator_class, pg_fom):
        estimator = estimator_class(pg_fom)
        # Update with some parameters
        for mu in test_parameters[:3]:
            estimator.update(mu)
        # reproduce added parameters
        for mu in test_parameters[:3]:
            assert np.isclose(estimator(mu), estimator.fom.continuity(mu))
        # upper bound for the rest
        for mu in test_parameters:
            assert estimator.fom.continuity(mu) - 1e-10 <= estimator(mu) < np.inf


# =============================================================================
# Test Exact Estimators

class TestConstantsExact:
    """Tests for unique exact computiation functionality."""
    
    def test_stability_exact(self, pg_fom):
        estimator = StabilityExact(pg_fom)
        for mu in test_parameters:
            assert np.isclose(estimator(mu), pg_fom.stability(mu))
            
    def test_continuity_exact(self, pg_fom):
        estimator = ContinuityExact(pg_fom)
        for mu in test_parameters:
            assert np.isclose(estimator(mu), pg_fom.continuity(mu))
            
            
# =============================================================================
# Test Min Theta Estimators

@pytest.mark.parametrize("estimator_class", [StabilityMinTheta, ContinuityMaxTheta])
class TestConstantsMinTheta:
    """Tests for unique for min/max-theta functionality."""
    
    def test_requires_positive_theta(self, estimator_class):
        # Create FOM with potentially negative theta
        B_bad = AffineLinear([lambda mu: mu - 1.0], [np.eye(2)])
        f = AffineLinear([lambda mu: 1.0], [np.array([1.0, 1.0])])
        U = EuclideanInnerProduct(2)
        fom_bad = FOM(B_bad, f, U, U)
        estimator = estimator_class(fom_bad)
        with pytest.raises(ValueError, match="positive theta functions"):
            estimator.update(0.5)  # mu - 1 = -0.5 < 0
            
    def test_without_update(self, estimator_class, pg_fom):
        estimator = estimator_class(pg_fom)
        with pytest.raises(RuntimeError, match="No precomputed constants"):
            estimator(0.5)



# =============================================================================
# Test ResidualCalculator
# =============================================================================

class _TestResidualCalculator:
    """Tests for ResidualCalculator implementations."""

    residual_calculator = None  # To be defined in subclasses
    used_fom = None  # To be defined in subclasses
    
    @staticmethod
    def exact_residual_dual_norm(fom, basis, mu, u):        
        """Compute exact residual dual norm for comparison."""
        r = fom.f(mu) - fom.B(mu) @ (basis @ u)
        return fom.V.dual.norm(mu, r)

    def test_set_with_fom(self, residual_calculator, used_fom):
        residual_calculator.set(used_fom)
        assert residual_calculator.fom is used_fom

    def test_set_with_fom_and_basis(self, residual_calculator, used_fom, simple_basis_U):
        residual_calculator.set(used_fom, simple_basis_U)
        assert residual_calculator.fom is used_fom
    
    @pytest.mark.skip(reason="set() with ROM not implemented yet.")
    def test_set_with_rom(self):
        raise NotImplementedError("Test for set() with ROM not implemented yet.")
        
    def test_set_then_dual_norm(self, residual_calculator, used_fom, simple_basis_U):
        residual_calculator.set(used_fom, simple_basis_U)
        
        for mu in test_parameters:
            for u in [np.array([0.1, 0.2]), np.array([0.5, 0.5])]:
                result = residual_calculator.dual_norm(mu, u)
                exact = self.exact_residual_dual_norm(used_fom, simple_basis_U, mu, u)
                assert np.isclose(result, exact)

    def test_add_then_dual_norm(self, residual_calculator, used_fom, simple_basis_U):
        residual_calculator.set(used_fom, simple_basis_U[:, :1])  # First basis vector
        residual_calculator.add_basis(simple_basis_U[:, 1:])       # remaining basis vector
        
        for mu in test_parameters:
            for u in [np.array([0.1, 0.2]), np.array([0.5, 0.5])]:
                result = residual_calculator.dual_norm(mu, u)
                exact = self.exact_residual_dual_norm(used_fom, simple_basis_U, mu, u)
                assert np.isclose(result, exact)

    def test_rotate_then_dual_norm(self, residual_calculator, used_fom, simple_basis_U):
        residual_calculator.set(used_fom, simple_basis_U)
        rotation = np.array([[0.0, 1.0], [1.0, 0.0]])  # Swap columns
        residual_calculator.rotate_basis(rotation)
        rotated_basis = simple_basis_U @ rotation
        
        for mu in test_parameters:
            for u in [np.array([0.1, 0.2]), np.array([0.5, 0.5])]:
                result = residual_calculator.dual_norm(mu, u)
                exact = self.exact_residual_dual_norm(used_fom, rotated_basis, mu, u)
                assert np.isclose(result, exact)

    def test_fom_property_error_handling(self, residual_calculator):
        """Test FOM property error when not set."""
        with pytest.raises(ValueError, match="FOM has not been set"):
            _ = residual_calculator.fom
            
# ============================================================================
# Test DirectResidual Specifics

class TestDirectResidual(_TestResidualCalculator):
    """Tests specific to DirectResidual."""
    
    @pytest.fixture
    def residual_calculator(self):
        return DirectResidual()
    
    @pytest.fixture
    def used_fom(self, pg_fom):
        return pg_fom

# =============================================================================
# Test AffineResidual Specifics

class TestAffineResidual(_TestResidualCalculator):
    """Tests specific to AffineResidual."""
    
    @pytest.fixture
    def residual_calculator(self):
        return AffineResidual()
    
    @pytest.fixture
    def used_fom(self, pg_fom_noparam_V):
        return pg_fom_noparam_V
    
    def test_affine_residual_requires_parameter_independent_inner_product(self, residual_calculator,  pg_fom):
        """Test that AffineResidual requires parameter-independent inner product."""
        
        with pytest.raises(ValueError, match="parameter-independent test space inner product"):
            residual_calculator.set(pg_fom)












# U_dim = 2
# V_dim = 2

# @pytest.fixture(params=matrix_classes)
# def affine_matrix(request):
#     """Affine matrix B(mu) = mu * A + (1-mu) * I."""
#     matrix_class = request.param
#     A = matrix_class(np.array([[2.0, 1.0], [1.0, 3.0]]))
#     I = matrix_class(np.eye(2))
#     return  AffineLinear([lambda mu: mu, lambda mu: 1-mu], [A, I])

# @pytest.fixture
# def affine_vector():
#     """Affine vector f(mu) = mu * b1 + (1-mu) * b2."""
#     b1 = np.array([1.0, 2.0])
#     b2 = np.array([0.5, 1.5])
#     return AffineLinear([lambda mu: mu, lambda mu: 1-mu], [b1, b2])


# @pytest.fixture(params=matrix_classes)
# def param_product_U(request):
#     """Standard inner product matrix."""
#     matrix_class = request.param
#     A1 = matrix_class(np.array([[1.0, 0.0], [0.0, 0.0]]))
#     A2 = matrix_class(np.array([[0.0, 0.0], [0.0, 1.0]]))
#     A =AffineLinear([lambda mu: abs(mu)+1.0, lambda mu: mu**2+1.0], [A1, A2])
#     return MatrixInnerProduct(A)

# @pytest.fixture(params=matrix_classes)
# def noparam_product_U(request):
#     """Standard inner product matrix."""
#     matrix_class = request.param
#     return MatrixInnerProduct(matrix_class(np.array([[2.0, 1.0], [1.0, 3.0]])))

# @pytest.fixture
# def identity_product_U():
#     """Identity inner product."""
#     return EuclideanInnerProduct(2)


# @pytest.fixture(params=matrix_classes)
# def param_product_V(request):
#     """Standard inner product matrix."""
#     matrix_class = request.param
#     A1 = matrix_class(np.array([[1.0, 0.0], [0.0, 0.0]]))
#     A2 = matrix_class(np.array([[0.0, 0.0], [0.0, 1.0]]))
#     A =AffineLinear([lambda mu: mu**2+1.0, lambda mu: abs(mu)+1.0], [A1, A2])
#     return MatrixInnerProduct(A)

# @pytest.fixture(params=matrix_classes)
# def noparam_product_V(request):
#     """Standard inner product matrix."""
#     matrix_class = request.param
#     return MatrixInnerProduct(matrix_class(np.array([[3.0, 1.0], [1.0, 2.0]])))

# @pytest.fixture
# def identity_product_V():
#     """Identity inner product."""
#     return EuclideanInnerProduct(2)


# @pytest.fixture
# def pg_fom(affine_matrix, affine_vector, param_product_U, param_product_V):
#     """Petrov-Galerkin FOM with different trial and test spaces."""
#     return FOM(affine_matrix, affine_vector, param_product_U, param_product_V)

# @pytest.fixture
# def pg_fom_noparam_V(affine_matrix, affine_vector, param_product_U, noparam_product_V):
#     """Petrov-Galerkin FOM with different trial and test spaces."""
#     return FOM(affine_matrix, affine_vector, param_product_U, noparam_product_V)

# @pytest.fixture
# def g_fom(affine_matrix, affine_vector, param_product_U):
#     """GalerkinFOM instance."""
#     return GalerkinFOM(affine_matrix, affine_vector, param_product_U)


# @pytest.fixture
# def simple_basis_U():
#     """Simple basis matrix."""
#     return np.array([[1.0, 0.0], [0.0, 1.0]])

# @pytest.fixture
# def simple_basis_V():
#     """Simple basis matrix."""
#     return np.array([[1.0, 0.5], [0.5, 1.0]])


# # =============================================================================
# # Test ROM Classes
# # =============================================================================

# #=============================================================================
# # Base functionalities for all ROM classes

# class _TestROMBase:
#     """Base class for ROM testing with common functionality."""
    
#     # To be overridden in subclasses via fixtures
#     rom_class = None  
#     used_fom = None
    
#     def test_initialization_empty(self, rom_class, used_fom):
#         """Test ROM initialization without basis."""
#         rom = rom_class(used_fom)
        
#         # Base assertions - only check basic properties
#         assert rom.fom is used_fom
#         assert rom.U_basis is None
#         assert rom.dim == 0
        
#         # Return rom for subclass extensions
#         return rom
        
#     def test_initialization_with_basis(self, rom_class, used_fom, trial_basis):
#         """Test ROM initialization with initial basis."""
#         rom = rom_class(used_fom, U_basis=trial_basis)
        
#         # Base assertions - only check U_basis
#         assert rom.fom is used_fom
#         assert np.allclose(rom.U_basis, trial_basis)
#         assert rom.dim == trial_basis.shape[1]
        
#         # Return rom and basis for subclass extensions
#         return rom, trial_basis
        
#     def test_add_basis_single_vector(self, rom_class, used_fom, single_basis):
#         """Test adding a single basis vector."""
#         rom = rom_class(used_fom)
#         rom.add_basis(single_basis)
        
#         # Base assertions - only check U_basis
#         assert rom.U_basis.shape == (3, 1)
#         assert np.allclose(rom.U_basis, single_basis)
#         assert rom.dim == 1
        
#         # Return rom and basis for subclass extensions
#         return rom, single_basis
        
#     def test_add_basis_multiple_vectors(self, rom_class, used_fom, trial_basis):
#         """Test adding multiple basis vectors."""
#         rom = rom_class(used_fom)
#         rom.add_basis(trial_basis)
        
#         # Base assertions - only check U_basis
#         assert rom.U_basis.shape == trial_basis.shape
#         assert np.allclose(rom.U_basis, trial_basis)
#         assert rom.dim == trial_basis.shape[1]
        
#         # Return rom and basis for subclass extensions
#         return rom, trial_basis
        
#     def test_orthonormalize_basis(self, rom_class, used_fom):
#         """Test basis orthonormalization."""
#         rom = rom_class(used_fom)
        
#         # Add non-orthogonal basis
#         basis = np.array([[1.0, 1.0], [0.0, 1.0], [0.2, 0.3]])
#         rom.add_basis(basis)
#         rom.orthonormalize()
        
#         # Base assertions - check U orthonormality
#         gram = rom.fom.U.inner(None, rom.U_basis)
#         assert np.allclose(gram, np.eye(rom.dim), atol=1e-10)
        
#         # Return rom and original basis for subclass extensions
#         return rom, basis
        
#     def test_solve_and_reconstruct_identity(self, rom_class, used_fom, identity_basis):
#         """Test solve and reconstruction with identity basis."""
#         rom = rom_class(used_fom)
#         rom.add_basis(identity_basis)
#         rom.orthonormalize()
        
#         mu = 0.5
#         u_rom = rom.solve(mu)
#         u_reconstructed = rom.reconstruct(u_rom)
#         u_fom = used_fom.solve(mu)
        
#         # Base assertions
#         assert np.allclose(u_reconstructed, u_fom, atol=1e-6)
        
#         # Return values for subclass extensions
#         return rom, mu, u_rom, u_reconstructed, u_fom
        
#     def test_dimension_mismatch_error(self, rom_class, used_fom):
#         """Test error handling for dimension mismatches."""
#         rom = rom_class(used_fom)
        
#         # Wrong dimension basis (2D instead of 3D)
#         wrong_basis = np.array([[1.0], [2.0]])
        
#         with pytest.raises(ValueError, match="incompatible dimension"):
#             rom.add_basis(wrong_basis)
            
#         return rom
    
   
# # =============================================================================
# # Test ROM Base Class

# class TestROM(_TestROMBase):
#     """Tests for ROM base class."""
    
#     @pytest.fixture
#     def rom_class(self):
#         return ROM
    
#     @pytest.fixture
#     def used_fom(self, pg_fom):
#         return pg_fom
    
    
# # ============================================================================
# # Test ROM functionalities for ROMs that only allow adding U_basis

# class _TestROMOnlyUBasis(_TestROMBase):
#     """Base class for ROMs that only allow adding U_basis (Trial2TestROM, GalerkinROM)."""
    
#     def test_initialization_with_basis(self, rom_class, used_fom, trial_basis):
#         """Test ROM initialization with initial basis - extended for auto V_basis."""
#         # Call parent test first
#         rom, trial_basis = super().test_initialization_with_basis(rom_class, used_fom, trial_basis)
        
#         # Extended assertions - check V_basis is set and has correct dimensions
#         assert rom.V_basis is not None
#         V_matrix = rom.V_basis(0.5)  # Evaluate at some parameter
#         assert V_matrix.shape == (3, trial_basis.shape[1])
        
#         # Check that B and f are assembled
#         assert rom.B is not None
#         assert rom.f is not None
        
#         return rom, trial_basis
        
#     def test_add_basis_single_vector(self, rom_class, used_fom, single_basis):
#         """Test adding a single basis vector - extended for auto V_basis."""
#         # Call parent test first
#         rom, single_basis = super().test_add_basis_single_vector(rom_class, used_fom, single_basis)
        
#         # Extended assertions - check V_basis dimensions and ROM assembly
#         assert rom.V_basis is not None
#         V_matrix = rom.V_basis(0.5)
#         assert V_matrix.shape == (3, 1)
        
#         # Check reduced system dimensions
#         B_N = rom.B(0.5)
#         f_N = rom.f(0.5)
#         assert B_N.shape == (1, 1)
#         assert f_N.shape == (1,)
        
#         return rom, single_basis
        
#     def test_add_basis_multiple_vectors(self, rom_class, used_fom, trial_basis):
#         """Test adding multiple basis vectors - extended for auto V_basis."""
#         # Call parent test first
#         rom, trial_basis = super().test_add_basis_multiple_vectors(rom_class, used_fom, trial_basis)
        
#         # Extended assertions - check V_basis dimensions and ROM assembly
#         assert rom.V_basis is not None
#         V_matrix = rom.V_basis(0.5)
#         assert V_matrix.shape == (3, trial_basis.shape[1])
        
#         # Check reduced system dimensions
#         B_N = rom.B(0.5)
#         f_N = rom.f(0.5)
#         assert B_N.shape == (trial_basis.shape[1], trial_basis.shape[1])
#         assert f_N.shape == (trial_basis.shape[1],)
        
#         return rom, trial_basis
        
#     def test_cannot_add_test_basis_directly(self, rom_class, used_fom):
#         """Test that adding test basis directly is not allowed."""
#         rom = rom_class(used_fom)
        
#         U_basis = single_basis = np.array([[1.0], [0.0], [0.0]])
#         # Create a simple V_basis
#         from ulmRBM.affine import AffineLinear
#         V_basis = AffineLinear([lambda mu: 1.0], [np.array([[1.0], [0.0], [0.0]])])
        
#         # Should raise error when trying to add both U and V basis
#         with pytest.raises((ValueError, RuntimeError, TypeError)):
#             rom.add_basis(U_basis, V_basis)
    

# #============================================================================
# # Test Trial2TestROM

# class TestTrial2TestROM(_TestROMOnlyUBasis):
#     """Tests for Trial2TestROM class."""
    
#     @pytest.fixture
#     def rom_class(self):
#         return Trial2TestROM
    
#     @pytest.fixture
#     def used_fom(self, pg_fom):
#         return pg_fom
    

# #============================================================================
# # Test GalerkinROM

# class TestGalerkinROM(_TestROMOnlyUBasis):
#     """Tests for GalerkinROM class."""
    
#     @pytest.fixture
#     def rom_class(self):
#         return GalerkinROM
    
#     @pytest.fixture
#     def used_fom(self, g_fom):
#         return g_fom
        
        
        
        
        
if __name__ == "__main__":
    pytest.main([__file__])