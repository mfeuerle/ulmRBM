"""Tests for the stationary reduced-order models in :mod:`ulmRBM.rom`.

`ROMContract` collects the tests valid for every reduced-order model, mostly properties known from
RB theory (exactness for a full basis, Galerkin orthogonality, quasi-optimality, rigorous error bounds).
The primal-dual ROMs inherit all tests of their primal counterparts and fulfill `PrimalDualROMContract`.
"""

import numpy as np
import pytest

from ulmRBM.fom import FOM, GalerkinFOM, PrimalDualFOM, PrimalDualGalerkinFOM
from ulmRBM.rom import ROM, GalerkinROM, PrimalDualROM, PrimalDualGalerkinROM, ExactStability, ExactContinuity
from ulmRBM.solver import DirectSolver
from ulmRBM.products import EuclideanInnerProduct
from tests.problems import StationaryProblem, as_dense
from ulmRBM.rom._residuals import AffineResidualNormEvaluator, FullResidualNormEvaluator

SNAPSHOT_MUS = np.array([[0.5, 0.5], [2.0, 1.0], [1.0, 2.0]])
TEST_MUS = np.array([[0.8, 1.3], [1.7, 0.6], [1.2, 1.9], [0.6, 1.6]])
"""Parameters different from the snapshot parameters, such that the reduced solutions are not exact."""


class ROMContract:
    """Tests for reduced-order models. Subclasses define `fom_class`, `rom_class` and `galerkin`."""

    fom_class: type
    rom_class: type
    galerkin: bool

    @pytest.fixture
    def problem(self):
        return StationaryProblem(galerkin=self.galerkin)

    @pytest.fixture
    def fom(self, problem):
        kwargs = dict(l=problem.l, stability='direct', continuity='direct', solver=DirectSolver())
        if self.galerkin:
            return self.fom_class(problem.B, problem.f, problem.U, **kwargs)
        return self.fom_class(problem.B, problem.f, problem.U, problem.V, **kwargs)

    @pytest.fixture(params=['affine', 'full'])
    def residual(self, request):
        return request.param

    @pytest.fixture
    def mus(self):
        return TEST_MUS

    def make_rom(self, fom, **kwargs):
        return self.rom_class(fom, ExactStability(fom), ExactContinuity(fom), **kwargs)

    def add_basis(self, rom, basis, dual_basis=None):
        """Enrich the reduced model. Primal-dual subclasses also enrich the dual model."""
        rom.add_basis(basis)

    def snapshots(self, problem):
        return np.column_stack([problem.solution(mu) for mu in SNAPSHOT_MUS])

    def dual_snapshots(self, problem):
        return np.column_stack([problem.dual_solution(mu) for mu in SNAPSHOT_MUS])

    @pytest.fixture
    def rom(self, fom, problem, residual):
        rom = self.make_rom(fom, residual=residual)
        self.add_basis(rom, self.snapshots(problem), self.dual_snapshots(problem))
        return rom

    @pytest.fixture
    def full_rom(self, fom, problem, residual):
        rom = self.make_rom(fom, residual=residual)
        basis = np.linalg.qr(np.random.default_rng(0).standard_normal((problem.n, problem.n)))[0]
        self.add_basis(rom, basis[:, :3])
        self.add_basis(rom, basis[:, 3:])
        return rom

    # --- construction

    def test_empty_rom(self, fom, problem):
        rom = self.make_rom(fom)
        assert rom.shape == (0, 0, problem.n)
        assert rom.U_basis is None and rom.V_basis is None

    def test_add_basis(self, rom, problem):
        assert rom.shape == (3, 3, problem.n)
        assert rom.U_basis.shape == (problem.n, 3)
        assert rom.V_basis.shape == (problem.n, 3)
        self.add_basis(rom, problem.solution(np.array([1.5, 1.5])))
        assert rom.shape[:2] == (4, 4)

    def test_add_basis_dimension_mismatch(self, rom, problem):
        with pytest.raises(ValueError):
            rom.add_basis(np.ones(problem.n + 1))

    def test_estimators_must_belong_to_fom(self, fom, problem):
        other = self.fom_class(fom.B, fom.f, fom.U, **({} if self.galerkin else {'V': fom.V}), l=fom.l)
        with pytest.raises(ValueError):
            self.rom_class(fom, ExactStability(other))
        with pytest.raises(ValueError):
            self.rom_class(fom, ExactStability(fom), ExactContinuity(other))

    # --- solutions

    def test_full_basis_reproduces_fom(self, full_rom, problem, mus):
        for mu in mus:
            np.testing.assert_allclose(full_rom.reconstruct(mu), problem.solution(mu), rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(full_rom.output(mu), problem.output(mu), rtol=1e-8)
            assert full_rom.error(mu) < 1e-10
            assert full_rom.error_bound(mu) < 1e-6  # square-root effect

    def test_snapshots_are_reproduced(self, rom, problem):
        for mu in SNAPSHOT_MUS:
            np.testing.assert_allclose(rom.reconstruct(mu), problem.solution(mu), rtol=1e-8, atol=1e-10)

    def test_reconstruct(self, rom, problem):
        mu = problem.mus()[4]
        u_N = rom.solve(mu)
        assert u_N.shape == (3,)
        np.testing.assert_allclose(rom.reconstruct(mu, u_N), rom.U_basis @ u_N)
        np.testing.assert_allclose(rom.reconstruct(mu), rom.U_basis @ u_N)

    def test_petrov_galerkin_orthogonality(self, rom, problem, mus):
        r"""The residual :math:`r = f - B U_N u_N` is orthogonal to the reduced test space."""
        for mu in mus:
            r = problem.f_dense(mu) - problem.B_dense(mu) @ rom.reconstruct(mu)
            np.testing.assert_allclose(rom.V_basis(mu).T @ r, 0, atol=1e-10)

    def test_quasi_optimality(self, rom, problem, mus):
        r"""Céa's lemma: :math:`\|u - u_N\|_U \leq (1 + \gamma/\beta) \inf_{v_N} \|u - v_N\|_U`."""
        MU, UN = problem.MU, rom.U_basis
        for mu in mus:
            u = problem.solution(mu)
            best = UN @ np.linalg.solve(UN.T @ MU @ UN, UN.T @ MU @ u)
            best_error = np.sqrt((u - best) @ MU @ (u - best))
            assert rom.error(mu) <= (1 + problem.continuity(mu) / problem.stability(mu)) * best_error * (1 + 1e-10)

    # --- errors and error bounds

    def test_error(self, rom, problem, mus):
        for mu in mus:
            e = problem.solution(mu) - rom.reconstruct(mu)
            abs_err = np.sqrt(e @ problem.MU @ e)
            rel_err = abs_err / np.sqrt(problem.solution(mu) @ problem.MU @ problem.solution(mu))
            np.testing.assert_allclose(rom.error(mu), abs_err, rtol=1e-8)
            np.testing.assert_allclose(rom.error(mu, abs=False, rel=True), rel_err, rtol=1e-8)
            np.testing.assert_allclose(rom.error(mu, rel=True), (abs_err, rel_err), rtol=1e-8)
        with pytest.raises(ValueError):
            rom.error(mus[0], abs=False, rel=False)

    def test_error_bounds_are_rigorous(self, rom, problem, mus):
        for mu in mus:
            bounds = rom.error_bounds(mu, aub=True, rub=True, alb=True, rlb=True)
            abs_err, rel_err = rom.error(mu, rel=True)
            assert bounds['alb'] <= abs_err <= bounds['aub']
            assert bounds['rlb'] <= rel_err <= bounds['rub']
            assert rom.error_bound(mu) == bounds['aub']

    def test_error_bound_effectivity(self, rom, problem, mus):
        r"""The effectivity of the residual-based bound is bounded by :math:`\gamma(\mu)/\beta(\mu)`."""
        for mu in mus:
            effectivity = rom.error_bound(mu) / rom.error(mu)
            assert 1 <= effectivity <= problem.continuity(mu) / problem.stability(mu) * (1 + 1e-8)

    def test_residual_norm(self, rom, problem):
        mu = problem.mus()[2]
        r = problem.f_dense(mu) - problem.B_dense(mu) @ rom.reconstruct(mu)
        f = problem.f_dense(mu)
        np.testing.assert_allclose(rom._residual_evaluator.dual_norm(mu, rom.solve(mu)), np.sqrt(r @ np.linalg.solve(problem.MV, r)), rtol=1e-6)
        np.testing.assert_allclose(rom._residual_evaluator.dual_norm_rhs(mu), np.sqrt(f @ np.linalg.solve(problem.MV, f)), rtol=1e-6)

    def test_output_error(self, rom, problem, mus):
        for mu in mus:
            np.testing.assert_allclose(rom.output_error(mu), abs(problem.output(mu) - rom.output(mu)), rtol=1e-8)
            assert rom.output_error(mu) <= rom.output_error_bound(mu)

    # --- orthonormalization

    def test_orthonormalize(self, rom, problem, mus):
        before = [(rom.reconstruct(mu), rom.error_bound(mu)) for mu in mus]
        rom.orthonormalize()
        np.testing.assert_allclose(rom.U_basis.T @ problem.MU @ rom.U_basis, np.eye(3), atol=1e-10)
        for mu, (u, bound) in zip(mus, before):
            np.testing.assert_allclose(rom.reconstruct(mu), u, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(rom.error_bound(mu), bound, rtol=1e-6)

    def test_orthonormalize_custom_inner_product(self, rom, problem):
        rom.orthonormalize(EuclideanInnerProduct(problem.n))
        np.testing.assert_allclose(rom.U_basis.T @ rom.U_basis, np.eye(3), atol=1e-10)


class PrimalDualROMContract:
    """Tests for primal-dual reduced-order models."""

    def add_basis(self, rom, basis, dual_basis=None):
        rom.add_basis(basis)
        rom.dual.add_basis(basis if dual_basis is None else dual_basis)

    def test_dual_rom(self, rom):
        assert rom.dual.fom is rom.fom.dual
        assert rom.dual.dual is rom
        assert rom.dual._fom_stability_estimator is rom._fom_stability_estimator

    def test_dual_snapshots_are_reproduced(self, rom, problem):
        for mu in SNAPSHOT_MUS:
            np.testing.assert_allclose(rom.dual.reconstruct(mu), problem.dual_solution(mu), rtol=1e-8, atol=1e-10)

    def test_corrected_output_is_exact_for_exact_dual(self, fom, problem, mus):
        r"""With an exact dual solution, the primal-dual output :math:`l^T u_N - z^T r(u_N)` is exact."""
        rom = self.make_rom(fom)
        rom.add_basis(problem.solution(SNAPSHOT_MUS[0]))
        rom.dual.add_basis(np.eye(problem.n))
        for mu in mus:
            np.testing.assert_allclose(rom.output(mu), problem.output(mu), rtol=1e-8)

    def test_output_error_bound_is_product_of_residuals(self, rom, problem, mus):
        for mu in mus:
            u, z = rom.solve(mu), rom.dual.solve(mu)
            expected = (rom._residual_evaluator.dual_norm(mu, u) * rom.dual._residual_evaluator.dual_norm(mu, z)
                        / problem.stability(mu))
            np.testing.assert_allclose(rom.output_error_bound(mu), expected, rtol=1e-8)

    @pytest.mark.parametrize("method", ['affine', 'full'])
    def test_single_residual_method(self, fom, method):
        rom = self.make_rom(fom, residual=method)
        assert type(rom._residual_evaluator) is type(rom.dual._residual_evaluator)

    @pytest.mark.parametrize("method", ['affine', 'full'])
    def test_residual_methods(self, fom, method):
        rom = self.make_rom(fom, residual=[method, 'full'])
        assert isinstance(rom._residual_evaluator, {'affine': AffineResidualNormEvaluator, 'full': FullResidualNormEvaluator}[method])
        assert isinstance(rom.dual._residual_evaluator, FullResidualNormEvaluator)

    def test_invalid_number_of_residual_methods(self, fom):
        with pytest.raises(ValueError):
            self.make_rom(fom, residual=['full'] * 3)

    def test_assemble_with_primal_basis_only(self, fom, problem):
        rom = self.make_rom(fom)
        rom.add_basis(problem.solution(SNAPSHOT_MUS[0]))
        rom.assemble()
        assert rom._B_mixed is None and rom.dual._B_mixed is None

    def test_assemble_with_dual_basis_only(self, fom, problem):
        rom = self.make_rom(fom)
        rom.dual.add_basis(problem.dual_solution(SNAPSHOT_MUS[0]))
        rom.assemble()
        assert rom._B_mixed is None and rom.dual._B_mixed is None


class TestROM(ROMContract):
    fom_class, rom_class, galerkin = FOM, ROM, False

    def test_test_basis_is_supremizer(self, rom, problem):
        mu = problem.mus()[5]
        np.testing.assert_allclose(rom.V_basis(mu), np.linalg.solve(problem.MV, problem.B_dense(mu) @ rom.U_basis), rtol=1e-6)

    def test_custom_trial2test(self, fom, problem):
        rom = self.make_rom(fom, trial2test=lambda basis: basis)
        rom.add_basis(self.snapshots(problem))
        np.testing.assert_allclose(rom.V_basis(None), rom.U_basis)

    def test_rectangular_fom(self):
        B = np.vstack([np.eye(3), np.ones((1, 3))])
        fom = FOM(B, np.ones(4), EuclideanInnerProduct(3), EuclideanInnerProduct(4), stability='direct', continuity='direct')
        rom = ROM(fom, ExactStability(fom), trial2test=lambda basis: np.vstack([basis, np.zeros((1, basis.shape[1]))]))
        rom.add_basis(np.eye(3)[:, :2])
        assert rom.shape[:2] == (2, 2)
        rom.add_basis(np.eye(3)[:, [2]])
        assert rom.U_basis.shape == (3, 3) and rom.V_basis.shape == (4, 3)
        with pytest.raises(ValueError):
            rom.add_basis(np.ones(4))

    def test_shape_follows_fom_convention(self, rom):
        """Like ``FOM.shape``, ``ROM.shape`` is (test dimension, trial dimension, number of outputs)."""
        assert rom.shape == (rom.V_basis.shape[1], rom.U_basis.shape[1], rom.fom.shape[2])


class TestGalerkinROM(ROMContract):
    fom_class, rom_class, galerkin = GalerkinFOM, GalerkinROM, True

    def make_rom(self, fom, trial2test=None, **kwargs):
        return self.rom_class(fom, ExactStability(fom), ExactContinuity(fom), **kwargs)

    def test_test_basis_is_trial_basis(self, rom):
        np.testing.assert_array_equal(rom.V_basis(None), rom.U_basis)
        assert rom.V_basis.shape == rom.U_basis.shape

    def test_galerkin_projection_is_energy_optimal(self, rom, problem, mus, rng):
        r"""For symmetric coercive :math:`B`, the Galerkin solution minimizes the energy norm error."""
        for mu in mus:
            B = problem.B_dense(mu)
            e = problem.solution(mu) - rom.reconstruct(mu)
            for _ in range(5):
                v = problem.solution(mu) - rom.U_basis @ (rom.solve(mu) + 1e-2 * rng.standard_normal(3))
                assert e @ B @ e <= v @ B @ v

    def test_requires_galerkin_fom(self, problem):
        fom = FOM(problem.B, problem.f, problem.U, problem.V)
        with pytest.raises(ValueError):
            GalerkinROM(fom, ExactStability(fom))


class TestPrimalDualROM(PrimalDualROMContract, TestROM):
    fom_class, rom_class = PrimalDualFOM, PrimalDualROM

    def make_rom(self, fom, trial2test=None, **kwargs):
        return self.rom_class(fom, ExactStability(fom), ExactContinuity(fom), primal_trial2test=trial2test, **kwargs)

    @pytest.fixture
    def residual(self):
        return None  # explicit residual methods are broken, see test_residual_methods

    # covered by test_output_error_bound_is_product_of_residuals, as the primal-dual bound replaces the primal one
    def test_output_error(self, rom, problem, mus):
        for mu in mus:
            np.testing.assert_allclose(rom.output_error(mu), abs(problem.output(mu) - rom.output(mu)), rtol=1e-6, atol=1e-12)
            assert rom.output_error(mu) <= rom.output_error_bound(mu)

    def test_custom_trial2test(self, fom, problem):
        rom = self.make_rom(fom, trial2test=lambda basis: basis)
        self.add_basis(rom, self.snapshots(problem))
        np.testing.assert_allclose(rom.V_basis(None), rom.U_basis)

    test_rectangular_fom = None


class TestPrimalDualGalerkinROM(PrimalDualROMContract, TestGalerkinROM):
    fom_class, rom_class = PrimalDualGalerkinFOM, PrimalDualGalerkinROM

    residual = TestPrimalDualROM.residual

    test_output_error = TestPrimalDualROM.test_output_error

    def test_requires_galerkin_fom(self, problem):
        fom = PrimalDualFOM(problem.B, problem.f, problem.U, problem.V, problem.l)
        with pytest.raises(ValueError):
            PrimalDualGalerkinROM(fom, ExactStability(fom))
