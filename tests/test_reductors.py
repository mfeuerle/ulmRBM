"""Tests for :mod:`ulmRBM.reductors`.

The algorithms are tested by their defining properties: the returned flags and indices, the size of the
constructed bases and the (estimated) error after termination on the training set.
"""

import numpy as np
import pytest

from ulmRBM.fom import GalerkinFOM, FOM, PrimalDualGalerkinFOM
from ulmRBM.rom import GalerkinROM, ROM, PrimalDualGalerkinROM, StationaryTimeSteppingGalerkinROM, ExactStability, ExactContinuity
from ulmRBM.reductors import greedy_rbm, pod_rbm, pod_greedy_rbm, primaldual_greedy_rbm
from ulmRBM.affine import wrap_affinelinear
from ulmRBM.solver import DirectSolver
from ulmRBM.products import EuclideanInnerProduct
from tests.problems import StationaryProblem, HeatProblem, parameter_grid
from tests.fom.test_timestepping import build_fom

TRAIN = parameter_grid(6)


@pytest.fixture(params=[True, False], ids=['galerkin', 'petrov-galerkin'])
def problem(request):
    return StationaryProblem(n=12, galerkin=request.param)


def make_rom(problem, fom_class=None, rom_class=None):
    kwargs = dict(l=problem.l, stability='direct', continuity='direct', solver=DirectSolver())
    if problem.galerkin:
        fom = (fom_class or GalerkinFOM)(problem.B, problem.f, problem.U, **kwargs)
        return (rom_class or GalerkinROM)(fom, ExactStability(fom), ExactContinuity(fom))
    fom = (fom_class or FOM)(problem.B, problem.f, problem.U, problem.V, **kwargs)
    return (rom_class or ROM)(fom, ExactStability(fom), ExactContinuity(fom))


class TestGreedy:

    @pytest.mark.parametrize("strong", [False, True], ids=['weak', 'strong'])
    @pytest.mark.parametrize("use_output", [False, True], ids=['state', 'output'])
    def test_reaches_tolerance(self, problem, strong, use_output):
        tol = 1e-3
        rom = make_rom(problem)
        flag, err, idx = greedy_rbm(rom, TRAIN, N=problem.n, tol=tol, strong=strong, use_output=use_output, display=False)
        assert flag == 0
        assert err[-1] < tol
        assert len(idx) == rom.shape[1] == len(set(idx))
        if use_output:
            measure = rom.output_error if strong else rom.output_error_bound
        else:
            measure = rom.error if strong else rom.error_bound
        assert max(measure(mu) for mu in TRAIN) < tol

    def test_errors_decrease(self, problem):
        rom = make_rom(problem)
        _, err, _ = greedy_rbm(rom, TRAIN, N=6, tol=0.0, strong=True, display=False)
        assert np.all(np.diff(err[1:-1]) <= 1e-12)  # strong greedy errors are monotone

    def test_selects_maximum_error(self, problem):
        rom = make_rom(problem)
        greedy_rbm(rom, TRAIN, N=2, tol=0.0, display=False)
        expected = np.argmax([rom.error_bound(mu) for mu in TRAIN])
        _, _, idx = greedy_rbm(rom, TRAIN, N=1, tol=0.0, display=False)
        assert idx[0] == expected

    def test_maximum_basis_size(self, problem):
        rom = make_rom(problem)
        flag, err, idx = greedy_rbm(rom, TRAIN, N=3, tol=0.0, display=False)
        assert flag == 1 and rom.shape[1] == 3 and len(idx) == 3
        assert np.isnan(err[-1])

    def test_training_set_exhausted(self, problem):
        rom = make_rom(problem)
        flag, _, idx = greedy_rbm(rom, TRAIN[:3], N=10, tol=-1.0, display=False)
        assert flag == 2 and sorted(idx) == [0, 1, 2]

    def test_basis_is_orthonormal(self, problem):
        rom = make_rom(problem)
        greedy_rbm(rom, TRAIN, N=4, tol=0.0, display=False)
        np.testing.assert_allclose(rom.U_basis.T @ problem.MU @ rom.U_basis, np.eye(4), atol=1e-10)
        rom = make_rom(problem)
        greedy_rbm(rom, TRAIN, N=4, tol=0.0, ortho=EuclideanInnerProduct(problem.n), display=False)
        np.testing.assert_allclose(rom.U_basis.T @ rom.U_basis, np.eye(4), atol=1e-10)

    def test_provided_snapshots_and_callback(self, problem):
        rom = make_rom(problem)
        snapshots = [problem.solution(mu) for mu in TRAIN]
        calls = []
        greedy_rbm(rom, TRAIN, N=2, tol=0.0, strong=snapshots, display=False, callback=lambda *args: calls.append(args))
        assert len(calls) == 2
        _, mu, u = calls[0]
        np.testing.assert_allclose(u, problem.solution(mu))
        with pytest.raises(ValueError):
            greedy_rbm(make_rom(problem), TRAIN, N=2, strong=snapshots[:-1], display=False)

    def test_infinite_error_bound_is_selected(self, problem):
        class VanishingStability(ExactStability):
            def lower_bound(self, mu):
                return 0.0 if np.allclose(mu, TRAIN[5]) else super().lower_bound(mu)

        rom = make_rom(problem)
        rom._fom_stability_estimator = VanishingStability(rom.fom)
        _, _, idx = greedy_rbm(rom, TRAIN, N=2, tol=0.0, display=False)
        assert idx[1] == 5

    def test_all_infinite_error_bounds_can_be_selected(self, problem):
        class VanishingStability(ExactStability):
            def lower_bound(self, mu):
                return 0.0 if np.allclose(mu, TRAIN[4]) or np.allclose(mu, TRAIN[5]) else super().lower_bound(mu)

        selected = set()
        np.random.seed(0)
        for _ in range(20):
            rom = make_rom(problem)
            rom._fom_stability_estimator = VanishingStability(rom.fom)
            selected.add(greedy_rbm(rom, TRAIN, N=2, tol=0.0, display=False)[2][1])
        assert selected == {4, 5}

    def test_output_requires_functional(self, problem):
        rom = make_rom(problem)
        rom.fom.l = None
        with pytest.raises(ValueError):
            greedy_rbm(rom, TRAIN, N=2, use_output=True, display=False)


class TestPOD:

    def test_error_matches_projection_error(self, problem):
        """The returned POD error is the root mean square projection error onto the POD space."""
        rom = make_rom(problem)
        S = np.array([problem.solution(mu) for mu in TRAIN])
        _, err, err_full = pod_rbm(rom, S, N=4, tol=1e-14, snapshots=True, display=False)
        assert rom.shape[1] == 4
        P = rom.U_basis  # orthonormal w.r.t. U
        M = problem.MU
        E = S.T - P @ (P.T @ M @ S.T)
        rms = np.sqrt(np.mean(np.einsum('ik,ij,jk->k', E, M, E)))
        np.testing.assert_allclose(err_full[3], rms, rtol=1e-6)  # err_full[i-1] uses i basis vectors
        np.testing.assert_array_equal(err, err_full[:4])

    @pytest.mark.parametrize("tol", [1e-1, 1e-3])
    def test_reaches_tolerance(self, problem, tol):
        rom = make_rom(problem)
        flag, err, err_full = pod_rbm(rom, TRAIN, N=problem.n, tol=tol, display=False)
        assert flag == 0
        assert err[-1] < tol
        assert len(err) == rom.shape[1]

    def test_maximum_basis_size(self, problem):
        rom = make_rom(problem)
        flag, _, _ = pod_rbm(rom, TRAIN, N=2, tol=1e-14, display=False)
        assert flag == 1 and rom.shape[1] == 2

    def test_zero_tolerance(self, problem):
        rom = make_rom(problem)
        flag, _, _ = pod_rbm(rom, TRAIN, N=2, tol=0.0, display=False)
        assert flag == 1 and rom.shape[1] == 2

    def test_rejects_parametric_inner_product(self, problem):
        from ulmRBM.products import MatrixInnerProduct
        from ulmRBM.affine import AffineLinear
        U = MatrixInnerProduct(AffineLinear([lambda mu: mu[0]], [problem.MU]))
        with pytest.raises(ValueError):
            pod_rbm(make_rom(problem), TRAIN, N=2, U=U, display=False)


@pytest.mark.parametrize("tol", [1e-1, 1e-3])
def test_pod_minimal_number_of_modes(tol):
    """POD adds the smallest number of modes reaching the tolerance."""
    problem = StationaryProblem(n=12)
    rom = make_rom(problem)
    _, _, err_full = pod_rbm(rom, TRAIN, N=problem.n, tol=tol, display=False)
    n = rom.shape[1]
    assert err_full[n - 1] < tol
    assert n == 1 or err_full[n - 2] >= tol


class TestPODGreedy:

    @pytest.fixture
    def heat(self):
        return HeatProblem(n=10, K=10)

    @pytest.fixture
    def rom(self, heat):
        fom = build_fom(heat, 'implicit_euler', l=wrap_affinelinear(heat.l_vec))
        return StationaryTimeSteppingGalerkinROM(fom, ExactStability(fom.LI), ExactContinuity(fom.LE), residual='full')

    @pytest.mark.parametrize("strong", [False, True], ids=['weak', 'strong'])
    @pytest.mark.parametrize("eig", ['direct', 'iterative'])
    def test_reaches_tolerance(self, rom, strong, eig):
        tol, mus = 1e-4, np.linspace(0.5, 2, 6)
        flag, err, idx = pod_greedy_rbm(rom, mus, N=10, tol=tol, Npod=1, strong=strong, eig=eig, display=False)
        assert flag == 0 and err[-1] < tol
        measure = rom.error if strong else rom.error_bound
        assert max(np.max(measure(mu)) for mu in mus) < tol

    @pytest.mark.parametrize("Npod", [1, 2, 3])
    def test_basis_size(self, rom, Npod):
        flag, _, idx = pod_greedy_rbm(rom, np.linspace(0.5, 2, 6), N=5, tol=0.0, Npod=Npod, display=False)
        assert flag == 1 and rom.n == 5

    def test_invalid_eig(self, rom):
        with pytest.raises(ValueError):
            pod_greedy_rbm(rom, [1.0], N=1, eig='foo', display=False)


class TestPrimalDualGreedy:

    @pytest.fixture
    def rom(self):
        return make_rom(StationaryProblem(n=12), PrimalDualGalerkinFOM, PrimalDualGalerkinROM)

    @pytest.mark.parametrize("strong", [False, True], ids=['weak', 'strong'])
    @pytest.mark.parametrize("mode", [0, 1])
    def test_reaches_output_tolerance(self, rom, strong, mode):
        tol = 1e-6
        flag, (err_out, err, err_dual), idx = primaldual_greedy_rbm(rom, TRAIN, N=12, tol=tol, strong=strong, mu_select_mode=mode, display=False)
        assert flag == 0 and err_out[-1] < tol
        measure = rom.output_error if strong else rom.output_error_bound
        assert max(measure(mu) for mu in TRAIN) < tol
        assert rom.shape[1] == rom.dual.shape[1]

    def test_output_bound_better_than_primal_bound(self, rom):
        """The primal-dual output bound decays like the product of primal and dual residual."""
        primaldual_greedy_rbm(rom, TRAIN, N=3, tol=0.0, display=False)
        mu = TRAIN[7]
        primal_only = rom._residual_evaluator.dual_norm(mu, rom.solve(mu)) * np.linalg.norm(rom.fom.l(mu)) / rom._fom_stability_estimator(mu)
        assert rom.output_error_bound(mu) < primal_only
