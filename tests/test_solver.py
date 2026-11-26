"""Tests for :mod:`ulmRBM.solver`.

Every solver has to fulfill the same contract (`SolverContract`): solve square systems, keep the shape
of single and multiple right-hand sides and respect the shape of a given initial guess. Solvers that
support least-squares problems additionally fulfill `LeastSquaresContract`.
"""

import numpy as np
import scipy as sp
import pytest

from ulmRBM.solver import Solver, DirectSolver, IterativeSolver, wrap_solver
from tests.problems import to_format

TOL = 1e-8


def spd_matrix(n=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n))
    return X @ X.T + n * np.eye(n)


def nonsymmetric_matrix(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + n * np.eye(n)


class SolverContract:
    """Base tests for all solvers. Subclasses provide the ``solver`` fixture and the supported ``formats``."""

    formats = ['dense', 'sparse', 'linop']
    symmetric_only = False

    @pytest.fixture
    def matrix(self, request):
        return spd_matrix() if self.symmetric_only else nonsymmetric_matrix()

    @pytest.fixture(params=formats)
    def A(self, request, matrix):
        if request.param not in self.formats:
            pytest.skip(f"{request.param} not supported by {type(self).__name__}")
        return to_format(matrix, request.param)

    def test_is_solver(self, solver):
        assert isinstance(wrap_solver(solver), Solver)

    def test_single_rhs(self, solver, A, matrix, rng):
        x = rng.standard_normal(matrix.shape[1])
        x_sol = solver(A, matrix @ x)
        assert x_sol.shape == x.shape
        np.testing.assert_allclose(x_sol, x, rtol=TOL)

    def test_multiple_rhs(self, solver, A, matrix, rng):
        X = rng.standard_normal((matrix.shape[1], 3))
        X_sol = solver(A, matrix @ X)
        assert X_sol.shape == X.shape
        np.testing.assert_allclose(X_sol, X, rtol=TOL)

    def test_column_rhs_keeps_2d_shape(self, solver, A, matrix, rng):
        x = rng.standard_normal((matrix.shape[1], 1))
        assert solver(A, matrix @ x).shape == (matrix.shape[1], 1)

    def test_initial_guess_determines_shape(self, solver, A, matrix, rng):
        x = rng.standard_normal(matrix.shape[1])
        x_sol = solver(A, (matrix @ x).reshape(-1, 1), np.zeros(matrix.shape[1]))
        assert x_sol.shape == x.shape
        np.testing.assert_allclose(x_sol, x, rtol=TOL)


class LeastSquaresContract:
    """Solvers supporting overdetermined systems return the least-squares solution."""

    def test_least_squares(self, solver, rng):
        A = rng.standard_normal((10, 4))
        b = rng.standard_normal(10)
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        x = solver(A, b)
        np.testing.assert_allclose(x, expected, rtol=1e-6)


class TestDirectSolver(SolverContract, LeastSquaresContract):

    @pytest.fixture
    def solver(self):
        return DirectSolver()


class TestFactorizedDirectSolver(SolverContract, LeastSquaresContract):

    @pytest.fixture
    def solver(self):
        return DirectSolver(factorize=True)

    def test_reuses_factorization_for_same_matrix(self, solver, rng):
        A = sp.sparse.csc_array(nonsymmetric_matrix())
        solver(A, np.ones(A.shape[0]))
        cached = solver._solveA
        x = rng.standard_normal(A.shape[0])
        np.testing.assert_allclose(solver(A, A @ x), x, rtol=TOL)
        assert solver._solveA is cached

    def test_refactorizes_for_new_matrix(self, solver, rng):
        A1, A2 = nonsymmetric_matrix(seed=1), nonsymmetric_matrix(seed=2)
        x = rng.standard_normal(A1.shape[0])
        solver(A1, A1 @ x)
        np.testing.assert_allclose(solver(A2, A2 @ x), x, rtol=TOL)


class TestGMRES(SolverContract, LeastSquaresContract):
    """``IterativeSolver(spd=False)`` uses GMRES for square and LSMR for rectangular systems."""

    @pytest.fixture
    def solver(self):
        return IterativeSolver(rtol=1e-12, atol=1e-14)


class TestCG(SolverContract):
    symmetric_only = True

    @pytest.fixture
    def solver(self):
        return IterativeSolver(spd=True, rtol=1e-12, atol=1e-14)

    def test_rejects_rectangular_systems(self, solver):
        with pytest.raises(ValueError):
            solver(np.ones((3, 2)), np.ones(3))


class TestWrappedCallable(SolverContract):
    """Callables are wrapped to fulfill the solver contract, even if they only accept 1D right-hand sides."""
    formats = ['dense']

    @pytest.fixture(params=['2d', '1d-only'])
    def solver(self, request):
        if request.param == '2d':
            return wrap_solver(lambda A, b, x0: np.linalg.solve(A, b))

        def solve_1d(A, b, x0):
            if b.ndim != 1:
                raise ValueError("only 1D right-hand sides")
            return np.linalg.solve(A, b)
        return wrap_solver(solve_1d)

    def test_rejects_underdetermined_systems(self, solver):
        with pytest.raises(ValueError):
            solver(np.ones((2, 3)), np.ones(2))
