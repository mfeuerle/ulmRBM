"""Tests for :mod:`ulmRBM.fom` operators.

`OperatorContract` collects the tests of the functionality of `ParametricOperator`. It is also inherited by
the model tests in ``test_models.py``, such that the operator functionality is checked for every subclass.
Subclasses provide a ``case`` fixture returning ``(operator, problem)``, where ``problem`` is a
`tests.problems.StationaryProblem` with analytically known constants.
"""

import numpy as np
import pytest

from ulmRBM.fom import ParametricOperator, ParametricGalerkinOperator
from ulmRBM.products import MatrixInnerProduct
from tests.problems import StationaryProblem, as_dense

MU = np.array([0.7, 1.6])
RTOL = 1e-7


def rayleigh_quotient(op, problem: StationaryProblem, mu, x):
    """Value of the quotient whose extrema define stability and continuity constant."""
    B = problem.B_dense(mu)
    if isinstance(op, ParametricGalerkinOperator):
        return (x @ B @ x) / (x @ problem.MU @ x)
    return np.sqrt((B @ x) @ np.linalg.solve(problem.MV, B @ x) / (x @ problem.MU @ x))


class OperatorContract:
    """Tests for the functionality of `ParametricOperator`, valid for all subclasses."""

    def make(self, problem, **kwargs):
        """Construct the object under test. Subclasses may override."""
        raise NotImplementedError

    @pytest.fixture
    def case(self, problem, fmt):
        problem.fmt = fmt
        return self.make(problem), problem

    def test_shape(self, case):
        op, problem = case
        assert op.shape[:2] == (problem.n, problem.n)

    def test_operator_matrix(self, case):
        op, problem = case
        np.testing.assert_allclose(as_dense(op.B(MU)), problem.B_dense(MU), rtol=1e-12)

    @pytest.mark.parametrize("method", ['direct', 'iterative'])
    def test_stability_exact(self, case, method):
        op, problem = case
        op = self.make(problem, stability=method)
        np.testing.assert_allclose(op.stability(MU), problem.stability(MU), rtol=RTOL)

    @pytest.mark.parametrize("method", ['direct', 'iterative'])
    def test_continuity_exact(self, case, method):
        op, problem = case
        op = self.make(problem, continuity=method)
        np.testing.assert_allclose(op.continuity(MU), problem.continuity(MU), rtol=RTOL)

    @pytest.mark.parametrize("method", ['direct', 'iterative'])
    def test_eigenvectors(self, case, method):
        op, problem = case
        op = self.make(problem, stability=method, continuity=method)
        val, x = op.stability(MU, eigenvector=True)
        np.testing.assert_allclose(rayleigh_quotient(op, problem, MU, x), val, rtol=1e-6)
        val, x = op.continuity(MU, eigenvector=True)
        np.testing.assert_allclose(rayleigh_quotient(op, problem, MU, x), val, rtol=1e-6)

    def test_estimates_are_bounds(self, case):
        op, problem = case
        op = self.make(problem, stability='estimate', continuity='estimate')
        assert 0 < op.stability(MU) <= problem.stability(MU) * (1 + 1e-10)
        assert op.continuity(MU) >= problem.continuity(MU) * (1 - 1e-10)

    def test_estimate_with_eigenvector_falls_back(self, case):
        op, problem = case
        op = self.make(problem, stability='estimate')
        with pytest.warns(UserWarning):
            val, _ = op.stability(MU, eigenvector=True)
        np.testing.assert_allclose(val, problem.stability(MU), rtol=RTOL)

    def test_custom_constants(self, case):
        op, problem = case
        op = self.make(problem, stability=lambda mu, op: 42.0 * mu[0], continuity=3.0)
        assert op.stability(MU) == 42.0 * MU[0]
        assert op.continuity(MU) == 3.0

    def test_invalid_method(self, case):
        _, problem = case
        with pytest.raises(ValueError):
            self.make(problem, stability='foo')
        with pytest.raises(ValueError):
            self.make(problem, continuity='foo')

    def test_supremizer(self, case, rng):
        op, problem = case
        u = rng.standard_normal((problem.n, 2))
        S = op.supremizer(u)
        expected = np.linalg.solve(problem.MV, problem.B_dense(MU) @ u)
        np.testing.assert_allclose(S(MU), expected, rtol=1e-6)

    def test_custom_supremizer(self, case):
        _, problem = case
        op = self.make(problem, supremizer=lambda u, op: 2 * u)
        np.testing.assert_array_equal(op.supremizer(np.ones(problem.n)), 2 * np.ones(problem.n))


class TestParametricOperator(OperatorContract):

    @pytest.fixture(params=[False, True], ids=['petrov-galerkin-problem', 'galerkin-problem'])
    def problem(self, request):
        return StationaryProblem(galerkin=request.param)

    def make(self, problem, **kwargs):
        return ParametricOperator(problem.B, problem.U, problem.V, **kwargs)

    def test_dimension_mismatch(self, problem):
        with pytest.raises(ValueError):
            ParametricOperator(problem.B, MatrixInnerProduct(np.eye(problem.n + 1)), problem.V)
        with pytest.raises(ValueError):
            ParametricOperator(problem.B, problem.U, MatrixInnerProduct(np.eye(problem.n + 1)))


class TestParametricGalerkinOperator(OperatorContract):

    @pytest.fixture
    def problem(self):
        return StationaryProblem(galerkin=True)

    def make(self, problem, **kwargs):
        return ParametricGalerkinOperator(problem.B, problem.U, **kwargs)

    def test_test_space_is_trial_space(self, case):
        op, problem = case
        assert op.V is op.U
        with pytest.raises(AttributeError):
            op.V = MatrixInnerProduct(np.eye(problem.n))

    def test_coercivity_uses_symmetric_part(self):
        """For a nonsymmetric operator, the coercivity constant only depends on the symmetric part."""
        rng = np.random.default_rng(0)
        S = np.diag(rng.uniform(1, 2, 5))
        K = rng.standard_normal((5, 5))
        K = K - K.T  # skew-symmetric part does not change u^T B u
        op = ParametricGalerkinOperator(S + K, MatrixInnerProduct(np.eye(5)), stability='direct')
        np.testing.assert_allclose(op.stability(None), np.min(np.diag(S)), rtol=1e-10)
