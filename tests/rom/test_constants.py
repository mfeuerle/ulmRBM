"""Tests for the constant estimators in :mod:`ulmRBM.rom`.

All estimators are checked via `EstimatorContract` on Galerkin and Petrov-Galerkin operators with
analytically known constants (`tests.problems.StationaryProblem`). The affine terms of these problems are
positive semi-definite with positive coefficients, which is the assumption of the Min-Theta estimators.
"""

import numpy as np
import pytest

from ulmRBM.fom import ParametricOperator, ParametricGalerkinOperator
from ulmRBM.rom import (
    StabilityEstimator, ContinuityEstimator, EfficientConstantEstimator,
    ExactStability, ExactContinuity, ThetaStability, ThetaContinuity, SCMStability, SCMContinuity,
    greedy_constant_estimator,
)
from ulmRBM.affine import AffineLinear
from ulmRBM.products import EuclideanInnerProduct
from tests.problems import StationaryProblem, parameter_grid

TRAIN = parameter_grid(5)
TEST = parameter_grid(4) + 0.1
RTOL = 1e-6


@pytest.fixture(params=[True, False], ids=['galerkin', 'petrov-galerkin'])
def problem(request):
    return StationaryProblem(galerkin=request.param)


@pytest.fixture
def operator(problem):
    if problem.galerkin:
        return ParametricGalerkinOperator(problem.B, problem.U, stability='direct', continuity='direct')
    return ParametricOperator(problem.B, problem.U, problem.V, stability='direct', continuity='direct')


class EstimatorContract:
    """Tests valid for all estimators. Subclasses define `make` and the estimated `constant`."""

    constant: str  # 'stability' or 'continuity'

    def make(self, operator):
        raise NotImplementedError

    def exact(self, problem, mu):
        return getattr(problem, self.constant)(mu)

    @pytest.fixture
    def estimator(self, operator):
        estimator = self.make(operator)
        for mu in TRAIN[::6]:
            estimator.update(mu)
        return estimator

    def test_counts_updates(self, operator):
        estimator = self.make(operator)
        assert estimator.n == 0
        estimator.update(TRAIN[0])
        estimator.update(TRAIN[1])
        assert estimator.n == 2
        assert estimator.B is operator

    def test_rigorous_bound(self, estimator, problem):
        for mu in TEST:
            if self.constant == 'stability':
                assert isinstance(estimator, StabilityEstimator)
                assert 0 <= estimator.lower_bound(mu) <= self.exact(problem, mu) * (1 + RTOL)
                assert estimator(mu) == estimator.lower_bound(mu)
            else:
                assert isinstance(estimator, ContinuityEstimator)
                assert estimator.upper_bound(mu) >= self.exact(problem, mu) * (1 - RTOL)
                assert estimator(mu) == estimator.upper_bound(mu)

    def test_efficient_bounds_enclose_constant(self, estimator, problem):
        assert isinstance(estimator, EfficientConstantEstimator)
        for mu in TEST:
            lb, ub, exact = estimator.lower_bound(mu), estimator.upper_bound(mu), self.exact(problem, mu)
            assert lb <= exact * (1 + RTOL) and exact <= ub * (1 + RTOL)
            rel, abs_ = estimator.error_bound(mu, rel=True, abs=True)
            np.testing.assert_allclose(abs_, abs(ub - lb), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(rel, abs(ub - lb) / lb, rtol=1e-12, atol=1e-14)
            assert estimator.error_bound(mu) == rel
            assert estimator.error_bound(mu, rel=False, abs=True) == abs_

    def test_exact_at_sampled_parameters(self, estimator, problem):
        for mu in TRAIN[::6]:
            np.testing.assert_allclose(estimator(mu), self.exact(problem, mu), rtol=RTOL)

    def test_improves_with_updates(self, operator):
        estimator = self.make(operator)
        estimator.update(TRAIN[0])
        before = max(estimator.error_bound(mu) for mu in TEST)
        for mu in TRAIN[1::3]:
            estimator.update(mu)
        assert max(estimator.error_bound(mu) for mu in TEST) <= before * (1 + 1e-10)


class OfflineEstimatorContract(EstimatorContract):
    """Estimators that require offline samples before evaluation."""

    def test_requires_update(self, operator):
        with pytest.raises(RuntimeError):
            self.make(operator).lower_bound(TEST[0])

    def test_rejects_parametric_inner_products(self, problem):
        U = problem.U
        Up = type(U)(AffineLinear([lambda mu: mu[0], 1.0], [problem.MU, problem.MU]))
        op = ParametricGalerkinOperator(problem.B, Up) if problem.galerkin else ParametricOperator(problem.B, Up, problem.V)
        with pytest.raises(ValueError):
            estimator = self.make(op)
            estimator.update(TRAIN[0])


class TestExactStability(EstimatorContract):
    constant = 'stability'

    def make(self, operator):
        return ExactStability(operator)

    def test_zero_error(self, estimator):
        assert estimator.error_bound(TEST[0]) == 0.0
        assert estimator.error_bound(TEST[0], abs=True) == (0.0, 0.0)


class TestExactContinuity(TestExactStability):
    constant = 'continuity'

    def make(self, operator):
        return ExactContinuity(operator)


class TestThetaStability(OfflineEstimatorContract):
    constant = 'stability'

    def make(self, operator):
        return ThetaStability(operator)

    def test_exact_for_single_affine_term(self):
        r"""For :math:`B(\mu) = \theta(\mu) B_1`, the constant scales exactly with :math:`\theta(\mu)`."""
        B = AffineLinear([lambda mu: mu[0] * mu[1]], [np.diag([1.0, 2.0, 3.0])])
        op = ParametricGalerkinOperator(B, EuclideanInnerProduct(3), stability='direct', continuity='direct')
        estimator = self.make(op)
        estimator.update(TRAIN[0])
        for mu in TEST:
            np.testing.assert_allclose(estimator.lower_bound(mu), getattr(op, self.constant)(mu), rtol=1e-10)
            np.testing.assert_allclose(estimator.upper_bound(mu), getattr(op, self.constant)(mu), rtol=1e-10)

    @pytest.mark.xfail(strict=True, reason="BUG: sign check of the coefficients is never active (_positive_theta is never set)")
    def test_rejects_sign_changing_coefficients(self):
        B = AffineLinear([lambda mu: mu], [np.eye(2)])
        estimator = self.make(ParametricGalerkinOperator(B, EuclideanInnerProduct(2), stability='direct', continuity='direct'))
        estimator.update(1.0)
        with pytest.raises(ValueError):
            estimator.lower_bound(-1.0)


class TestThetaContinuity(TestThetaStability):
    constant = 'continuity'

    def make(self, operator):
        return ThetaContinuity(operator)


@pytest.mark.slow
class TestSCMStability(OfflineEstimatorContract):
    constant = 'stability'

    @pytest.fixture(params=['direct', 'iterative'])
    def eigenvalues(self, request):
        return request.param

    @pytest.fixture
    def operator(self, operator, eigenvalues):
        self._eigenvalues = eigenvalues
        return operator

    def make(self, operator):
        return SCMStability(operator, TRAIN, Me=10, Mp=10, eigenvalues=getattr(self, '_eigenvalues', 'direct'))

    def test_estimate_box_constraints(self, operator, problem):
        """With estimated (wider) box constraints, the bounds are still rigorous."""
        estimator_class = SCMStability if self.constant == 'stability' else SCMContinuity
        estimator = estimator_class(operator, TRAIN, Me=10, Mp=10, eigenvalues='estimate')
        for mu in TRAIN[::6]:
            estimator.update(mu)
        for mu in TEST:
            assert estimator.lower_bound(mu) <= self.exact(problem, mu) * (1 + RTOL) <= estimator.upper_bound(mu) * (1 + 2 * RTOL)

    def test_invalid_eigenvalue_method(self, operator):
        with pytest.raises(ValueError):
            SCMStability(operator, TRAIN, eigenvalues='foo')


@pytest.mark.slow
class TestSCMContinuity(TestSCMStability):
    constant = 'continuity'

    def make(self, operator):
        return SCMContinuity(operator, TRAIN, Me=10, Mp=10, eigenvalues=getattr(self, '_eigenvalues', 'direct'))


class TestGreedyConstantEstimator:

    @pytest.mark.parametrize("estimator_class", [ThetaStability, ThetaContinuity])
    def test_reaches_tolerance(self, operator, estimator_class):
        estimator = estimator_class(operator)
        tol = 0.5
        flag, err, idx = greedy_constant_estimator(estimator, TRAIN, N=len(TRAIN), tol=tol)
        assert flag == 0
        assert err[-1] <= tol
        assert max(estimator.error_bound(mu) for mu in TRAIN) <= tol
        assert len(set(idx)) == len(idx) == estimator.n

    def test_maximum_number_of_updates(self, operator):
        estimator = ThetaStability(operator)
        flag, err, idx = greedy_constant_estimator(estimator, TRAIN, N=2, tol=0.0)
        assert flag == 1
        assert estimator.n == 2 and len(idx) == 2
        assert np.isnan(err[-1])

    def test_training_set_exhausted(self, operator):
        estimator = ThetaStability(operator)
        flag, _, idx = greedy_constant_estimator(estimator, TRAIN[:3], N=10, tol=-1.0)
        assert flag == 2
        assert sorted(idx) == [0, 1, 2]

    def test_callback(self, operator):
        calls = []
        greedy_constant_estimator(ThetaStability(operator), TRAIN, N=3, tol=0.0, callback=lambda est, mu: calls.append(mu))
        assert len(calls) == 3
