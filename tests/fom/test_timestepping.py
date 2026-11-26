"""Tests for the time-stepping full-order models in :mod:`ulmRBM.fom`.

The tests use `tests.problems.HeatProblem`, for which the discrete solutions of all one-step schemes
are known in closed form.
"""

import numpy as np
import pytest

from ulmRBM.fom import (
    ParametricGalerkinOperator, TimeSteppingSolution,
    StationaryTimeSteppingGalerkinFOM, DualStationaryTimeSteppingGalerkinFOM, PrimalDualStationaryTimeSteppingGalerkinFOM,
    theta_scheme, explicit_euler, implicit_euler, crank_nicolson,
)
from ulmRBM.solver import DirectSolver
from ulmRBM.affine import wrap_affinelinear
from tests.problems import HeatProblem, SCHEME_THETA

SCHEMES = {'explicit_euler': explicit_euler, 'implicit_euler': implicit_euler, 'crank_nicolson': crank_nicolson,
           'theta_scheme_0.3': lambda *args: theta_scheme(*args, theta=SCHEME_THETA['theta_scheme_0.3'])}
MUS = [0.5, 1.0, 2.0]


def build_fom(problem: HeatProblem, scheme: str, model_class=StationaryTimeSteppingGalerkinFOM, **kwargs):
    LI, LE, b, t = SCHEMES[scheme](problem.A, problem.M, problem.f, (0, problem.T), problem.K)
    W = problem.W
    LI = ParametricGalerkinOperator(LI, W, stability='direct', continuity='direct')
    LE = ParametricGalerkinOperator(LE, W, stability='direct', continuity='direct')
    return model_class(LI, LE, b, problem.u0, t, **kwargs)


@pytest.fixture(params=list(SCHEMES))
def scheme(request):
    return request.param


class TestSchemes:

    def test_time_grid(self, heat_problem, scheme):
        _, _, b, t = SCHEMES[scheme](heat_problem.A, heat_problem.M, heat_problem.f, (0, heat_problem.T), heat_problem.K)
        np.testing.assert_allclose(t, np.linspace(0, heat_problem.T, heat_problem.K + 1))
        assert b.shape == (heat_problem.n, heat_problem.K)

    def test_operators(self, heat_problem, scheme):
        p = heat_problem
        LI, LE, _, _ = SCHEMES[scheme](p.A, p.M, p.f, (0, p.T), p.K)
        dt, mu = p.T / p.K, 1.3
        A = p.A(mu)
        theta = SCHEME_THETA[scheme]
        np.testing.assert_allclose(LI(mu), p.M + theta * dt * A, rtol=1e-12)
        np.testing.assert_allclose(LE(mu), p.M - (1 - theta) * dt * A, rtol=1e-12)

    @pytest.mark.parametrize("alias, theta", [(explicit_euler, 0.0), (implicit_euler, 1.0), (crank_nicolson, 0.5)])
    def test_aliases(self, heat_problem, alias, theta):
        p, mu = heat_problem, 1.3
        args = (p.A, p.M, p.f, (0, p.T), p.K)
        for X, Y in zip(alias(*args)[:3], theta_scheme(*args, theta=theta)[:3]):
            np.testing.assert_array_equal(X(mu), Y(mu))

    def test_affine_terms_are_not_padded(self, heat_problem):
        """Terms with zero weight are not added, e.g. the implicit operator of explicit Euler is the mass matrix only."""
        p = heat_problem
        LI, LE, _, _ = explicit_euler(p.A, p.M, p.f, (0, p.T), p.K)
        assert len(LI) == 1
        LI, LE, _, _ = implicit_euler(p.A, p.M, p.f, (0, p.T), p.K)
        assert len(LE) == 1

    @pytest.mark.parametrize("theta", [-0.1, 1.1])
    def test_invalid_theta(self, heat_problem, theta):
        p = heat_problem
        with pytest.raises(ValueError):
            theta_scheme(p.A, p.M, p.f, (0, p.T), p.K, theta)

    @pytest.mark.parametrize("scheme, order", [('explicit_euler', 1), ('implicit_euler', 1), ('crank_nicolson', 2), ('theta_scheme_0.3', 1)])
    def test_convergence_order(self, scheme, order):
        errors = []
        for K in [40, 80, 160]:
            p = HeatProblem(K=K)
            u = build_fom(p, scheme).solve(1.0)
            errors.append(np.max(np.abs(u.u - p.exact_solution(1.0, u.t))))
        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        np.testing.assert_allclose(rates, order, atol=0.15)


class TestStationaryTimeSteppingGalerkinFOM:

    @pytest.fixture
    def fom(self, heat_problem, scheme):
        return build_fom(heat_problem, scheme, l=wrap_affinelinear(heat_problem.l_vec))

    def test_dimensions(self, fom, heat_problem):
        assert fom.n == heat_problem.n
        assert fom.K == heat_problem.K
        assert fom.W is fom.LI.U

    @pytest.mark.parametrize("mu", MUS)
    def test_solve_matches_discrete_solution(self, fom, heat_problem, scheme, mu):
        u = fom.solve(mu)
        assert isinstance(u, TimeSteppingSolution)
        np.testing.assert_allclose(u.t, fom.t)
        np.testing.assert_allclose(u.u, heat_problem.discrete_solution(scheme, mu), rtol=1e-8, atol=1e-10)

    def test_solution_iteration(self, fom):
        u = fom.solve(1.0)
        pairs = list(u)
        assert len(pairs) == fom.K + 1
        t, uk = pairs[3]
        assert t == u.t[3]
        np.testing.assert_array_equal(uk, u.u[:, 3])

    @pytest.mark.parametrize("mu", MUS)
    def test_output_is_final_time_output(self, fom, heat_problem, scheme, mu):
        expected = heat_problem.l_vec @ heat_problem.discrete_solution(scheme, mu)[:, -1]
        np.testing.assert_allclose(fom.output(mu), expected, rtol=1e-8)
        np.testing.assert_allclose(fom.output(mu, fom.solve(mu)), expected, rtol=1e-8)

    def test_output_requires_functional(self, heat_problem, scheme):
        fom = build_fom(heat_problem, scheme)
        with pytest.raises(ValueError):
            fom.output(1.0)


class TestPrimalDualStationaryTimeSteppingGalerkinFOM(TestStationaryTimeSteppingGalerkinFOM):
    """Inherits all primal tests and checks the dual problem additionally."""

    @pytest.fixture
    def fom(self, heat_problem, scheme):
        return build_fom(heat_problem, scheme, PrimalDualStationaryTimeSteppingGalerkinFOM,
                         solver=[DirectSolver(factorize=True), DirectSolver(factorize=True)], l=wrap_affinelinear(heat_problem.l_vec))

    def test_dual_type(self, fom):
        assert isinstance(fom.dual, DualStationaryTimeSteppingGalerkinFOM)
        assert fom.dual.K == fom.K and fom.dual.n == fom.n

    @pytest.mark.parametrize("mu", MUS)
    def test_dual_recursion(self, fom, heat_problem, mu):
        r"""The dual solution satisfies :math:`L_I^T z_K = l` and :math:`L_I^T z_{k-1} = L_E^T z_k`."""
        z = fom.dual.solve(mu).u
        LI, LE = fom.LI.B(mu), fom.LE.B(mu)
        np.testing.assert_allclose(LI.T @ z[:, -1], heat_problem.l_vec, atol=1e-10)
        np.testing.assert_allclose(LI.T @ z[:, :-1], LE.T @ z[:, 1:], atol=1e-10)

    @pytest.mark.parametrize("mu", MUS)
    def test_primal_dual_identity(self, fom, mu):
        r"""The output can be computed from the dual solution: :math:`l^T u_K = z_1^T L_E u_0 + \sum_k z_{k+1}^T b_k`."""
        z = fom.dual.solve(mu).u
        b = fom.b(mu)
        dual_output = z[:, 1] @ (fom.LE.B(mu) @ fom.u0(mu)) + np.sum(z[:, 1:] * b)
        np.testing.assert_allclose(dual_output, fom.output(mu), rtol=1e-8)

    def test_output_requires_functional(self, heat_problem, scheme):
        with pytest.raises(AssertionError):
            build_fom(heat_problem, scheme, PrimalDualStationaryTimeSteppingGalerkinFOM)

    def test_dual_default_solver(self, heat_problem, scheme):
        fom = build_fom(heat_problem, scheme, l=wrap_affinelinear(heat_problem.l_vec))
        dual = DualStationaryTimeSteppingGalerkinFOM(fom.LI, fom.LE, heat_problem.l_vec, fom.t)
        dual.solve(1.0)


def test_default_solvers_are_not_shared(heat_problem):
    """Factorizing default solvers cache the factorized matrix, so they must not be shared between models."""
    fom1 = build_fom(heat_problem, 'implicit_euler')
    fom2 = build_fom(heat_problem, 'implicit_euler')
    assert fom1._solver is not fom2._solver
    pd = build_fom(heat_problem, 'implicit_euler', PrimalDualStationaryTimeSteppingGalerkinFOM, l=heat_problem.l_vec)
    assert pd._solver is not pd.dual._solver
