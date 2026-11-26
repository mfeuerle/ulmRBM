"""Tests for the stationary full-order models in :mod:`ulmRBM.fom`.

The model test classes inherit `OperatorContract`, so the operator functionality (stability, continuity,
supremizer, ...) is also verified for each model. The primal-dual models inherit all tests of their
primal counterparts and additionally fulfill `PrimalDualContract`.
"""

import numpy as np
import pytest

from ulmRBM.fom import (
    FOM, GalerkinFOM, PrimalDualFOM, PrimalDualGalerkinFOM,
    ParametricOperator, ParametricGalerkinOperator,
)
from ulmRBM.solver import DirectSolver
from ulmRBM.products import MatrixInnerProduct
from tests.problems import StationaryProblem
from tests.fom.test_operators import OperatorContract

RTOL = 1e-6


class ModelContract(OperatorContract):
    """Tests for the functionality of models, i.e. operators with right-hand side and output."""

    model_class: type
    galerkin: bool

    @pytest.fixture
    def problem(self):
        return StationaryProblem(galerkin=self.galerkin)

    def test_solve(self, case, mus):
        model, problem = case
        for mu in mus:
            np.testing.assert_allclose(model.solve(mu), problem.solution(mu), rtol=RTOL)

    def test_output(self, case, mus):
        model, problem = case
        for mu in mus:
            np.testing.assert_allclose(model.output(mu), problem.output(mu), rtol=RTOL)
            np.testing.assert_allclose(model.output(mu, problem.solution(mu)), problem.output(mu), rtol=1e-12)

    def test_shape_includes_output_dimension(self, case):
        model, problem = case
        assert model.shape == (problem.n, problem.n, problem.n)
        assert model.dim == model.shape

    def test_default_solver(self, problem):
        problem.fmt = 'dense'
        model = self.make(problem, solver=None)
        mu = problem.mus()[0]
        np.testing.assert_allclose(model.solve(mu), problem.solution(mu), rtol=1e-4)  # default tolerance of GMRES

    def test_custom_solver_is_used(self, case):
        _, problem = case
        calls = []

        def solver(A, b, x0):
            calls.append(1)
            return np.linalg.solve(A, b)

        problem.fmt = 'dense'
        model = self.make(problem, solver=solver)
        np.testing.assert_allclose(model.solve(problem.mus()[0]), problem.solution(problem.mus()[0]), rtol=1e-10)
        assert calls

    def test_rhs_dimension_mismatch(self, problem):
        with pytest.raises(ValueError):
            self.make(problem, f=np.ones(problem.n + 1))


class SingleModelContract(ModelContract):
    """Models that may be defined without an output of interest."""

    def test_output_requires_functional(self, problem):
        model = self.make(problem, l=None)
        assert model.l is None
        assert model.shape[2] == 0
        with pytest.raises(ValueError):
            model.output(problem.mus()[0])


class PrimalDualContract:
    r"""Tests for primal-dual models with dual model :math:`B^T z = -l`, :math:`l_{\text{dual}} = -f`."""

    def test_dual_solve(self, case, mus):
        model, problem = case
        for mu in mus:
            np.testing.assert_allclose(model.dual.solve(mu), problem.dual_solution(mu), rtol=RTOL)

    def test_dual_output_coincides_with_primal_output(self, case, mus):
        r""":math:`-f^T z = (B u)^T B^{-1}... = l^T u`, i.e. the dual output equals the primal output."""
        model, problem = case
        for mu in mus:
            np.testing.assert_allclose(model.dual.output(mu), problem.output(mu), rtol=RTOL)

    def test_dual_spaces_are_swapped(self, case):
        model, _ = case
        assert model.dual.U is model.V
        assert model.dual.V is model.U

    def test_dual_shares_constants(self, case):
        model, problem = case
        model = self.make(problem, stability=lambda mu, op: 1.5, continuity=2.5)
        assert model.dual.stability(None) == 1.5
        assert model.dual.continuity(None) == 2.5

    def test_dual_constants_equal_primal_constants(self, case):
        """The stability and continuity constants of the primal and dual model coincide."""
        model, problem = case
        mu = problem.mus()[3]
        transposed = ParametricOperator(model.B.T, model.V, model.U, stability='direct', continuity='direct')
        np.testing.assert_allclose(transposed.stability(mu), model.stability(mu), rtol=RTOL)
        np.testing.assert_allclose(transposed.continuity(mu), model.continuity(mu), rtol=RTOL)

    def test_separate_solvers(self, case):
        _, problem = case
        model = self.make(problem, solver=[DirectSolver(), DirectSolver(factorize=True)])
        assert model._solver is not model.dual._solver
        with pytest.raises(ValueError):
            self.make(problem, solver=[DirectSolver()] * 3)

    @pytest.mark.xfail(strict=True, raises=TypeError, reason="BUG: primal-dual models crash with an obscure TypeError if no output functional is given")
    def test_output_functional_is_required(self, problem):
        with pytest.raises(ValueError):
            self.make(problem, l=None)


class TestFOM(SingleModelContract):
    galerkin = False

    def make(self, problem, **kwargs):
        args = dict(B=problem.B, f=problem.f, U=problem.U, V=problem.V, l=problem.l, solver=DirectSolver())
        args.update(kwargs)
        return FOM(**args)

    def test_from_operator(self, problem):
        op = ParametricOperator(problem.B, problem.U, problem.V, stability=lambda mu, op: 1.0)
        model = FOM(op, problem.f, l=problem.l)
        assert model.U is op.U and model.V is op.V
        assert model.stability(None) == 1.0
        mu = problem.mus()[0]
        np.testing.assert_allclose(model.solve(mu), problem.solution(mu), rtol=RTOL)

    def test_inner_products_required(self, problem):
        with pytest.raises(ValueError):
            FOM(problem.B, problem.f, problem.U)


class TestGalerkinFOM(SingleModelContract):
    galerkin = True

    def make(self, problem, **kwargs):
        args = dict(B=problem.B, f=problem.f, U=problem.U, l=problem.l, solver=DirectSolver())
        args.update(kwargs)
        return GalerkinFOM(**args)

    def test_from_operator(self, problem):
        op = ParametricGalerkinOperator(problem.B, problem.U, stability=lambda mu, op: 1.0)
        model = GalerkinFOM(op, problem.f, l=problem.l)
        assert model.U is op.U and model.V is op.U
        assert model.stability(None) == 1.0

    def test_rejects_petrov_galerkin_operator(self, problem):
        op = ParametricOperator(problem.B, problem.U, problem.U)
        with pytest.raises(ValueError):
            GalerkinFOM(op, problem.f)

    def test_inner_product_required(self, problem):
        with pytest.raises(ValueError):
            GalerkinFOM(problem.B, problem.f)


class TestPrimalDualFOM(TestFOM, PrimalDualContract):

    def make(self, problem, **kwargs):
        args = dict(B=problem.B, f=problem.f, U=problem.U, V=problem.V, l=problem.l, solver=DirectSolver())
        args.update(kwargs)
        if 'supremizer' in args: args['primal_supremizer'] = args.pop('supremizer')
        return PrimalDualFOM(**args)

    def test_dual_type(self, case):
        model, _ = case
        assert type(model.dual) is FOM

    @pytest.mark.xfail(strict=True, reason="BUG: PrimalDualFOM fails for a ParametricOperator as B (uses B.T of the operator)")
    def test_from_operator(self, problem):
        op = ParametricOperator(problem.B, problem.U, problem.V)
        model = PrimalDualFOM(op, problem.f, l=problem.l)
        mu = problem.mus()[0]
        np.testing.assert_allclose(model.dual.solve(mu), problem.dual_solution(mu), rtol=RTOL)

    # primal-dual models always need an output functional
    test_output_requires_functional = None


class TestPrimalDualGalerkinFOM(TestGalerkinFOM, PrimalDualContract):

    def make(self, problem, **kwargs):
        args = dict(B=problem.B, f=problem.f, U=problem.U, l=problem.l, solver=DirectSolver())
        args.update(kwargs)
        if 'supremizer' in args: args['primal_supremizer'] = args.pop('supremizer')
        return PrimalDualGalerkinFOM(**args)

    def test_dual_type(self, case):
        model, _ = case
        assert type(model.dual) is GalerkinFOM

    @pytest.mark.xfail(strict=True, reason="BUG: PrimalDualGalerkinFOM fails for a ParametricGalerkinOperator as B (uses B.T of the operator)")
    def test_from_operator(self, problem):
        op = ParametricGalerkinOperator(problem.B, problem.U)
        model = PrimalDualGalerkinFOM(op, problem.f, l=problem.l)
        mu = problem.mus()[0]
        np.testing.assert_allclose(model.dual.solve(mu), problem.dual_solution(mu), rtol=RTOL)

    test_output_requires_functional = None
