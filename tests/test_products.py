"""Tests for :mod:`ulmRBM.products`.

All inner products are checked against a dense reference matrix :math:`M(\\mu)` via `InnerProductContract`.
Each concrete inner product is set up as a ``case`` ``(inner_product, M_ref)``, where ``M_ref(mu)`` returns
the dense reference matrix. The contract also checks the derived inner products (dual and restricted),
so that each implementation of ``_get_inverse`` and ``_restrict`` is covered.
"""

import numpy as np
import scipy as sp
import pytest

from ulmRBM.affine import AffineLinear
from ulmRBM.products import (
    InnerProduct, EuclideanInnerProduct, MatrixInnerProduct, OperatorInnerProduct,
    InverseInnerProduct, RestrictedInnerProduct, orthonormalize,
)
from ulmRBM.solver import DirectSolver
from tests.problems import to_format, as_dense

N = 6
MUS = [0.5, 1.0, 2.0]
RTOL = 1e-8


def spd(seed, n=N):
    X = np.random.default_rng(seed).standard_normal((n, n))
    return X @ X.T + n * np.eye(n)


M0, M1 = spd(0), spd(1)
B0 = spd(2) + np.triu(np.ones((N, N)))  # invertible, nonsymmetric
BASIS = np.random.default_rng(3).standard_normal((N, 3))


class InnerProductContract:
    """Common tests for all inner products, provided a ``case`` fixture ``(ip, M_ref)``."""

    @pytest.fixture(params=MUS)
    def mu(self, request):
        return request.param

    def test_shape(self, case):
        ip, M_ref = case
        assert ip.shape == M_ref(MUS[0]).shape

    def test_call_is_matrix(self, case, mu):
        ip, M_ref = case
        np.testing.assert_allclose(as_dense(ip(mu)), M_ref(mu), rtol=RTOL)

    def test_riesz(self, case, mu, rng):
        ip, M_ref = case
        u = rng.standard_normal((N, 2))
        np.testing.assert_allclose(ip.riesz(mu, u), M_ref(mu) @ u, rtol=RTOL)
        np.testing.assert_allclose(ip.riesz(mu, u[:, 0]), M_ref(mu) @ u[:, 0], rtol=RTOL)

    def test_inner(self, case, mu, rng):
        ip, M_ref = case
        u, v = rng.standard_normal((N, 2)), rng.standard_normal((N, 3))
        np.testing.assert_allclose(ip.inner(mu, u, v), u.T @ M_ref(mu) @ v, rtol=RTOL)
        np.testing.assert_allclose(ip.inner(mu, u[:, 0], v[:, 0]), u[:, 0] @ M_ref(mu) @ v[:, 0], rtol=RTOL)
        np.testing.assert_allclose(ip.inner(mu, u), u.T @ M_ref(mu) @ u, rtol=RTOL)

    def test_norm(self, case, mu, rng):
        ip, M_ref = case
        u = rng.standard_normal((N, 3))
        expected = np.sqrt(np.einsum('ik,ij,jk->k', u, M_ref(mu), u))
        np.testing.assert_allclose(ip.norm(mu, u), expected, rtol=RTOL)
        np.testing.assert_allclose(ip.norm(mu, u[:, 0]), expected[0], rtol=RTOL)

    def test_dual_is_inverse(self, case, mu, rng):
        ip, M_ref = case
        u = rng.standard_normal((N, 2))
        np.testing.assert_allclose(ip.dual.riesz(mu, u), np.linalg.solve(M_ref(mu), u), rtol=1e-6)

    def test_dual_of_dual_is_self(self, case):
        ip, _ = case
        assert ip.dual.dual is ip

    def test_restrict(self, case, mu, rng):
        ip, M_ref = case
        restricted = ip.restrict(BASIS)
        assert restricted.shape == (BASIS.shape[1], BASIS.shape[1])
        np.testing.assert_allclose(as_dense(restricted(mu)), BASIS.T @ M_ref(mu) @ BASIS, rtol=RTOL)
        u = rng.standard_normal(BASIS.shape[1])
        np.testing.assert_allclose(restricted.norm(mu, u), np.sqrt((BASIS @ u) @ M_ref(mu) @ (BASIS @ u)), rtol=RTOL)

    def test_restrict_twice(self, case, mu):
        ip, M_ref = case
        P = np.random.default_rng(4).standard_normal((BASIS.shape[1], 2))
        twice = ip.restrict(BASIS).restrict(P)
        np.testing.assert_allclose(as_dense(twice(mu)), P.T @ BASIS.T @ M_ref(mu) @ BASIS @ P, rtol=RTOL)

    def test_is_parametric(self, case):
        ip, M_ref = case
        constant = all(np.allclose(M_ref(MUS[0]), M_ref(mu)) for mu in MUS)
        assert ip.is_parametric is not constant

    def test_explicit_dual_is_used_and_restricted(self, case, mu, rng):
        ip, M_ref = case
        if ip._dual is not None:
            pytest.skip("dual already linked")
        explicit = MatrixInnerProduct(np.linalg.inv(M_ref(MUS[0]))) if not ip.is_parametric else None
        if explicit is None:
            pytest.skip("requires a parameter-independent inner product")
        ip.dual = explicit
        assert ip.dual is explicit
        np.testing.assert_allclose(as_dense(ip.restrict(BASIS).dual(mu)),
                                   np.linalg.inv(BASIS.T @ M_ref(mu) @ BASIS), rtol=1e-4)  # default iterative solver


class TestEuclideanInnerProduct(InnerProductContract):

    @pytest.fixture
    def case(self):
        return EuclideanInnerProduct(N), lambda mu: np.eye(N)

    def test_dual_is_self(self, case):
        ip, _ = case
        assert ip.dual is ip


class TestMatrixInnerProduct(InnerProductContract):

    @pytest.fixture(params=['dense', 'sparse', 'linop'])
    def case(self, request):
        return MatrixInnerProduct(to_format(M0, request.param), solver=DirectSolver()), lambda mu: M0


class TestParametricMatrixInnerProduct(InnerProductContract):

    @pytest.fixture(params=['dense', 'sparse'])
    def case(self, request):
        M = AffineLinear([lambda mu: mu, 1.0], [to_format(M0, request.param), to_format(M1, request.param)])
        return MatrixInnerProduct(M, solver=DirectSolver()), lambda mu: mu * M0 + M1


class TestInverseInnerProduct(InnerProductContract):

    @pytest.fixture
    def case(self):
        return InverseInnerProduct(MatrixInnerProduct(M0), solver=DirectSolver()), lambda mu: np.linalg.inv(M0)

    def test_default_solver_warns(self):
        ip = InverseInnerProduct(MatrixInnerProduct(M0))
        with pytest.warns(UserWarning, match="default solver"):
            ip.riesz(None, np.ones(N))


class TestRestrictedInnerProduct(InnerProductContract):
    """Restriction of a larger inner product, i.e. a nested restriction in `test_restrict`."""

    @pytest.fixture(params=['constant', 'parametric'])
    def case(self, request):
        P = np.random.default_rng(5).standard_normal((N + 2, N))
        M_big = spd(6, N + 2)
        if request.param == 'constant':
            return RestrictedInnerProduct(P, MatrixInnerProduct(M_big, solver=DirectSolver())), lambda mu: P.T @ M_big @ P
        Pmu = AffineLinear([lambda mu: mu, 1.0], [P, np.eye(N + 2, N)])
        return (RestrictedInnerProduct(Pmu, MatrixInnerProduct(M_big, solver=DirectSolver())),
                lambda mu: (mu * P + np.eye(N + 2, N)).T @ M_big @ (mu * P + np.eye(N + 2, N)))


class TestOperatorInnerProduct(InnerProductContract):
    r"""Inner product :math:`B^T V B`, whose dual is computed via :math:`B^{-1} V^{-1} B^{-T}`."""

    @pytest.fixture(params=['V', 'V_dual'])
    def case(self, request):
        V = MatrixInnerProduct(M0, solver=DirectSolver())
        if request.param == 'V':
            return OperatorInnerProduct(B0, V, solver=DirectSolver()), lambda mu: B0.T @ M0 @ B0
        Vinv = np.linalg.inv(M0)
        return OperatorInnerProduct(B0, V.dual, solver=DirectSolver()), lambda mu: B0.T @ Vinv @ B0

    def test_requires_square_operator(self):
        with pytest.raises(ValueError):
            OperatorInnerProduct(np.ones((N, N - 1)), MatrixInnerProduct(M0))


def test_inner_product_must_be_square():
    with pytest.raises(ValueError):
        MatrixInnerProduct(np.ones((2, 3)))


class TestOrthonormalize:

    @pytest.fixture(params=['euclidean', 'matrix', 'raw_matrix'])
    def ip_and_M(self, request):
        if request.param == 'euclidean':
            return EuclideanInnerProduct(N), np.eye(N)
        if request.param == 'matrix':
            return MatrixInnerProduct(M0), M0
        return M0, M0

    def test_orthonormal_and_same_span(self, ip_and_M):
        ip, M = ip_and_M
        basis_orth, Q = orthonormalize(BASIS, ip)
        np.testing.assert_allclose(basis_orth.T @ M @ basis_orth, np.eye(BASIS.shape[1]), atol=1e-10)
        np.testing.assert_allclose(basis_orth, BASIS @ Q, rtol=1e-10, atol=1e-12)

    def test_rejects_parametric_inner_product(self):
        ip = MatrixInnerProduct(AffineLinear([lambda mu: mu], [M0]))
        with pytest.raises(ValueError):
            orthonormalize(BASIS, ip)
