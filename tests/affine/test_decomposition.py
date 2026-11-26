"""Tests for the affine decompositions in :mod:`ulmRBM.affine`.

Every operation on an `AffineLinear` is checked by evaluating the result at several parameters and
comparing with the same operation applied to the evaluated dense reference matrices.
"""

import numpy as np
import scipy as sp
import pytest

from ulmRBM.core import TrivialParametric
from ulmRBM.core import KRON_AVAILABLE
from ulmRBM.affine import (
    AffineObject, AffineLinear, AffineFunction, ScalarComponentList,
    wrap_affinelinear, unwrap_affinelinear, multiply_theta, affine_kron,
)
from tests.problems import to_format, as_dense

MUS = [0.5, 1.0, 2.0]
RNG = np.random.default_rng(0)
A_TERMS = [RNG.standard_normal((4, 3)) for _ in range(3)]
C_TERMS = [RNG.standard_normal((3, 2)) for _ in range(2)]
THETA_A = [lambda mu: mu, lambda mu: mu**2, 2.0]
THETA_C = [lambda mu: 1 / mu, 1.0]


def ref_A(mu):
    return mu * A_TERMS[0] + mu**2 * A_TERMS[1] + 2.0 * A_TERMS[2]


def ref_C(mu):
    return C_TERMS[0] / mu + C_TERMS[1]


@pytest.fixture
def A(fmt):
    return AffineLinear(THETA_A, [to_format(Aq, fmt) for Aq in A_TERMS])


@pytest.fixture
def A_dense():
    return AffineLinear(THETA_A, A_TERMS)


@pytest.fixture
def C():
    return AffineLinear(THETA_C, C_TERMS)


def assert_affine_equal(affine, reference, mus=MUS):
    for mu in mus:
        np.testing.assert_allclose(as_dense(affine(mu)), reference(mu), rtol=1e-12, atol=1e-12)


class TestAffineObject:

    def test_evaluation(self):
        ad = AffineObject([lambda mu: mu, lambda mu: mu**2], [1.0, 2.0])
        assert ad(3.0) == 3.0 + 18.0

    def test_construct_from_pairs_and_copy(self):
        pairs = [(lambda mu: mu, 1.0), (2.0, 3.0)]
        ad = AffineObject(pairs)
        assert ad(1.0) == 7.0
        assert AffineObject(ad)(1.0) == 7.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            AffineObject([1.0, 2.0], [1.0])

    def test_sequence_interface(self):
        ad = AffineObject([lambda mu: mu, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert len(ad) == 3
        theta, data = ad[1]
        assert theta(None) == 2.0 and data == 2.0
        assert isinstance(ad[1:], AffineObject) and len(ad[1:]) == 2
        ad[0] = (lambda mu: 5.0, 10.0)
        assert ad(None) == 5 * 10 + 2 * 2 + 3 * 3
        del ad[0]
        assert ad(None) == 13.0
        ad.insert(0, (1.0, 1.0))
        assert ad(None) == 14.0
        assert [d for _, d in ad] == [1.0, 2.0, 3.0]

    def test_setitem_wraps_scalars(self):
        ad = AffineObject([1.0], [1.0])
        ad[0] = (5.0, 10.0)
        assert ad(None) == 50.0

    def test_scalars_are_wrapped(self):
        ad = AffineObject([2.0], [1.0])
        assert isinstance(ad.theta[0], TrivialParametric)
        assert not ad.is_parametric
        assert AffineObject([lambda mu: mu], [1.0]).is_parametric


class TestAffineLinearArithmetic:

    def test_evaluation(self, A):
        assert A.shape == (4, 3)
        assert_affine_equal(A, ref_A)

    def test_add_and_sub(self, A):
        assert_affine_equal(A + A, lambda mu: 2 * ref_A(mu))
        assert_affine_equal(A - 2 * A, lambda mu: -ref_A(mu))
        assert_affine_equal(-A, lambda mu: -ref_A(mu))
        assert len(A + A) == 2 * len(A)

    def test_add_pairs_and_zero(self, A_dense):
        B = A_dense + [(lambda mu: 3 * mu, A_TERMS[0])]
        assert_affine_equal(B, lambda mu: ref_A(mu) + 3 * mu * A_TERMS[0])
        assert_affine_equal(0 + A_dense, ref_A)  # enables sum(...)

    def test_scalar_multiplication(self, A):
        assert_affine_equal(2.5 * A, lambda mu: 2.5 * ref_A(mu))
        assert_affine_equal(A * 2.5, lambda mu: 2.5 * ref_A(mu))

    def test_matmul_with_matrix(self, A, rng):
        X = rng.standard_normal((3, 2))
        Y = rng.standard_normal((5, 4))
        assert_affine_equal(A @ X, lambda mu: ref_A(mu) @ X)
        assert_affine_equal(Y @ A, lambda mu: Y @ ref_A(mu))

    def test_matmul_with_vector(self, A, rng):
        x = rng.standard_normal(3)
        assert (A @ x).shape == (4,)
        assert_affine_equal(A @ x, lambda mu: ref_A(mu) @ x)

    def test_matmul_with_sparse_from_left(self, A_dense):
        Y = sp.sparse.random_array((5, 4), density=0.5, rng=0)
        assert_affine_equal(Y @ A_dense, lambda mu: Y @ ref_A(mu))

    def test_matmul_affine_affine(self, A, C):
        product = A @ C
        assert len(product) == len(A) * len(C)
        assert_affine_equal(product, lambda mu: ref_A(mu) @ ref_C(mu))

    def test_elementwise_affine_affine(self, A_dense):
        assert_affine_equal(A_dense * A_dense, lambda mu: ref_A(mu) * ref_A(mu))

    def test_transpose(self, A):
        assert A.T.shape == (3, 4)
        assert_affine_equal(A.T, lambda mu: ref_A(mu).T)
        assert A.T is A.T  # cached

    @pytest.mark.xfail(strict=True, reason="BUG: cached transpose is not invalidated by __setitem__/__delitem__")
    def test_transpose_after_modification(self, A_dense):
        _ = A_dense.T
        A_dense[0] = (1.0, np.zeros((4, 3)))
        assert_affine_equal(A_dense.T, lambda mu: (ref_A(mu) - mu * A_TERMS[0]).T)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            AffineLinear([1.0, 1.0], [np.ones(2), np.ones(3)])
        A = AffineLinear([1.0], [np.ones(2)])
        with pytest.raises(ValueError):
            A.insert(0, (1.0, np.ones(3)))


class TestCompress:

    def test_merges_constant_terms(self, A):
        compressed = A.compress()
        assert len(compressed) == 3  # mu, mu**2 and one constant term
        assert_affine_equal(compressed, ref_A)

    def test_removes_zero_terms(self):
        A = AffineLinear([0.0, lambda mu: mu, 0.0], [np.ones(2), np.ones(2), np.ones(2)])
        compressed = A.compress()
        assert len(compressed) == 1
        assert_affine_equal(compressed, lambda mu: mu * np.ones(2))

    def test_all_zero_keeps_one_term(self):
        compressed = AffineLinear([0.0, 0.0], [np.ones(2), np.ones(2)]).compress()
        assert len(compressed) == 1
        assert_affine_equal(compressed, lambda mu: np.zeros(2))

    def test_does_not_modify_constant_input_data(self):
        data = [np.ones(2), np.ones(2)]
        AffineLinear([1.0, 1.0], data).compress()
        np.testing.assert_array_equal(data[0], np.ones(2))

    @pytest.mark.xfail(strict=True, reason="BUG: compress adds zero-coefficient data terms in-place into the user's array")
    def test_does_not_modify_zero_input_data(self):
        data = [np.ones(2), np.ones(2)]
        AffineLinear([0.0, 0.0], data).compress()
        np.testing.assert_array_equal(data[0], np.ones(2))


class TestApply:

    def test_apply2data(self, A_dense):
        assert_affine_equal(A_dense.apply2data(lambda d: 2 * d.T), lambda mu: 2 * ref_A(mu).T)

    def test_apply2theta(self, A_dense):
        shifted = A_dense.apply2theta(lambda theta: (lambda mu, theta=theta: theta(mu + 1)))
        assert_affine_equal(shifted, lambda mu: ref_A(mu + 1))


class TestMultiplyTheta:

    @pytest.mark.parametrize("s1, s2, expected", [
        (2.0, 3.0, 6.0), (0.0, lambda mu: mu, 0.0), (1.0, lambda mu: mu, 5.0),
        (lambda mu: mu, 1.0, 5.0), (2.0, lambda mu: mu, 10.0), (lambda mu: mu, lambda mu: mu, 25.0),
    ])
    def test_product(self, s1, s2, expected):
        assert multiply_theta(s1, s2)(5.0) == expected

    def test_trivial_results_stay_trivial(self):
        assert isinstance(multiply_theta(2.0, 3.0), TrivialParametric)
        assert isinstance(multiply_theta(0.0, lambda mu: mu), TrivialParametric)

    def test_nested_scaling_is_flattened(self):
        f = lambda mu: mu
        nested = multiply_theta(2.0, multiply_theta(3.0, f))
        assert nested.scalar is f and nested(1.0) == 6.0


class TestAffineFunction:

    def test_evaluation(self):
        af = AffineFunction([lambda mu: mu, 2.0], [np.sin, np.cos])
        x = np.linspace(0, 1, 5)
        np.testing.assert_allclose(af(3.0)(x), 3 * np.sin(x) + 2 * np.cos(x))


class TestWrapping:

    def test_wrap_and_unwrap(self):
        A = np.eye(2)
        wrapped = wrap_affinelinear(A)
        assert isinstance(wrapped, AffineLinear) and wrapped(None) is A
        assert wrap_affinelinear(wrapped) is wrapped
        np.testing.assert_array_equal(unwrap_affinelinear(wrapped), A)

    def test_unwrap_merges_constant_terms(self):
        A = AffineLinear([1.0, 2.0], [np.eye(2), np.eye(2)])
        np.testing.assert_array_equal(unwrap_affinelinear(A), 3 * np.eye(2))

    def test_unwrap_parametric_fails(self):
        with pytest.raises(ValueError):
            unwrap_affinelinear(AffineLinear([lambda mu: mu], [np.eye(2)]))


class TestAffineKron:

    @pytest.fixture(params=['dense', 'sparse', pytest.param('implicit', marks=pytest.mark.skipif(not KRON_AVAILABLE, reason="kron not installed"))])
    def format(self, request):
        return request.param

    def test_matches_numpy_kron(self, format, A_dense, C):
        K = affine_kron(A_dense, C, format=format)
        assert len(K) == len(A_dense) * len(C)
        assert_affine_equal(K, lambda mu: np.kron(ref_A(mu), ref_C(mu)))

    def test_constant_matrices(self, format):
        X, Y = np.arange(4.0).reshape(2, 2), np.eye(3)
        K = affine_kron(X, Y, format=format)
        assert_affine_equal(K, lambda mu: np.kron(X, Y))

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            affine_kron(np.eye(2), np.eye(2), format='foo')


class TestScalarComponentList:

    def test_components_and_caching(self):
        calls = []

        def func(mu):
            calls.append(mu)
            return np.array([mu, 2 * mu, 3 * mu])

        components = ScalarComponentList(func, 3)
        assert len(components) == 3
        mu = np.array([2.0])
        assert [theta(mu) for theta in components] == [2.0, 4.0, 6.0]
        assert len(calls) == 1  # vector-valued function is only evaluated once per parameter

    def test_usable_as_theta(self):
        components = ScalarComponentList(lambda mu: np.array([mu, mu**2]), 2)
        ad = AffineLinear(list(components), [np.ones(2), np.ones(2)])
        assert_affine_equal(ad, lambda mu: (mu + mu**2) * np.ones(2))
