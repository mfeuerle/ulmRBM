"""Tests for :mod:`ulmRBM.core`."""

import numpy as np
import scipy as sp
import pytest

from ulmRBM.core import TrivialParametric, NO_MU, wrap_linear, wrap_scalar, unwrap


class TestTrivialParametric:

    def test_returns_data_for_any_parameter(self):
        A = np.eye(3)
        p = TrivialParametric(A)
        assert p(NO_MU) is A
        assert p(1.0) is A
        assert p() is A

    def test_delegates_attributes_to_data(self):
        A = sp.sparse.csr_array(np.arange(6.0).reshape(2, 3))
        p = TrivialParametric(A)
        assert p.shape == (2, 3)
        assert p.nnz == A.nnz


@pytest.mark.parametrize("value", [np.ones(3), np.eye(2), sp.sparse.eye_array(2), sp.sparse.linalg.aslinearoperator(np.eye(2))])
def test_wrap_linear_wraps_vectors_and_matrices(value):
    wrapped = wrap_linear(value)
    assert isinstance(wrapped, TrivialParametric)
    assert unwrap(wrapped) is value


@pytest.mark.parametrize("value", [2, 2.5, np.float64(3.0)])
def test_wrap_scalar_wraps_numbers(value):
    wrapped = wrap_scalar(value)
    assert isinstance(wrapped, TrivialParametric)
    assert wrapped(1.0) == value


def test_wrappers_pass_through_other_objects():
    func = lambda mu: mu
    assert wrap_linear(func) is func
    assert wrap_scalar(func) is func
    assert unwrap(func) is func
