"""Tests for :func:`ulmRBM.affine.empirical_interpolation`."""

import numpy as np
import pytest

from ulmRBM.affine import AffineLinear, AffineFunction, empirical_interpolation

POINTS = np.linspace(0, 1, 41)
MUS_TRAIN = np.linspace(0.1, 10, 30)
MUS_TEST = np.linspace(0.15, 9.5, 7)


def affine_func(mu, x):
    """Function with exact affine decomposition of length 2."""
    x = np.asarray(x)
    return mu * np.sin(3 * x) + mu**2 * x**2


def nonaffine_func(mu, x):
    return 1 / (1 + mu * np.asarray(x))


def max_error(approx, func, mus, points=POINTS):
    if isinstance(approx, AffineFunction):
        return max(np.max(np.abs(approx(mu)(points) - func(mu, points))) for mu in mus)
    return max(np.max(np.abs(approx(mu) - func(mu, points))) for mu in mus)


@pytest.fixture(params=[False, True], ids=['discrete', 'continuous'])
def continuous(request):
    return request.param


@pytest.fixture(params=[None, False, True], ids=['residual-default', 'no-residual', 'residual'])
def residual(request):
    return request.param


def test_return_type(continuous):
    approx = empirical_interpolation(affine_func, MUS_TRAIN, POINTS, continuous=continuous)
    assert isinstance(approx, AffineFunction if continuous else AffineLinear)


def test_recovers_affine_function_exactly(continuous, residual):
    approx = empirical_interpolation(affine_func, MUS_TRAIN, POINTS, tol=1e-10, continuous=continuous, residual=residual)
    assert len(approx) == 2
    assert max_error(approx, affine_func, MUS_TEST) < 1e-9


def test_reaches_tolerance_on_training_set(continuous, residual):
    tol = 1e-6
    approx = empirical_interpolation(nonaffine_func, MUS_TRAIN, POINTS, tol=tol, continuous=continuous, residual=residual)
    assert len(approx) < len(MUS_TRAIN)
    assert max_error(approx, nonaffine_func, MUS_TRAIN) < 10 * tol
    assert max_error(approx, nonaffine_func, MUS_TEST) < 1e-3


def test_continuous_approximation_at_new_points():
    approx = empirical_interpolation(nonaffine_func, MUS_TRAIN, POINTS, tol=1e-8, continuous=True)
    x_new = np.linspace(0.01, 0.99, 17)
    assert max_error(approx, nonaffine_func, MUS_TEST, x_new) < 1e-4


def test_respects_maximum_number_of_terms():
    approx = empirical_interpolation(nonaffine_func, MUS_TRAIN, POINTS, Nmax=3, tol=1e-14)
    assert len(approx) == 3


def test_vector_valued_range():
    func = lambda mu, x: np.stack([affine_func(mu, x), 2 * affine_func(mu, x)], axis=-1)
    approx = empirical_interpolation(func, MUS_TRAIN, POINTS, tol=1e-10)
    for mu in MUS_TEST:
        np.testing.assert_allclose(approx(mu), func(mu, POINTS), atol=1e-9)


def test_rejects_non_numeric_values():
    with pytest.raises(ValueError):
        empirical_interpolation(lambda mu, x: np.array(['a'] * len(x)), MUS_TRAIN, POINTS)
