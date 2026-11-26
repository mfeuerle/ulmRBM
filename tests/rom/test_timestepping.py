"""Tests for :class:`ulmRBM.rom.StationaryTimeSteppingGalerkinROM`."""

import numpy as np
import pytest

from ulmRBM.rom import StationaryTimeSteppingGalerkinROM, ExactStability, ExactContinuity
from ulmRBM.affine import wrap_affinelinear
from ulmRBM.products import EuclideanInnerProduct
from tests.fom.test_timestepping import build_fom, SCHEMES

MUS = [0.6, 1.3, 1.9]
SNAPSHOT_MU = 1.0


@pytest.fixture(params=list(SCHEMES))
def scheme(request):
    return request.param


@pytest.fixture
def fom(heat_problem, scheme):
    return build_fom(heat_problem, scheme, l=wrap_affinelinear(heat_problem.l_vec))


def make_rom(fom, **kwargs):
    kwargs.setdefault('residual', 'full')
    return StationaryTimeSteppingGalerkinROM(fom, ExactStability(fom.LI), ExactContinuity(fom.LE), **kwargs)


@pytest.fixture
def rom(fom):
    """ROM with a (non-exact) two-dimensional basis."""
    rom = make_rom(fom)
    rom.add_basis(fom.solve(SNAPSHOT_MU).u[:, [2, -1]])
    return rom


@pytest.fixture
def full_rom(fom, heat_problem):
    rom = make_rom(fom)
    rom.add_basis(np.linalg.qr(np.random.default_rng(0).standard_normal((heat_problem.n, heat_problem.n)))[0])
    return rom


def test_add_basis(fom, heat_problem):
    rom = make_rom(fom)
    assert rom.n == 0
    rom.add_basis(np.ones(heat_problem.n))
    rom.add_basis(np.eye(heat_problem.n)[:, :2])
    assert rom.n == 3 and rom.W_basis.shape == (heat_problem.n, 3)
    with pytest.raises(ValueError):
        rom.add_basis(np.ones(heat_problem.n + 1))


def test_estimators_must_belong_to_fom(fom):
    with pytest.raises(ValueError):
        StationaryTimeSteppingGalerkinROM(fom, ExactStability(fom.LE), ExactContinuity(fom.LE))
    with pytest.raises(ValueError):
        StationaryTimeSteppingGalerkinROM(fom, ExactStability(fom.LI), ExactContinuity(fom.LI))


def test_affine_residual_falls_back_with_warning(fom):
    with pytest.warns(UserWarning, match="not implemented"):
        make_rom(fom, residual='affine')


def test_full_basis_reproduces_fom(full_rom, fom):
    for mu in MUS:
        u = full_rom.reconstruct(mu)
        np.testing.assert_allclose(u.u, fom.solve(mu).u, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(u.t, fom.t)
        assert np.max(full_rom.error(mu)) < 1e-10
        assert np.max(full_rom.error_bound(mu)) < 1e-6


def test_galerkin_orthogonality_per_time_step(rom, fom):
    W = rom.W_basis
    for mu in MUS:
        u = rom.reconstruct(mu).u
        r = fom.LE.B(mu) @ u[:, :-1] + fom.b(mu) - fom.LI.B(mu) @ u[:, 1:]
        np.testing.assert_allclose(W.T @ r, 0, atol=1e-10)


def test_initial_value_is_orthogonal_projection(rom, fom, heat_problem):
    W, M = rom.W_basis, heat_problem.M
    for mu in MUS:
        u0 = fom.u0(mu)
        expected = W @ np.linalg.solve(W.T @ M @ W, W.T @ M @ u0)
        np.testing.assert_allclose(rom.reconstruct(mu).u[:, 0], expected, rtol=1e-8, atol=1e-12)


def test_error_matches_definition(rom, fom, heat_problem):
    for mu in MUS:
        e = fom.solve(mu).u - rom.reconstruct(mu).u
        expected = np.sqrt(np.einsum('ik,ij,jk->k', e, heat_problem.M, e))
        np.testing.assert_allclose(rom.error(mu), expected, rtol=1e-8, atol=1e-14)


def test_error_bound_is_rigorous(rom):
    for mu in MUS:
        error, bound = rom.error(mu), rom.error_bound(mu)
        assert error.shape == bound.shape == (rom.K + 1,)
        assert np.all(error <= bound * (1 + 1e-8) + 1e-14)


def test_orthonormalize(rom, heat_problem):
    before = [rom.reconstruct(mu).u for mu in MUS]
    rom.orthonormalize()
    np.testing.assert_allclose(rom.W_basis.T @ heat_problem.M @ rom.W_basis, np.eye(rom.n), atol=1e-10)
    for mu, u in zip(MUS, before):
        np.testing.assert_allclose(rom.reconstruct(mu).u, u, rtol=1e-8, atol=1e-10)
    rom.orthonormalize(EuclideanInnerProduct(heat_problem.n))
    np.testing.assert_allclose(rom.W_basis.T @ rom.W_basis, np.eye(rom.n), atol=1e-10)


@pytest.mark.xfail(strict=True, reason="BUG: the reduced LI is constructed with the solver as stability argument")
def test_reduced_operators_keep_constants(full_rom, fom):
    np.testing.assert_allclose(full_rom.LI.stability(MUS[0]), fom.LI.stability(MUS[0]), rtol=1e-6)


@pytest.mark.xfail(strict=True, reason="BUG: the ROM does not reduce the output functional l of the FOM")
def test_output(full_rom, fom):
    np.testing.assert_allclose(full_rom.output(MUS[0]), fom.output(MUS[0]), rtol=1e-8)
