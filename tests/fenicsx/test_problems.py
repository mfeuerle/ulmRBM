"""Tests for the FEniCSx standard problems and norms in :mod:`ulmRBM.fenicsx`.

Skipped if FEniCSx (dolfinx) is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from ulmRBM.fenicsx import norms, FEniCSxSpaceWithDirichletBCs
from ulmRBM.fenicsx.problems import simple_elliptic, simple_heat_timestepping
from ulmRBM.fom import GalerkinFOM, ParametricGalerkinOperator, StationaryTimeSteppingGalerkinFOM
from ulmRBM.solver import DirectSolver


def free_dof_coordinates(U: FEniCSxSpaceWithDirichletBCs) -> np.ndarray:
    return U.space.tabulate_dof_coordinates()[U.dofs]


@pytest.mark.parametrize("mu", [0.5, 1.0, 3.0])
def test_elliptic_1d_is_nodally_exact(mu):
    r"""Linear finite elements are nodally exact for :math:`-u'' = 1`, :math:`u(0) = u(1) = 0`."""
    B, f, U, V = simple_elliptic(8)
    fom = GalerkinFOM(B, f, norms.h10(U), solver=DirectSolver())
    x = free_dof_coordinates(U)[:, 0]
    np.testing.assert_allclose(fom.solve(mu), x * (1 - x) / 2, atol=1e-12)


def test_elliptic_2d_symmetry():
    r"""For :math:`\mu = 1` the problem :math:`-\partial_{xx} u - \mu \partial_{yy} u = 1` is symmetric in :math:`x \leftrightarrow y`."""
    B, f, U, V = simple_elliptic([6, 6])
    fom = GalerkinFOM(B, f, norms.h10(U), solver=DirectSolver())
    u = fom.solve(1.0)
    X = free_dof_coordinates(U)[:, :2]
    swapped = np.array([np.argmin(np.linalg.norm(X - x[::-1], axis=1)) for x in X])
    np.testing.assert_allclose(u, u[swapped], atol=1e-12)
    assert not np.allclose(fom.solve(3.0), fom.solve(3.0)[swapped])


def test_l2_and_h10_norms():
    """Constants: the L2 norm is the measure of the domain and the H1_0 seminorm vanishes."""
    _, _, U, _ = simple_elliptic([4, 4])
    one = np.ones(U.dim)
    np.testing.assert_allclose(norms.l2(U.space, bcs=False).norm(None, one), 1.0, rtol=1e-12)
    np.testing.assert_allclose(norms.h10(U.space, bcs=False).norm(None, one), 0.0, atol=1e-6)
    h1 = norms.h1(U.space, bcs=False).inner(None, one)
    np.testing.assert_allclose(h1, 1.0, rtol=1e-12)
    assert norms.h10(U).shape == (sum(U.dofs), sum(U.dofs))


@pytest.mark.parametrize("method", ['IE', 'CN'])
def test_heat_converges_to_stationary_solution(method):
    r"""For :math:`u_t - \mu u_{xx} = 1`, the solution approaches :math:`x(1-x)/(2\mu)` for large times."""
    mu = 2.0
    LI, LE, b, u0, t, W = simple_heat_timestepping(K=200, nx=[16], method=method)
    Wip = norms.l2(W)
    LI = ParametricGalerkinOperator(LI, Wip)
    LE = ParametricGalerkinOperator(LE, Wip)
    fom = StationaryTimeSteppingGalerkinFOM(LI, LE, b, u0, t)
    u = fom.solve(mu)
    x = free_dof_coordinates(W)[:, 0]
    np.testing.assert_allclose(u.u[:, -1], x * (1 - x) / (2 * mu), atol=1e-3)
    np.testing.assert_allclose(u.u[:, 0], 0.0)
