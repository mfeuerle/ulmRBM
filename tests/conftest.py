import numpy as np
import pytest

from tests.problems import MATRIX_FORMATS, StationaryProblem, HeatProblem, parameter_grid


@pytest.fixture(params=MATRIX_FORMATS)
def fmt(request) -> str:
    """Matrix representation (dense, sparse or matrix-free) used for the system matrices."""
    return request.param


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def mus() -> np.ndarray:
    """Small set of test parameters in [0.5, 2]^2."""
    return parameter_grid(3)


@pytest.fixture
def galerkin_problem() -> StationaryProblem:
    return StationaryProblem(galerkin=True)


@pytest.fixture
def petrov_galerkin_problem() -> StationaryProblem:
    return StationaryProblem(galerkin=False)


@pytest.fixture
def heat_problem() -> HeatProblem:
    return HeatProblem()
