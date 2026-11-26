"""
Analytic test problems with known solutions and constants.

All stationary problems are of the form

.. math::
    B(\\mu) = H \\operatorname{diag}(s(\\mu)) G^T, \\qquad s(\\mu) = \\sum_q \\theta_q(\\mu) s_q,

with trial inner product matrix :math:`M_U = G G^T` and test inner product matrix :math:`M_V = H H^T`.
Substituting :math:`a = G^T u` and :math:`c = H^T v` shows that the inf-sup and continuity constants
are given by :math:`\\min_i |s_i(\\mu)|` and :math:`\\max_i |s_i(\\mu)|`, respectively. For the Galerkin case
(``H = G``) the same holds for the coercivity constant if all :math:`s_i(\\mu) > 0`.

The time-stepping problem decouples in the same way, so the discrete solutions of the
one-step schemes are known in closed form.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy as sp
from scipy.sparse.linalg import aslinearoperator

from ulmRBM.affine import AffineLinear, AffineFunction
from ulmRBM.products import MatrixInnerProduct

MATRIX_FORMATS = ['dense', 'sparse', 'linop']
"""Matrix representations every operator should support."""


def to_format(A: np.ndarray, fmt: str):
    """Convert a dense matrix to the requested representation."""
    if fmt == 'dense':
        return np.asarray(A)
    if fmt == 'sparse':
        return sp.sparse.csr_array(A)
    if fmt == 'linop':
        return aslinearoperator(A)
    raise ValueError(fmt)


def as_dense(A) -> np.ndarray:
    """Convert dense, sparse or matrix-free representations to a dense array."""
    if isinstance(A, np.ndarray):
        return A
    if sp.sparse.issparse(A):
        return A.toarray()
    return A @ np.eye(A.shape[1])


THETA = [lambda mu: mu[0], lambda mu: mu[1], lambda mu: 1.0]
r"""Positive affine coefficients :math:`\theta(\mu) = (\mu_1, \mu_2, 1)` for :math:`\mu \in [0.5, 2]^2`."""

THETA_F = [lambda mu: 1.0, lambda mu: mu[0]]
r"""Affine coefficients of right-hand side and output :math:`(1, \mu_1)`."""


def parameter_grid(k: int = 5) -> np.ndarray:
    """Tensor grid of ``k*k`` parameters in :math:`[0.5, 2]^2`."""
    x = np.linspace(0.5, 2.0, k)
    return np.array([[a, b] for a in x for b in x])


def _random_invertible(n: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q @ np.diag(rng.uniform(0.5, 2.0, n))


@dataclass
class StationaryProblem:
    r"""Affine linear system :math:`B(\mu) u = f(\mu)` with output :math:`s(\mu) = l(\mu)^T u` and known constants."""

    n: int = 8
    galerkin: bool = True
    seed: int = 0
    fmt: str = 'dense'

    G: np.ndarray = field(init=False)
    H: np.ndarray = field(init=False)
    s_terms: list = field(init=False)
    f_terms: list = field(init=False)
    l_terms: list = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.G = _random_invertible(self.n, rng)
        self.H = self.G if self.galerkin else _random_invertible(self.n, rng)
        # nonnegative diagonals => all affine terms are positive semi-definite (needed by Min-Theta / SCM)
        self.s_terms = [rng.uniform(0.0, 1.0, self.n), rng.uniform(0.0, 1.0, self.n), rng.uniform(0.2, 1.0, self.n)]
        self.f_terms = [rng.standard_normal(self.n), rng.standard_normal(self.n)]
        self.l_terms = [rng.standard_normal(self.n), rng.standard_normal(self.n)]

    # --- dense reference data
    def s(self, mu) -> np.ndarray:
        return sum(theta(mu) * s_q for theta, s_q in zip(THETA, self.s_terms))

    def B_terms_dense(self) -> list[np.ndarray]:
        return [self.H @ np.diag(s_q) @ self.G.T for s_q in self.s_terms]

    def B_dense(self, mu) -> np.ndarray:
        return self.H @ np.diag(self.s(mu)) @ self.G.T

    def f_dense(self, mu) -> np.ndarray:
        return sum(theta(mu) * f_q for theta, f_q in zip(THETA_F, self.f_terms))

    def l_dense(self, mu) -> np.ndarray:
        return sum(theta(mu) * l_q for theta, l_q in zip(THETA_F, self.l_terms))

    @property
    def MU(self) -> np.ndarray:
        return self.G @ self.G.T

    @property
    def MV(self) -> np.ndarray:
        return self.H @ self.H.T

    def solution(self, mu) -> np.ndarray:
        return np.linalg.solve(self.B_dense(mu), self.f_dense(mu))

    def dual_solution(self, mu) -> np.ndarray:
        return np.linalg.solve(self.B_dense(mu).T, -self.l_dense(mu))

    def output(self, mu) -> float:
        return self.l_dense(mu) @ self.solution(mu)

    def stability(self, mu) -> float:
        return np.min(np.abs(self.s(mu)))

    def continuity(self, mu) -> float:
        return np.max(np.abs(self.s(mu)))

    # --- ulmRBM objects
    @property
    def B(self) -> AffineLinear:
        return AffineLinear(THETA, [to_format(Bq, self.fmt) for Bq in self.B_terms_dense()])

    @property
    def f(self) -> AffineLinear:
        return AffineLinear(THETA_F, self.f_terms)

    @property
    def l(self) -> AffineLinear:
        return AffineLinear(THETA_F, self.l_terms)

    @property
    def U(self) -> MatrixInnerProduct:
        return MatrixInnerProduct(self.MU)

    @property
    def V(self) -> MatrixInnerProduct:
        return self.U if self.galerkin else MatrixInnerProduct(self.MV)

    mus = staticmethod(parameter_grid)


@dataclass
class HeatProblem:
    r"""Linear time-invariant problem :math:`M u' + A(\mu) u = f(\mu)` with closed-form discrete solutions.

    With :math:`M = G G^T`, :math:`A(\mu) = G \operatorname{diag}(a(\mu)) G^T` and :math:`f(\mu) = G g(\mu)`,
    the coordinates :math:`w = G^T u` decouple into :math:`w' + a w = g`. All one-step schemes in
    `ulmRBM.fom` share the fixed point :math:`w^* = g/a`, hence

    .. math::
        w_k = w^* + r^k (w_0 - w^*)

    with amplification factor :math:`r` of the scheme.
    """

    n: int = 6
    seed: int = 1
    T: float = 1.0
    K: int = 20

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.G = _random_invertible(self.n, rng)
        self.a_terms = [rng.uniform(0.5, 2.0, self.n), rng.uniform(0.5, 2.0, self.n)]
        self.g = rng.standard_normal(self.n)
        self.w0 = rng.standard_normal(self.n)
        self.l_vec = rng.standard_normal(self.n)

    theta = [lambda mu: mu, lambda mu: 1.0]

    def a(self, mu) -> np.ndarray:
        return mu * self.a_terms[0] + self.a_terms[1]

    @property
    def M(self) -> np.ndarray:
        return self.G @ self.G.T

    @property
    def A(self) -> AffineLinear:
        return AffineLinear(self.theta, [self.G @ np.diag(a_q) @ self.G.T for a_q in self.a_terms])

    @property
    def f(self) -> AffineFunction:
        Gg = self.G @ self.g
        return AffineFunction([1.0], [lambda t: Gg])

    @property
    def u0(self) -> AffineLinear:
        return AffineLinear([1.0], [np.linalg.solve(self.G.T, self.w0)])

    @property
    def W(self) -> MatrixInnerProduct:
        return MatrixInnerProduct(self.M)

    def amplification(self, scheme: str, mu) -> np.ndarray:
        dta = self.T / self.K * self.a(mu)
        return {'explicit_euler': 1 - dta,
                'implicit_euler': 1 / (1 + dta),
                'crank_nicolson': (1 - dta / 2) / (1 + dta / 2)}[scheme]

    def discrete_solution(self, scheme: str, mu) -> np.ndarray:
        r = self.amplification(scheme, mu)
        w_star = self.g / self.a(mu)
        k = np.arange(self.K + 1)
        w = w_star[:, None] + r[:, None] ** k[None, :] * (self.w0 - w_star)[:, None]
        return np.linalg.solve(self.G.T, w)

    def exact_solution(self, mu, t: np.ndarray) -> np.ndarray:
        w_star = self.g / self.a(mu)
        w = w_star[:, None] + np.exp(-self.a(mu)[:, None] * t[None, :]) * (self.w0 - w_star)[:, None]
        return np.linalg.solve(self.G.T, w)
