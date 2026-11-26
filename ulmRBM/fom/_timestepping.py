"""
Time-stepping (full-order) models for initial value problems.
"""

from __future__ import annotations


__all__ = [
    'TimeSteppingSolution',
    'StationaryTimeSteppingGalerkinFOM',
    'theta_scheme',
    'explicit_euler',
    'implicit_euler',
    'crank_nicolson',
    'DualStationaryTimeSteppingGalerkinFOM',
    'PrimalDualStationaryTimeSteppingGalerkinFOM',
]

from collections.abc import Callable
from typing import Generic, Optional
from dataclasses import dataclass

import numpy as np

from ulmRBM.core import (
    Mu, Matrix, Vector, ParametricLinear
)
from ulmRBM.solver import Solver, DirectSolver, wrap_solver
from ulmRBM.affine import AffineLinear, AffineFunction, wrap_affinelinear
from ulmRBM.products import InnerProduct, MatrixInnerProduct

from ._models import FOM, GalerkinFOM
from ._operators import ParametricGalerkinOperator, ParametricOperator

    
@dataclass
class TimeSteppingSolution(Generic[Mu]):
    r"""Solution of a time-stepping problem for a given parameter.
    """
    t: np.ndarray[float]
    r"""Time points at which the solution is computed."""
    u: np.ndarray[float]
    r"""Solution values at the time points in `t`, i.e. ``u[:,k]`` is the solution at time ``t[k]``."""
    s: Optional[np.ndarray[float]] = None
    r"""Output values at the time points in `t`, i.e. ``s(u[:,k])=lu[:,k]``."""
    
    def __iter__(self) -> zip[float, np.ndarray[float]]:
        r"""Iterate over the time points and corresponding solution values."""
        return zip(self.t, self.u.T)
    
    
class StationaryTimeSteppingGalerkinFOM(Generic[Mu]):
    r"""Model for time-stepping problems with a Galerkin operator and stationary operators.
    
    For some time-interval :math:`I = [t_0, t_K]`, two Galerkin operators :math:`M(\mu), A(\mu) : W \to W'`, a right-hand side :math:`f(\mu) \in C(I;W')` and a initial value :math:`u_0(\mu) \in W`, consider the problem of finding :math:`u(\mu) \in C^1(I;W)` such that
    
    .. math::
        \begin{aligned}
        u(t_0; \mu) &= u_0(\mu),\\
        M(\mu) u'(t; \mu) + A(\mu) u(t; \mu) &= f(t; \mu)\quad\text{in $W'$} &&\text{for } t \in (t_0, t_K],
        \end{aligned}
    
    which can be solved by a time-stepping scheme.
    
    For a given sequence of time points :math:`t_0, ..., t_K` in :math:`I`, this class represents a abstract time-stepping scheme of finding :math:`u_k(\mu) \approx u(t_k; \mu)` for :math:`k=0,\ldots,K` such that
    
    .. math::
        \begin{aligned}
        u_0(\mu) &= u_0(\mu),\\
        \mathcal{L}^I(\mu) u_{k+1}(\mu) &= \mathcal{L}^E(\mu) u_k(\mu) + b_k(\mu)\quad\text{in $W'$} &&\text{for } k=0,\ldots,K-1.
        \end{aligned}
        
    where :math:`\mathcal{L}^I(\mu), \mathcal{L}^E(\mu) : W \to W'` are the implicit and explicit part of the time-stepping scheme, respectively, and :math:`b_k(\mu) \in W'` is the inhomogeneity at time :math:`t_k`. This model is called "stationary", since the operators :math:`\mathcal{L}^I(\mu)` and :math:`\mathcal{L}^E(\mu)` are independent of the time step :math:`k`, i.e. they only depend on the parameter :math:`\mu` but not on the time points :math:`t_k` and are thus stationary in time. Typically, this is the case if the time-steps are equidistant.
    """
    
    LI: ParametricGalerkinOperator[Mu, Matrix]
    r"""Implicit part of the time-stepping scheme, i.e. the operator :math:`\mathcal{L}^I(\mu):W\to W'` applied to the solution at the next time step."""
    LE: ParametricGalerkinOperator[Mu, Matrix]
    r"""Explicit part of the time-stepping scheme, i.e. the operator :math:`\mathcal{L}^E(\mu):W\to W'` applied to the solution at the current time step."""
    b: AffineLinear[Mu, Vector]
    """Inhomogeneity of the time-stepping scheme at each time step."""
    u0: AffineLinear[Mu, Vector]
    """Initial value."""
    t: np.ndarray[float]
    """Time points at which the solution is approximated."""
    W: InnerProduct[Mu]
    """Inner product on the space :math:`W`."""
    n: int
    """Dimension of the space :math:`W`."""
    K: int
    """Number of time intervals, i.e. ``K+1`` time points."""
    l: AffineLinear[Mu, Vector]
    r"""Affine decomposition of the output(s) of interest functional :math:`l(\mu) = \sum_{q=1}^{Q_l} \theta_q^l(\mu) l_q`. Might be ``None``, if no output of interest is given."""
    
    
    @property
    def n(self) -> int:
        return self.W.shape[0]
    
    @property
    def K(self) -> int:
        return len(self.t)-1
    
    
    def __init__(self, 
                 LI: ParametricGalerkinOperator[Mu, Matrix],
                 LE: ParametricGalerkinOperator[Mu, Matrix],
                 b: AffineLinear[Mu, Vector],
                 u0: AffineLinear[Mu, Vector],
                 t: np.ndarray[float],
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None,
                 l: AffineLinear[Mu, Vector] | None = None):
        r"""
        Args:
            LI:
                Implicit part of the time-stepping scheme, i.e. the operator applied to the solution at the next time step.
            LE:
                Explicit part of the time-stepping scheme, i.e. the operator applied to the solution at the current time step.
            b:
                Inhomogeneity at each iteration, i.e. ``b(mu)`` has shape ``(n, K)``, with ``b(mu)[:,k]`` being :math:`b_k(\mu)`.
            u0:
                Initial value at ``t_0``.
            t:
                Time points ``t_0, ..., t_K`` at which the solution is approximated.
            solver:
                Solver for the linear system per time step. Defaults to a factorizing `DirectSolver`.
            l:
                Affine decomposition of the output(s) of interest functional :math:`l(\mu) = \sum_{q=1}^{Q_l} \theta_q^l(\mu) l_q`. Might be ``None``, if no output of interest is given.
        """
        assert LI.shape == LE.shape
        assert LI.V is LE.V is LE.U is LI.U
        assert b.shape == (LI.shape[0], len(t)-1)
        
        self.LI = LI
        self.LE = LE
        self.b = wrap_affinelinear(b).compress()
        self.u0 = wrap_affinelinear(u0).compress()
        self.t = t
        self.W = LI.U
        self._solver = wrap_solver(solver if solver is not None else DirectSolver(factorize=True))
        self.l = wrap_affinelinear(l) if l is not None else None
        
    def solve(self, mu: Mu) -> TimeSteppingSolution[Mu]:
        r"""Solve the time-stepping problem for a given parameter.
        
        Args:
            mu:
                Parameter for which to solve the time-stepping problem.
                
        Returns:
            Solution for the given parameter.
        """
        
        LI = self.LI.B(mu)
        LE = self.LE.B(mu)
        b = self.b(mu)
        
        u = np.zeros((self.n, self.K+1))
        u[:,0] = self.u0(mu)
        for k in range(self.K):
            u[:,k+1] = self._solver(LI, LE @ u[:,k] + b[:,k], u[:,k])
            
        return TimeSteppingSolution(self.t, u)

    def output(self, mu: Mu, u: TimeSteppingSolution | None = None):
        r"""
        Compute the output of interest :math:`s(\mu) = l(\mu) u(\mu)` at the given parameter value.
        
        Args:
            mu:
                Parameter for which to solve the time-stepping problem.
            u:
                Optional time-stepping solution. If ``None``, the state is computed via :meth:`solve`.
        
        Returns:
            Output of interest :math:`s(\mu) \in \mathbb{R}^p`.
        """
        if self.l is None:
            raise ValueError("No output functional defined for this model.")
        if u is None: u = self.solve(mu)
        u = u.u
        # s = np.zeros_like(self.t)
        # for k in range(self.K+1):
        #     s[k] = self.l(mu) @ u[:,k]
        # return TimeSteppingSolution(self.t, u, s)
        return self.l(mu) @ u[:,-1]

class DualStationaryTimeSteppingGalerkinFOM():
    r"""Model for dual time-stepping problems with a Galerkin operator and stationary operators.
        
    For details refer to `StationaryTimeSteppingGalerkinFOM`.
    """
    
    LI: ParametricGalerkinOperator[Mu, Matrix]
    r"""Implicit part of the time-stepping scheme, i.e. the operator :math:`\mathcal{L}^I(\mu):W\to W'` applied to the solution at the next time step."""
    LE: ParametricGalerkinOperator[Mu, Matrix]
    r"""Explicit part of the time-stepping scheme, i.e. the operator :math:`\mathcal{L}^E(\mu):W\to W'` applied to the solution at the current time step."""
    b: AffineLinear[Mu, Vector]
    """Inhomogeneity of the first time step."""
    t: np.ndarray[float]
    """Time points at which the solution is approximated."""
    W: InnerProduct[Mu]
    """Inner product on the space :math:`W`."""
    n: int
    """Dimension of the space :math:`W`."""
    K: int
    """Number of time intervals, i.e. ``K+1`` time points."""
    
    @property
    def n(self) -> int:
        return self.W.shape[0]
    
    @property
    def K(self) -> int:
        return len(self.t)-1
    
    
    def __init__(self, 
                 LI: ParametricGalerkinOperator[Mu, Matrix],
                 LE: ParametricGalerkinOperator[Mu, Matrix],
                 b: AffineLinear[Mu, Vector],
                 t: np.ndarray[float],
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None):
        r"""
        Args:
            LI:
                Implicit part of the time-stepping scheme, i.e. the operator applied to the solution at the next time step.
            LE:
                Explicit part of the time-stepping scheme, i.e. the operator applied to the solution at the current time step.
            b:
                Inhomogeneity at the first iteration.
            t:
                Time points ``t_0, ..., t_K`` at which the solution is approximated.
            solver:
                Solver for the linear system per time step. Defaults to a factorizing `DirectSolver`.
        """
        assert LI.shape == LE.shape
        assert LI.V is LE.V is LE.U is LI.U
        assert b.shape == (LI.shape[0],)
        
        self.LI = LI
        self.LE = LE
        self.b = wrap_affinelinear(b).compress()
        self.t = t
        self.W = LI.U
        self._solver = wrap_solver(solver if solver is not None else DirectSolver(factorize=True))
        
    def solve(self, mu: Mu) -> TimeSteppingSolution[Mu]:
        r"""Solve the time-stepping problem for a given parameter.
        
        Args:
            mu:
                Parameter for which to solve the time-stepping problem.
                
        Returns:
            Solution for the given parameter.
        """
        
        LI = self.LI.B(mu)
        LE = self.LE.B(mu)
        b = self.b(mu)
        
        u = np.zeros((self.n, self.K+1))
        u[:,self.K] = self._solver(LI, b)
        for k in range(self.K,0,-1):
            u[:,k-1] = self._solver(LI, LE @ u[:,k], u[:,k])
            
        return TimeSteppingSolution(self.t, u)

class PrimalDualStationaryTimeSteppingGalerkinFOM(StationaryTimeSteppingGalerkinFOM):
    r"""Model for primal-dual time-stepping problems with a Galerkin operator and stationary operators.
        
    For primal problem refer to `StationaryTimeSteppingGalerkinFOM`. For dual problem refer to `DualStationaryTimeSteppingGalerkinFOM`.
    """

    dual: DualStationaryTimeSteppingGalerkinFOM[Mu]
    r"""Dual problem."""

    def __init__(self, 
                 LI: ParametricGalerkinOperator[Mu, Matrix],
                 LE: ParametricGalerkinOperator[Mu, Matrix],
                 b: AffineLinear[Mu, Vector],
                 u0: AffineLinear[Mu, Vector],
                 t: np.ndarray[float],
                 solver: Solver | Callable[[Matrix, Vector, Vector|None], Vector] | None = None,
                 l: AffineLinear[Mu, Vector] | None = None):
        r"""
        Args:
            LI:
                Implicit part of the time-stepping scheme, i.e. the operator applied to the solution at the next time step.
            LE:
                Explicit part of the time-stepping scheme, i.e. the operator applied to the solution at the current time step.
            b:
                Inhomogeneity at each iteration, i.e. ``b(mu)`` has shape ``(n, K)``, with ``b(mu)[:,k]`` being :math:`b_k(\mu)`.
            u0:
                Initial value at ``t_0``.
            t:
                Time points ``t_0, ..., t_K`` at which the solution is approximated.
            solver:
                List of two solvers for the primal and dual model, respectively. Alternatively, a single solver which is used for both models. Defaults to a factorizing `DirectSolver` for each model.
            l:
                Affine decomposition of the output(s) of interest functional :math:`l(\mu) = \sum_{q=1}^{Q_l} \theta_q^l(\mu) l_q`. Required, as it defines the right-hand side of the dual problem.
        """
        assert l is not None
        if solver is None:
            solver = [None, None]
        try:
            solver = list(solver)
        except TypeError:
            solver = [solver, solver]
            
        if len(solver) != 2:
            raise ValueError("Solver must be a list of two solvers for primal and dual.")
        
        super().__init__(LI, LE, b, u0, t, solver[0], l)
        
        LI_T = ParametricGalerkinOperator(LI.B.T, LI.U, stability=LI._stability_solver, continuity=LI._continuity_solver)
        LE_T = ParametricGalerkinOperator(LE.B.T, LE.U, stability=LE._stability_solver, continuity=LE._continuity_solver)
        self.dual = DualStationaryTimeSteppingGalerkinFOM(LI_T, LE_T, l, t, solver[1])    

def theta_scheme(A: Matrix | AffineLinear[Mu, Matrix],
                 M: Matrix | AffineLinear[Mu, Matrix],
                 f: AffineFunction[Mu],
                 I: tuple[float, float],
                 K: int,
                 theta: float) -> tuple[AffineLinear[Mu, Matrix], AffineLinear[Mu, Matrix], AffineLinear[Mu, Vector], np.ndarray[float]]:
    r"""Constructs time-stepping components for the :math:`\vartheta`-scheme.
    
    For two Galerkin operators :math:`M(\mu), A(\mu) : W \to W'`, a right-hand side :math:`f(\mu) \in C(I;W')`, a initial value :math:`u_0(\mu) \in W` and a time-interval :math:`I = (t_0, t_K)`, consider the linear time-invariant initial value problem of finding :math:`u(\mu) \in C^1(I;W)` such that
    
    .. math::
        \begin{aligned}
        u(t_0; \mu) &= u_0(\mu),\\
        M(\mu) u'(t; \mu) + A(\mu) u(t; \mu) &= f(t; \mu)\quad\text{in $W'$} &&\text{for } t \in (t_0, t_K].
        \end{aligned}
        
    Using :math:`t_k = t_0 + k \cdot \Delta t` for a constant step-size :math:`k=0,\ldots,K` with :math:`\Delta t = (t_K - t_0) / K` and using the notation of `StationaryTimeSteppingGalerkinFOM`, the :math:`\vartheta`-scheme with :math:`\vartheta \in [0,1]` is given by
    
    .. math::
        \mathcal{L}^I(\mu) = M(\mu) + \vartheta \Delta t A(\mu), \quad \mathcal{L}^E(\mu) = M(\mu) - (1-\vartheta) \Delta t A(\mu), \quad b_k(\mu) = \Delta t \big( (1-\vartheta) f(t_k; \mu) + \vartheta f(t_{k+1}; \mu) \big).
    
    Special cases are the explicit Euler (:math:`\vartheta=0`, `explicit_euler`), the implicit Euler (:math:`\vartheta=1`, `implicit_euler`) and the Crank-Nicolson scheme (:math:`\vartheta=1/2`, `crank_nicolson`).
    
    Args:
        A:
            The stiffness matrix.
        M:
            The mass matrix.
        f:
            The affine-linear right-hand side.
        I:
            The time interval.
        K:
            The number of discrete time intervals.
        theta:
            The weight :math:`\vartheta \in [0,1]` of the implicit part.

    Returns
    -------
    LI:
        The implicit part of the time-stepping scheme, i.e. the operator applied to the solution at the next time step.
    LE:
        The explicit part of the time-stepping scheme, i.e. the operator applied to the
        solution at the current time step.
    b:
        The inhomogeneity of the time-stepping scheme at each time step.
    t:
        The time points at which the solution is computed.
    """
    if not 0 <= theta <= 1:
        raise ValueError("theta must be in [0, 1].")
    
    t = np.linspace(*I, K+1)
    dt = (I[1] - I[0]) / K
    
    # only add the A terms with nonzero weight, to keep the affine decompositions short
    LI = wrap_affinelinear(M)
    LE = wrap_affinelinear(M)
    if theta != 0: LI = LI + theta * dt * wrap_affinelinear(A)
    if theta != 1: LE = LE - (1-theta) * dt * wrap_affinelinear(A)
    
    def assemble_b(f):
        b = np.zeros((A.shape[1], K))
        for k in range(K):
            if theta != 1: b[:,k] += (1-theta) * dt * np.asarray(f(t[k])).reshape(-1)
            if theta != 0: b[:,k] +=    theta  * dt * np.asarray(f(t[k+1])).reshape(-1)
        return b
    
    b = AffineLinear(f.apply2data(assemble_b))
    return LI, LE, b, t


def explicit_euler(A: Matrix | AffineLinear[Mu, Matrix],
                   M: Matrix | AffineLinear[Mu, Matrix],
                   f: AffineFunction[Mu],
                   I: tuple[float, float],
                   K: int) -> tuple[AffineLinear[Mu, Matrix], AffineLinear[Mu, Matrix], AffineLinear[Mu, Vector], np.ndarray[float]]:
    r"""Constructs time-stepping components for the explicit Euler scheme, i.e. `theta_scheme` with :math:`\vartheta=0`:
    
    .. math::
        \mathcal{L}^I(\mu) = M(\mu), \quad \mathcal{L}^E(\mu) = M(\mu) - \Delta t A(\mu), \quad b_k(\mu) = \Delta t f(t_k; \mu).
    
    For the arguments and return values refer to `theta_scheme`.
    """
    return theta_scheme(A, M, f, I, K, theta=0.0)


def implicit_euler(A: Matrix | AffineLinear[Mu, Matrix],
                   M: Matrix | AffineLinear[Mu, Matrix],
                   f: AffineFunction[Mu],
                   I: tuple[float, float],
                   K: int) -> tuple[AffineLinear[Mu, Matrix], AffineLinear[Mu, Matrix], AffineLinear[Mu, Vector], np.ndarray[float]]:
    r"""Constructs time-stepping components for the implicit Euler scheme, i.e. `theta_scheme` with :math:`\vartheta=1`:
    
    .. math::
        \mathcal{L}^I(\mu) = M(\mu) + \Delta t A(\mu), \quad \mathcal{L}^E(\mu) = M(\mu), \quad b_k(\mu) = \Delta t f(t_{k+1}; \mu).
    
    For the arguments and return values refer to `theta_scheme`.
    """
    return theta_scheme(A, M, f, I, K, theta=1.0)
    
    
def crank_nicolson(A: Matrix | AffineLinear[Mu, Matrix],
                   M: Matrix | AffineLinear[Mu, Matrix],
                   f: AffineFunction[Mu],
                   I: tuple[float, float],
                   K: int) -> tuple[AffineLinear[Mu, Matrix], AffineLinear[Mu, Matrix], AffineLinear[Mu, Vector], np.ndarray[float]]:
    r"""Constructs time-stepping components for the Crank-Nicolson scheme, i.e. `theta_scheme` with :math:`\vartheta=1/2`:
    
    .. math::
        \mathcal{L}^I(\mu) = M(\mu) + \frac{\Delta t}{2} A(\mu), \quad \mathcal{L}^E(\mu) = M(\mu) - \frac{\Delta t}{2} A(\mu), \quad b_k(\mu) = \frac{\Delta t}{2} ( f(t_k; \mu) + f(t_{k+1}; \mu) ).
    
    For the arguments and return values refer to `theta_scheme`.
    """
    return theta_scheme(A, M, f, I, K, theta=0.5)
