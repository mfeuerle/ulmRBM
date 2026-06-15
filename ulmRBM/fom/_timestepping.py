"""
Time-stepping (full-order) models for initial value problems.
"""

from __future__ import annotations


__all__ = [
    'TimeSteppingSolution',
    'StationaryTimeSteppingGalerkinModel',
    'explicit_euler',
    'implicit_euler',
    'crank_nicolson',
]

from collections.abc import Callable
from typing import Generic
from dataclasses import dataclass

import numpy as np

from ulmRBM.core import (
    Mu, Matrix, Vector, ParametricLinear
)
from ulmRBM.solver import Solver, DirectSolver, wrap_solver
from ulmRBM.affine import AffineLinear, AffineFunction, wrap_affinelinear
from ulmRBM.products import InnerProduct, MatrixInnerProduct

from ._basic import FOM, GalerkinFOM
from ._operators import ParametricGalerkinOperator, ParametricOperator

    
@dataclass
class TimeSteppingSolution(Generic[Mu]):
    r"""Solution of a time-stepping problem for a given parameter.
    """
    mu: Mu
    r"""Parameter for which the solution is computed."""
    t: np.ndarray[float]
    r"""Time points at which the solution is computed."""
    u: np.ndarray[float]
    r"""Solution values at the time points in `t`, i.e. ``u[:,k]`` is the solution at time ``t[k]``."""
    
    def __iter__(self) -> zip[float, np.ndarray[float]]:
        r"""Iterate over the time points and corresponding solution values."""
        return zip(self.t, self.u.T)
    
    
class StationaryTimeSteppingGalerkinModel(Generic[Mu]):
    r"""Model for time-stepping problems with a Galerkin operator and stationary operators.
    
    For some time-interval :math:`I = [t_0, t_K]`, a Galerkin operator :math:`A(\mu) : W \to W'`, a right-hand side :math:`f(\mu) \in C(I;W')` and a initial value :math:`u_0(\mu) \in W`, consider the problem of finding :math:`u(\mu) \in C^1(I;W)` such that
    
    .. math::
        \begin{aligned}
        u(t_0; \mu) &= u_0(\mu),\\
        u'(t; \mu) - A(\mu) u(t; \mu) &= f(t; \mu)\quad\text{in $W'$} &&\text{for } t \in (t_0, t_K].
        \end{aligned}        
    
    For a given sequence of time points :math:`t_0, ..., t_K` in :math:`I`, the model represents the time-stepping problem of finding :math:`u_k(\mu) \approx u(t_k; \mu)` for :math:`k=0,\ldots,K` such that
    
    .. math::
        \begin{aligned}
        u_0(\mu) &= u_0(\mu),\\
        \mathcal{L}^I(\mu) u_{k+1}(\mu) &= \mathcal{L}^E(\mu) u_k(\mu) + b_k(\mu)\quad\text{in $W'$} &&\text{for } k=0,\ldots,K-1.
        \end{aligned}
        
    where :math:`\mathcal{L}^I(\mu), \mathcal{L}^E(\mu) : W \to W'` are the implicit and explicit part of the time-stepping scheme, respectively, and :math:`b_k(\mu) \in W'` is the inhomogeneity at time :math:`t_k`. This model is called "stationary", since the operators :math:`\mathcal{L}^I(\mu)` and :math:`\mathcal{L}^E(\mu)` are independent of the time step :math:`k`, i.e. they only depend on the parameter :math:`\mu` but not on the time points :math:`t_k` and are thus stationary in time. Typically, this is the case if :math:`A(\mu)` is stationary and the time-steps are equidistant.
    """
    
    LI: ParametricGalerkinOperator[Mu, Matrix]
    """Implicit part of the time-stepping scheme, i.e. the operator applied to the solution at the next time step."""
    LE: ParametricGalerkinOperator[Mu, Matrix]
    """Explicit part of the time-stepping scheme, i.e. the operator applied to the solution at the current time step."""
    b: np.ndarray[AffineLinear[Mu, Vector]]
    """Inhomogeneity of the time-stepping scheme at each time step."""
    u0: AffineLinear[Mu, Vector]
    """Initial value."""
    t: np.ndarray[float]
    """Time points at which the solution is computed."""
    W: InnerProduct[Mu]
    """Inner product on the space :math:`W`."""
    n: int
    """Dimension of the space :math:`W`."""
    K: int
    """Number of time intervals, i.e. ``K+1`` time points."""
    
    
    @property
    def n(self) -> int:
        return self.LI.shape[0]
    
    @property
    def K(self) -> int:
        return len(self.t)-1
    
    
    def __init__(self, 
                 LI: ParametricGalerkinOperator[Mu, Matrix],
                 LE: ParametricGalerkinOperator[Mu, Matrix],
                 b: AffineLinear[Mu, Vector],
                 t: np.ndarray[float],
                 u0: AffineLinear[Mu, Vector],
                 solver = DirectSolver(factorize=True)):
        r"""
        Args:
            LI:
                Implicit part of the time-stepping scheme, i.e. the operator applied to the solution at the next time step.
            LE:
                Explicit part of the time-stepping scheme, i.e. the operator applied to the solution at the current time step.
            b:
                Inhomogeneity at each iteration, i.e. ``b(mu)`` has shape ``(n, K)``, with ``b(mu)[:,k]`` being :math:`b_k(\mu)`.
            t:
                Time points ``t_0, ..., t_K`` at which the solution is computed.
            u0:
                Initial value at ``t_0``.
        """
        assert LI.shape == LE.shape
        assert LI.V is LE.V is LE.U is LI.U
        assert b.shape == (LI.shape[0], len(t)-1)
        
        self.LI = LI
        self.LE = LE
        self.b = b
        self.u0 = u0
        self.t = t
        self.W = LI.U
        self._solver = wrap_solver(solver)
        
    def solve(self, mu: Mu) -> TimeSteppingSolution[Mu]:
        r"""Solve the time-stepping problem for a given parameter.
        
        Args:
            mu:
                Parameter for which to solve the time-stepping problem.
                
        Returns:
            Solution for the given parameter."""
        
        LI = self.LI.B(mu)
        LE = self.LE.B(mu)
        b = self.b(mu)
        
        u = np.zeros((self.n, self.K+1))
        u[:,0] = self.u0(mu)
        for k in range(self.K):
            u[:,k+1] = self._solver(LI, LE @ u[:,k] + b[:,k], u[:,k])
            
        return TimeSteppingSolution(mu, self.t, u)
    
    
    def _space_time_matrix(self) -> AffineLinear[Mu, Matrix]:
        raise NotImplementedError("The space-time matrix is not implemented yet.")
    
    def _space_time_rhs(self) -> AffineLinear[Mu, Vector]:
        # homogenization needed for initial value, must be moved to right-hand side
        raise NotImplementedError("The space-time right-hand side is not implemented yet.")
    
    def convert_to_space_time(self, U, 
                            V= None,
                            l: AffineLinear[Mu, Vector] | Vector | None = None,
                            stability:  Callable[[Mu, ParametricOperator[Mu]], float] | float | str = None,
                            continuity: Callable[[Mu, ParametricOperator[Mu]], float] | float | str = None,
                            supremizer: Callable[[Vector, ParametricOperator[Mu]], ParametricLinear[Mu, Vector] | AffineLinear[Mu, Vector]] = None) -> FOM:
        r"""Convert the time-stepping model to a space-time model with trial space :math:`U` and test space :math:`V`.
        
        Args:
            U:
                Trial space for the space-time model.
            V:
                Test space for the space-time model.
                
        Returns:
            The corresponding space-time model.
        """
        B = self._space_time_matrix()
        f = self._space_time_rhs()
        
        if V is None:
            fom = GalerkinFOM(B, f, U, l, stability=stability, continuity=continuity, supremizer=supremizer)
        else:
            fom = FOM(B, f, U, V, l, stability=stability, continuity=continuity, supremizer=supremizer)
            
        fom._solver = None
        fom.solve = lambda mu: self.solve(mu).u[:,1:].T.flatten()
        
        return fom

def explicit_euler(A: AffineLinear[Mu, Matrix],
                 f: AffineFunction[Mu],
                 I: tuple[float, float],
                 K: int,
                 H: Matrix | AffineLinear[Mu, Matrix] | MatrixInnerProduct[Mu]) -> tuple[ParametricGalerkinOperator[Mu, Matrix], ParametricGalerkinOperator[Mu, Matrix], AffineLinear[Mu, Vector], np.ndarray[float]]:
    r"""Constructs time-stepping components for the explicit Euler scheme.
    
    Let :math:`(W,H,W')` be a Gelfand triple and :math:`A(\mu) : W \to W'` an affine-linear Galerkin operator, :math:`f(\mu) \in C(I;W')` an affine-linear right-hand side and :math:`u_0(\mu) \in W` an affine-linear initial value for a time interval :math:`I = [t_0, t_K]` and a time steps :math:`t_k = t_0 + k \cdot \Delta t` for :math:`k=0,\ldots,K` with :math:`\Delta t = (t_K - t_0) / K`.
    
    Using the notation of `EquidistantTimeSteppingGalerkinModel`, the explicit Euler scheme is given by
    
    .. math::
        \mathcal{L}^I(\mu) = H, \quad \mathcal{L}^E(\mu) = H + \Delta t A(\mu), \quad b_k(\mu) = \Delta t f(t_k; \mu).
        
    where :math:`H` is the matrix representation of the inner product on :math:`H`.
    
    Args:
        A:
            The affine-linear Galerkin operator.
        f:
            The affine-linear right-hand side.
        I:
            The time interval.
        K:
            The number of discrete time intervals.
        H:
            The inner product on :math:`H`.

    Returns:
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
    t = np.linspace(*I, K+1)
    dt = (I[1] - I[0]) / K
    
    if isinstance(H, MatrixInnerProduct):
        H = H._M
    
    LI = wrap_affinelinear(H)
    LE = wrap_affinelinear(H) + dt * wrap_affinelinear(A)
    
    def assemble_b(f):
        b = np.empty(A.shape[1], K)
        for k in range(K):
            b[:,k] = dt * np.asarray(f(t[k])).reshape(-1)
        return b
    
    b = AffineLinear(f.apply2data(assemble_b))
    return LI, LE, b, t

def implicit_euler(A: AffineLinear[Mu, Matrix],
                 f: AffineFunction[Mu],
                 I: tuple[float, float],
                 K: int,
                 H: Matrix | AffineLinear[Mu, Matrix] | MatrixInnerProduct) -> tuple[AffineLinear[Mu, Matrix], AffineLinear[Mu, Matrix], AffineLinear[Mu, Vector], np.ndarray[float]]:
    r"""Constructs time-stepping components for the implicit Euler scheme.
    
    Let :math:`(W,H,W')` be a Gelfand triple and :math:`A(\mu) : W \to W'` an affine-linear Galerkin operator, :math:`f(\mu) \in C(I;W')` an affine-linear right-hand side and :math:`u_0(\mu) \in W` an affine-linear initial value for a time interval :math:`I = [t_0, t_K]` and a time steps :math:`t_k = t_0 + k \cdot \Delta t` for :math:`k=0,\ldots,K` with :math:`\Delta t = (t_K - t_0) / K`.
    
    Using the notation of `EquidistantTimeSteppingGalerkinModel`, the implicit Euler scheme is given by 
    
    .. math::
        \mathcal{L}^I(\mu) = H - \Delta t A(\mu), \quad \mathcal{L}^E(\mu) = H, \quad b_k(\mu) = \Delta t f(t_{k+1}; \mu).
        
    where :math:`H` is the matrix representation of the inner product on :math:`H`.
    
    Args:
        A:
            The affine-linear Galerkin operator.
        f:
            The affine-linear right-hand side.
        I:
            The time interval.
        K:
            The number of discrete time intervals.
        H:
            The inner product on :math:`H`.

    Returns:
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
    t = np.linspace(*I, K+1)
    dt = (I[1] - I[0]) / K
    
    if isinstance(H, MatrixInnerProduct):
        H = H._M
    
    LI = wrap_affinelinear(H._M) - dt * wrap_affinelinear(A)
    LE = wrap_affinelinear(H._M)
    
    def assemble_b(f):
        b = np.empty(A.shape[1], K)
        for k in range(K):
            b[:,k] = dt * np.asarray(f(t[k+1])).reshape(-1)
        return b
    
    b = AffineLinear(f.apply2data(assemble_b))
    return LI, LE, b, t
    
    
def crank_nicolson(A: AffineLinear[Mu, Matrix],
                 f: AffineFunction[Mu],
                 I: tuple[float, float],
                 K: int,
                 H: Matrix | AffineLinear[Mu, Matrix] | MatrixInnerProduct) -> tuple[AffineLinear[Mu, Matrix], AffineLinear[Mu, Matrix], AffineLinear[Mu, Vector], np.ndarray[float]]:
    r"""Constructs time-stepping components for the Crank-Nicolson scheme.
    
    Let :math:`(W,H,W')` be a Gelfand triple and :math:`A(\mu) : W \to W'` an affine-linear Galerkin operator, :math:`f(\mu) \in C(I;W')` an affine-linear right-hand side and :math:`u_0(\mu) \in W` an affine-linear initial value for a time interval :math:`I = [t_0, t_K]` and a time steps :math:`t_k = t_0 + k \cdot \Delta t` for :math:`k=0,\ldots,K` with :math:`\Delta t = (t_K - t_0) / K`.
    
    Using the notation of `EquidistantTimeSteppingGalerkinModel`, the Crank-Nicolson scheme is given by
    
    .. math::
        \mathcal{L}^I(\mu) = H - \frac{\Delta t}{2} A(\mu), \quad \mathcal{L}^E(\mu) = H + \frac{\Delta t}{2} A(\mu), \quad b_k(\mu) = \frac{\Delta t}{2} ( f(t_k; \mu) + f(t_{k+1}; \mu) ).
        
    where :math:`H` is the matrix representation of the inner product on :math:`H`.
    
    Args:
        A:
            The affine-linear Galerkin operator.
        f:
            The affine-linear right-hand side.
        I:
            The time interval.
        K:
            The number of discrete time intervals.
        H:
            The inner product on :math:`H`.

    Returns:
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
    t = np.linspace(*I, K+1)
    dt = (I[1] - I[0]) / K
    
    if isinstance(H, MatrixInnerProduct):
        H = H._M
    
    LI = wrap_affinelinear(H._M) - dt/2 * wrap_affinelinear(A)
    LE = wrap_affinelinear(H._M) + dt/2 * wrap_affinelinear(A)
    
    def assemble_b(f):
        b = np.empty(A.shape[1], K)
        for k in range(K):
            b[:,k] = dt/2 * np.asarray(f(t[k]) + f(t[k+1])).reshape(-1)
        return b
    
    b = AffineLinear(f.apply2data(assemble_b))
    return LI, LE, b, t