import numpy as np
import matplotlib.pyplot as plt

from ulmRBM.affine import AffineFunction
from ulmRBM.fenicsx import norms
from ulmRBM.fenicsx.problems import simple_heat_timestepping
from ulmRBM.solver import DirectSolver
from ulmRBM.fom import *
from ulmRBM.rom import *


K = 50
nx = [50]

# f = AffineFunction([1.0], [lambda t: lambda x: np.ones(x.shape[1])])
# u0 = AffineFunction([1.0], [lambda x:  np.sin(np.pi*x[0])])

A, M, f, u0, W_fnx = simple_heat_timestepping(nx)
W = norms.h10(W_fnx, DirectSolver(factorize=True))

LI, LE, b, t = crank_nicolson(A, M, f, [0,1], K)

LI = ParametricGalerkinOperator(LI, W, stability='direct', continuity='direct')
LE = ParametricGalerkinOperator(LE, W, stability='direct', continuity='direct')

fom = StationaryTimeSteppingGalerkinFOM(LI, LE, b, u0, t)

def solve_fom(mu):
    u = fom.solve(mu)
    u_full = np.zeros((W_fnx.dim, len(u.t)))
    for k in range(len(u.t)):
        u_full[:,k] = W_fnx.set_dirichletbcs(mu, u.u[:,k])
    return u_full

mu = 1
u = solve_fom(mu)

if np.isscalar(nx) or len(nx) == 1:
    T, X = np.meshgrid(t,W_fnx.space.mesh.geometry.x[:,0])

    fig = plt.figure(f'solution mu = {mu:.2f}')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(T, X, u)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(t,x)$')

    plt.show()
    
print()