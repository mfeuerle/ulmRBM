import numpy as np
import matplotlib.pyplot as plt

from ulmRBM.affine import AffineFunction, AffineObject
from ulmRBM.fenicsx import norms
from ulmRBM.fenicsx.problems import simple_heat_timestepping
from ulmRBM.solver import DirectSolver
from ulmRBM.fom import *
from ulmRBM.rom import *

Omega = [0, 1]
nx = 500

# Omega = [[0,0], [1,1]]
# nx = [50,50]

I = [0,1]
K = 200

mu_range = (0.01, 1.0)

#########################
# FULL-ORDER MODEL
#########################

f = AffineFunction([1.0], [lambda t,x: np.ones(x.shape[1])])
u0 = AffineObject([1.0], [lambda x:  np.sin(np.pi*x[0])])

LI, LE, b, u0, t, W_fnx = simple_heat_timestepping(K, nx, f, u0)
W = norms.h10(W_fnx, DirectSolver(factorize=True))
LI = ParametricGalerkinOperator(LI, W, stability='direct', continuity='direct')
LE = ParametricGalerkinOperator(LE, W, stability='direct', continuity='direct')

fom = StationaryTimeSteppingGalerkinFOM(LI, LE, b, u0, t)

print(f"FOM dimension: {fom.n}")
print(f"Number of time steps: {fom.K+1}")
print(f"Number of affine terms in LI: {len(LI.B)}")
print(f"Number of affine terms in LE: {len(LE.B)}")
print(f"Number of affine terms in b: {len(b)}")

def solve_fom(mu):
    u = fom.solve(mu)
    u_full = np.zeros((W_fnx.dim, len(u.t)))
    for k in range(len(u.t)):
        u_full[:,k] = W_fnx.set_dirichletbcs(mu, u.u[:,k])
    return TimeSteppingSolution(u.t, u_full)

########################################
# PLOT SOLUTION
########################################

if np.isscalar(nx) or len(nx) == 1:
    mus = [mu_range[0], (mu_range[1]-mu_range[0])/2 + mu_range[0], mu_range[1]]

    T, X = np.meshgrid(t,W_fnx.space.mesh.geometry.x[:,0])

    for mu in mus:
        u_fom = solve_fom(mu)

        fig = plt.figure(f'solution mu = {mu:.2f}')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(T, X, u_fom.u)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u(t,x)$')
        
        plt.show(block=False)

plt.show()
print("\nDone.")