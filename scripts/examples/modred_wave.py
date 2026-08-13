import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from ulmRBM.fenicsx import utils, norms, SpaceTimeKey
from ulmRBM.fenicsx.problems import simple_wave_hilbert
from ulmRBM.solver import DirectSolver
from ulmRBM.products import OperatorInnerProduct
from ulmRBM.fom import GalerkinFOM, FOM
from ulmRBM.rom import *
from ulmRBM.reductors import greedy_rbm

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME

eigenvalues = 'direct'
B_solver = lambda: DirectSolver()
U_solver = lambda: DirectSolver(factorize=True)
V_solver = lambda: DirectSolver(factorize=True)

########################################
# Model Parameters
########################################

mu_range = np.array([1e-4, 1e2])
n_train = 500
N_max = 100

strong = True

nx = 64
K  = 64

f  = 1.0   # right-hand side
u0 = 0.0   # initial condition u(0)
u1 = 0.0   # initial velocity u_t(0)
g  = 0.0   # boundary condition on IxGamma

########################################
# Full-Order Model (based on Hilbert transform)
########################################
    
B, f, U_fnx, V_fnx = simple_wave_hilbert(K, nx, f, g, u0, u1)

print(f"fom dimension: {B.shape[0]} x {B.shape[1]}")

U_H10 = norms.space_time(U_fnx, 
                    [{TIME: 'l2',  SPACE: 'h10'},
                     {TIME: 'h10', SPACE: 'l2'}],
                    solver=U_solver())
V_H10 = norms.space_time(V_fnx, 
                    [{TIME: 'l2',  SPACE: 'h10'},
                     {TIME: 'h10', SPACE: 'l2'}],
                    solver=V_solver())
U_BiV = OperatorInnerProduct(B, V_H10.dual, solver=B_solver())

fom_H10 = FOM(B, f, U_H10, V_H10, stability=eigenvalues, continuity=eigenvalues, solver=B_solver())
fom_BiV = FOM(B, f, U_BiV, V_H10, stability=1.0, continuity=1.0, solver=B_solver())

########################################
# Reduced-Order Model
########################################
stability_H10  = ExactStability(fom_H10)
continuity_H10 = ExactContinuity(fom_H10)

rom_H10 = ROM(fom_H10, stability_H10, continuity_H10)
rom_BiV = ROM(fom_BiV, ExactStability(fom_BiV), ExactContinuity(fom_BiV))


########################################
# Build reduced models
########################################

mu_train = np.linspace(*mu_range, n_train)

greedy_constant_estimator(stability_H10,  mu_train, N_max)
greedy_constant_estimator(continuity_H10, mu_train, N_max)

err_H10, mus_H10 = greedy_rbm(rom_H10, mu_train, N_max, 1e-99, strong)[1:]
err_BiV, mus_BiV = greedy_rbm(rom_BiV, mu_train, N_max, 1e-99, False)[1:]     # strong and weak coincide


########################################
# Plotting
########################################

plt.figure()
plt.loglog(err_H10, label='H10')
plt.loglog(err_BiV, label='BiV')
plt.title('Error decay')
plt.xlabel('reduced dimension $N$')
plt.ylabel('max. error over training set')
plt.legend()

plt.figure()
plt.plot(mus_H10, np.ones(len(mus_H10)), 'o', label='H10')
plt.plot(mus_BiV, np.ones(len(mus_BiV)), 'x', label='BiV')
plt.title('Selected parameters')
plt.legend()

plt.show()