import numpy as np
import matplotlib.pyplot as plt
import time
import pathlib

from ulmRBM.fenicsx import norms, SpaceTimeKey
from ulmRBM.fenicsx.problems import simple_wave_hilbert
from ulmRBM.solver import DirectSolver, IterativeSolver
from ulmRBM.products import OperatorInnerProduct
from ulmRBM.fom import FOM
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

mu_range = [1e-4, 1e2]
n_train = 1000
N_max = 200

strong = True

nx = 256
K  = 256

f  = 1.0   # right-hand side
u0 = 0.0   # initial condition u(0)
u1 = 0.0   # initial velocity u_t(0)
g  = 0.0   # boundary condition on IxGamma

########################################
# Full-Order Model (based on Hilbert transform)
########################################
    
B, f, U_fnx, V_fnx = simple_wave_hilbert(K, nx, f, g, u0, u1)

print(f"fom dimension: {B.shape[0]} x {B.shape[1]}\n")

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

mu_train = np.linspace(mu_range[0], mu_range[1], n_train)

start_stability = time.time()
greedy_constant_estimator(stability_H10,  mu_train, N_max)
start_continuity = time.time()
greedy_constant_estimator(continuity_H10, mu_train, N_max)
start_rom_H10 = time.time()
err_H10, mus_H10 = greedy_rbm(rom_H10, mu_train, N_max, 1e-99, strong)[1:]
start_rom_BiV = time.time()
err_BiV, mus_BiV = greedy_rbm(rom_BiV, mu_train, N_max, 1e-99, False)[1:]     # strong and weak coincide
end = time.time()

########################################
# Save Results
########################################

header = (
    "Wave Reduction\n"
    f"K = {K}\n"
    f"nx = {nx}\n"
    f"mu_range = [{mu_range[0]}, {mu_range[1]}]\n"
    f"n_train = {n_train}\n"
    f"N_max = {N_max}\n"
    f"strong = {strong}\n"
    f"stability_H10 = {stability_H10}\n"
    f"continuity_H10 = {continuity_H10}\n"
    f"times: stability = {start_continuity - start_stability:.2f}s, continuity = {start_rom_H10 - start_continuity:.2f}s, rom_H10 = {start_rom_BiV - start_rom_H10:.2f}s, rom_BiV = {end - start_rom_BiV:.2f}s\n"
    "\n"
    "err_H10, err_BiV, mus_H10, mus_BiV"
)

np.savetxt(
    str(pathlib.Path(__file__).parent.resolve() / "plot_modered_wave_data.txt"),
    np.column_stack((err_H10, err_BiV, mu_train[mus_H10], mu_train[mus_BiV])),
    header=header
)

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
plt.hist([mu_train[mus_H10], mu_train[mus_BiV]], bins=20, label=["selected H10", "selected BiV"])
plt.legend()

plt.show()