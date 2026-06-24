import numpy as np
import matplotlib.pyplot as plt

from ulmRBM.fenicsx import norms, SpaceTimeKey
from ulmRBM.fenicsx.problems import *
from ulmRBM.solver import DirectSolver
from ulmRBM.fom import *
from ulmRBM.rom import *
from ulmRBM.reductors import greedy_rbm, pod_greedy_rbm

def poisson(K, nx):
    n = [K] + nx
    B, f, U_fnx, _ = simple_elliptic(n)
    U = norms.h10(U_fnx, solver=DirectSolver(factorize=True))
    fom = GalerkinFOM(B, f, U, stability='direct', continuity='direct', solver=DirectSolver())
    rom = GalerkinROM(fom, stability=ExactStability(fom), continuity=ExactContinuity(fom), solver=DirectSolver())
    return fom, rom

def heat(K, nx):
    SPACE = SpaceTimeKey.SPACE
    TIME  = SpaceTimeKey.TIME
    B, f, U_fnx, V_fnx = simple_heat(K, nx)
    U = norms.space_time(U_fnx, 
                        [{TIME: 'l2',  SPACE: 'h10'}, 
                         {TIME: 'h10', SPACE: 'h10 dual'}], 
                        solver=DirectSolver(factorize=True))
    V = norms.space_time(V_fnx, 
                        [{TIME: 'l2',  SPACE: 'h10'}], 
                        solver=DirectSolver(factorize=True))
    fom = FOM(B, f, U, V, stability='direct', continuity='direct', solver=DirectSolver())
    rom = ROM(fom, stability=ExactStability(fom), continuity=ExactContinuity(fom), solver=DirectSolver())
    return fom, rom

def heat_timestepping(K, nx):
    LI, LE, b, u0, t, W_fnx = simple_heat_timestepping(K, nx)
    W = norms.h10(W_fnx, DirectSolver(factorize=True))
    LI = ParametricGalerkinOperator(LI, W, stability='direct', continuity='direct')
    LE = ParametricGalerkinOperator(LE, W, stability='direct', continuity='direct')
    fom = StationaryTimeSteppingGalerkinFOM(LI, LE, b, u0, t)
    rom = StationaryTimeSteppingGalerkinROM(fom, LI_stability=ExactStability(LI), LE_continuity=ExactContinuity(LE))
    return fom, rom

def wave(K, nx):
    B, f, U_fnx, V_fnx = simple_wave(K, nx)
    U = norms.h10(U_fnx, solver=DirectSolver(factorize=True))
    V = norms.h10(V_fnx, solver=DirectSolver(factorize=True))
    fom = FOM(B, f, U, V, stability='direct', continuity='direct', solver=DirectSolver())
    rom = ROM(fom, stability=ExactStability(fom), continuity=ExactContinuity(fom), solver=DirectSolver())
    return fom, rom

K = 50
nx = [50]

mu_range = [1e-2, 1]
n_train = 500
N_max = 40
tol = 1e-5

mu_train = np.linspace(mu_range[0], mu_range[1], n_train)

PROBLEMS = ['poisson', 'heat', 'heat_timestepping', 'wave']

problems = {PROBLEM: eval(PROBLEM)(K, nx) for PROBLEM in PROBLEMS}
fom = {PROBLEM: problems[PROBLEM][0] for PROBLEM in PROBLEMS}
rom = {PROBLEM: problems[PROBLEM][1] for PROBLEM in PROBLEMS}

# cranck-nicolson coincides with space time: use faster solver
fom['heat'].solve = lambda mu: fom['heat_timestepping'].solve(mu).u[:,1:].flatten()  

greedy = {PROBLEM: greedy_rbm for PROBLEM in PROBLEMS}
greedy['heat_timestepping'] = pod_greedy_rbm

err_decay = {PROBLEM: greedy[PROBLEM](rom[PROBLEM], mu_train, N_max, tol, strong=True)[1] for PROBLEM in PROBLEMS}

N = {PROBLEM: np.arange(len(err_decay[PROBLEM])) + 1 for PROBLEM in PROBLEMS}
online_effort = {PROBLEM: N[PROBLEM]**3 for PROBLEM in PROBLEMS}
online_effort['heat_timestepping'] = K * online_effort['heat_timestepping']

plt.figure()
plt.subplot(1, 2, 1)
for PROBLEM in PROBLEMS:
    plt.semilogy(N[PROBLEM], err_decay[PROBLEM], '--*', label=PROBLEM)
plt.xlabel('N')
plt.ylabel('max error')
plt.legend()

plt.subplot(1, 2, 2)
for PROBLEM in PROBLEMS:
    plt.semilogy(online_effort[PROBLEM], err_decay[PROBLEM], '--*', label=PROBLEM)
plt.xlabel('online effort')
plt.ylabel('max error')
plt.legend()



plt.show()
print()




