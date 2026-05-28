import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time

from ulmRBM.core import NO_MU
from ulmRBM.fenicsx import utils, norms
from ulmRBM.fenicsx.problems import thermal_block, assemble_system
from ulmRBM.solver import DirectSolver, IterativeSolver
from ulmRBM.fom import GalerkinFOM, FOM
from ulmRBM.rom import *
from ulmRBM.reductors import pod_rbm, greedy_rbm

nblocks = [2, 2]
mu_range = (0.05, 2.0)
n_dofs = [150, 150] # total dofs: n_dofs[0]*n_dofs[1]
solver_V = DirectSolver()
solver_B = DirectSolver()

galerkin = True
strong = False
ortho = True
N_train = 500
N_test = 50
Nmax = 50

#########################
# FULL-ORDER MODEL
#########################

B, f, U, V = thermal_block(n_dofs, nblocks)
B, f = assemble_system(B, f, U, V)

print(f"System matrix shape: {B.shape}")
print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

U_H10 = norms.h10(U, solver_V)
coercivity_analytic = lambda mu, *args: np.min(mu)
continuity_analytic = lambda mu, *args: np.max(mu)

if galerkin:
    fom = GalerkinFOM(B, f, U_H10, solver=solver_B, 
                    stability =coercivity_analytic, 
                    continuity=continuity_analytic)
else:
    fom = FOM(B, f, U_H10, U_H10, solver=solver_B, 
            stability =coercivity_analytic, 
            continuity=continuity_analytic)
    
    
#########################
# ROM: CONFIG
#########################

mu_train = np.asarray([np.random.uniform(*mu_range, size=nblocks) for _ in range(N_train)])

coercivity = ThetaStability(fom)
continuity = ThetaContinuity(fom)

if galerkin:
    rom_greedy = GalerkinROM(fom, stability=coercivity, continuity=continuity)
    rom_pod = GalerkinROM(fom, stability=coercivity, continuity=continuity)
else:
    rom_greedy = ROM(fom, stability=coercivity, continuity=continuity)
    rom_pod = ROM(fom, stability=coercivity, continuity=continuity)


#########################
# ROM: BUILD
#########################
start_time = time.time()
greedy_constant_estimator(coercivity, mu_train)
greedy_constant_estimator(continuity, mu_train)
time_constants = time.time() - start_time

start_time = time.time()
err_greedy = greedy_rbm(rom_greedy, mu_train, Nmax)[1]
time_greed_rom = time.time() - start_time + time_constants


start_time = time.time()
err_pod    = pod_rbm(rom_pod, mu_train, Nmax)[1]
time_pod_rom = time.time() - start_time + time_constants

print(f"\nTime for building Greedy-ROM: {time_greed_rom:.2f}s")
print(f"Time for building POD-ROM: {time_pod_rom:.2f}s")


print(f"\nFull-order dimension: {fom.dim}.")
print(f"Greedy-Rom dimension: {rom_greedy.dim}.")
print(f"POD-Rom dimension: {rom_pod.dim}.")


#########################
# TIME AND ERROR
#########################

mu_test = np.asarray([np.random.uniform(*mu_range, size=nblocks) for _ in range(N_test)])

err_greedy_exact = np.empty(N_test)
err_pod_exact = np.empty(N_test)
for i in range(N_test):
    u_fom = fom.solve(mu_test[i])
    err_greedy_exact[i] = rom_greedy.error(mu_test[i], u_fom=u_fom)
    err_pod_exact[i] = rom_pod.error(mu_test[i], u_fom=u_fom)

print(f"\nMax error of Greedy-ROM: {np.max(err_greedy_exact):.2e}")
print(f"Average error of Greedy-ROM: {np.mean(err_greedy_exact):.2e}")

print(f"\nMax error of POD-ROM: {np.max(err_pod_exact):.2e}")
print(f"Average error of POD-ROM: {np.mean(err_pod_exact):.2e}")

########################################
# PLOT SOLUTION
########################################

rows = 4    
cols = np.ceil(rom_greedy.dim[1]/4).astype(int)
plotter = pv.Plotter(shape=(rows,cols) , title=f"reduced basis greedy")
for i in range(rom_greedy.dim[1]):
    plotter.subplot(np.floor(i/cols).astype(int),np.mod(i,cols,dtype=int))
    utils.plot_pyvista(U.set_dirichletbcs(NO_MU, rom_greedy.U_basis[:,i]), U.space, f"xi_{i}", plotter)
plotter.show(interactive_update=True)

cols = np.ceil(rom_pod.dim[1]/4).astype(int)
plotter = pv.Plotter(shape=(rows,cols) , title=f"reduced basis pod")
for i in range(rom_pod.dim[1]):
    plotter.subplot(np.floor(i/cols).astype(int),np.mod(i,cols,dtype=int))
    utils.plot_pyvista(U.set_dirichletbcs(NO_MU, rom_pod.U_basis[:,i]), U.space, f"xi_{i}", plotter)
plotter.show(interactive_update=True)
    
start = 7
stop = len(err_greedy)-1
C_greedy, b_greedy = np.polyfit(range(start,stop), np.log(err_greedy[start:stop]), 1)
y_exp_greedy = np.exp(b_greedy) * np.exp(C_greedy * np.arange(start,stop))

start = 7
stop = len(err_pod)-1
C_pod, b_pod = np.polyfit(range(start,stop), np.log(err_pod[start:stop]), 1)
y_exp_pod = np.exp(b_pod) * np.exp(C_pod * np.arange(start,stop))

plt.figure('error decay')
plt.semilogy(err_greedy, label="greedy error decay")
plt.semilogy(err_pod, label="pod error decay")

start = 7
stop = len(err_greedy)-1
C_greedy, b_greedy = np.polyfit(range(start,stop), np.log(err_greedy[start:stop]), 1)
y_exp_greedy = np.exp(b_greedy) * np.exp(C_greedy * np.arange(start,stop))
plt.semilogy(range(start,stop), y_exp_greedy, "--", label=fr"$e^{{{C_greedy:.3f}x}}$")

start = 7
stop = len(err_pod)-1
C_pod, b_pod = np.polyfit(range(start,stop), np.log(err_pod[start:stop]), 1)
y_exp_pod = np.exp(b_pod) * np.exp(C_pod * np.arange(start,stop))
plt.semilogy(range(start,stop), y_exp_pod, "--", label=fr"$e^{{{C_pod:.3f}x}}$")

plt.xlabel('size of the reduced model')
plt.ylabel('max. error (greedy) / average error (pod) over training set')
plt.legend()

plt.show()

print("\nDone.")