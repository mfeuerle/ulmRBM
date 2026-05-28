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
from ulmRBM.reductors import greedy_rbm

nblocks = [2, 2]
mu_range = (0.05, 2.0)
n_dofs = [350, 350] # total dofs: n_dofs[0]*n_dofs[1]
solver_V = DirectSolver()
solver_B = DirectSolver()

galerkin = True
strong = False
ortho = True
N_train = 1000
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
    rom = GalerkinROM(fom, stability=coercivity, continuity=continuity)
else:
    rom = ROM(fom, stability=coercivity, continuity=continuity)


#########################
# ROM: BUILD
#########################
start_time = time.time()

greedy_constant_estimator(coercivity, mu_train)
greedy_constant_estimator(continuity, mu_train)

err_decay = greedy_rbm(rom, mu_train, Nmax, strong=strong, ortho=ortho)[1]

time_buildin_rom = time.time() - start_time
print(f"Time for building ROM: {time_buildin_rom:.2f}s")


print(f"Full-order dimension: {fom.dim}.")
print(f"Reduced-order dimension: {rom.dim}.")


#########################
# TIME AND ERROR
#########################

mu_test = np.asarray([np.random.uniform(*mu_range, size=nblocks) for _ in range(N_test)])

u_fom = np.empty(N_test, dtype=object)
u_rom = np.empty(N_test, dtype=object)
err_exact = np.empty(N_test)
err_bound = np.empty(N_test)

start_time = time.time()
for i in range(N_test):
    u_fom[i] = fom.solve(mu_test[i])
fom_time = time.time() - start_time
print(f"\nAverage time for FOM solve: {fom_time / N_test:.2e}s")

start_time = time.time()
for i in range(N_test):
    u_rom[i] = rom.solve(mu_test[i])
rom_time = time.time() - start_time
print(f"Average time for ROM solve: {rom_time / N_test:.2e}s")
print(f"Speedup: {fom_time / rom_time:.1f}x")

start_time = time.time()
for i in range(N_test):
    err_exact[i] = rom.error(mu_test[i], u_fom=u_fom[i])
error_exact_time = time.time() - start_time

start_time = time.time()
for i in range(N_test):
    err_bound[i] = rom.error_bound(mu_test[i])
error_bound_time = time.time() - start_time   
print(f"Average time for ROM solve + error bound: {(rom_time+error_bound_time) / N_test:.2e}s")
print(f"Speedup: {fom_time / (rom_time+error_bound_time):.1f}x")

print(f"\nMax error of ROM: {np.max(err_exact):.2e}")
print(f"Average error of ROM: {np.mean(err_exact):.2e}")

print(f"\nMax overestimation of error: {np.max(err_bound) / np.max(err_exact):.1f}")
print(f"Average overestimation of error: {np.mean(err_bound) / np.mean(err_exact):.1f}")

########################################
# PLOT SOLUTION
########################################
    
mus = [np.random.uniform(*mu_range, size=nblocks) for i in range(3)]

for mu in mus:
    plotter = pv.Plotter(shape=(1, 2), title=f"mu = {mu}")
    plotter.subplot(0,0)
    utils.plot_pyvista(U.set_dirichletbcs(mu, fom.solve(mu)), U.space, f"FOM solution", plotter)
    plotter.subplot(0,1)
    utils.plot_pyvista(U.set_dirichletbcs(mu, rom.reconstruct(mu)), U.space, f"ROM solution", plotter)
    plotter.show(interactive_update=True)

rows = 4    
cols = np.ceil(rom.dim[1]/4).astype(int)
plotter = pv.Plotter(shape=(rows,cols) , title=f"reduced basis")
for i in range(rom.dim[1]):
    plotter.subplot(np.floor(i/cols).astype(int),np.mod(i,cols,dtype=int))
    utils.plot_pyvista(U.set_dirichletbcs(NO_MU, rom.U_basis[:,i]), U.space, f"xi_{i}", plotter)
plotter.show(interactive_update=True)
    
start = 7
stop = len(err_decay)-1
C, b = np.polyfit(range(start,stop), np.log(err_decay[start:stop]), 1)
y_exp = np.exp(b) * np.exp(C * np.arange(start,stop))

plt.figure('error decay')
plt.semilogy(err_decay, label="greedy error decay")
plt.semilogy(range(start,stop), y_exp, "--", label=fr"$e^{{{C:.3f}x}}$")
plt.xlabel('greedy iteration')
plt.ylabel('max. error over training set')
plt.legend()

plt.show()

print("\nDone.")