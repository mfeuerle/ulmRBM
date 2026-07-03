import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time

from ulmRBM.core import NO_MU
from ulmRBM.fenicsx import utils, norms
from ulmRBM.fenicsx.problems import thermal_block_with_output
from ulmRBM.solver import DirectSolver, IterativeSolver
from ulmRBM.fom import GalerkinFOM, FOM, PrimalDualGalerkinFOM
from ulmRBM.rom import *
from ulmRBM.reductors import primaldual_greedy_rbm

nblocks = [2, 2]
mu_range = (0.05, 2.0)
n_dofs = [150, 150] # total dofs: n_dofs[0]*n_dofs[1]
solver_V = DirectSolver()
solver_B = DirectSolver()

galerkin = True
strong = False
ortho = True
N_train = 100
N_test = 50
Nmax = 50

#########################
# FULL-ORDER MODEL
#########################

B, f, U, V, l, s0 = thermal_block_with_output(n_dofs, nblocks, output_mode=1)

print(f"System matrix shape: {B.shape}")
print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

U_H10 = norms.h10(U, solver_V)
coercivity_analytic = lambda mu, *args: np.min(mu)
continuity_analytic = lambda mu, *args: np.max(mu)

# fom = GalerkinFOM(B, f, U_H10, solver=solver_B, 
#                 stability =coercivity_analytic, 
#                 continuity=continuity_analytic,
#                 l=l)

fom_pd = PrimalDualGalerkinFOM(B, f, U_H10, l,
                               solver=[solver_B, solver_B],
                               stability =coercivity_analytic, 
                               continuity=continuity_analytic,)
    
#########################
# ROM: CONFIG
#########################

mu_train = np.asarray([np.random.uniform(*mu_range, size=nblocks) for _ in range(N_train)])

rom = PrimalDualGalerkinROM(fom_pd, stability=ExactStability(fom_pd), continuity=ExactContinuity(fom_pd))

#########################
# ROM: BUILD
#########################

start_time = time.time()

err_decay = primaldual_greedy_rbm(rom, mu_train, Nmax, strong=strong, ortho=ortho, mu_select_mode=1)[1]

time_buildin_rom = time.time() - start_time
print(f"Time for building ROM: {time_buildin_rom:.2f}s")


print(f"Full-order dimension: {fom_pd.shape}.")
print(f"Reduced-order dimension: {rom.shape}.")

#########################
# TIME AND ERROR
#########################

mu_test = np.asarray([np.random.uniform(*mu_range, size=nblocks) for _ in range(N_test)])

u_fom = np.empty(N_test, dtype=object)
u_rom = np.empty(N_test, dtype=object)

s_fom = np.empty(N_test)
s_rom = np.empty(N_test)

err_exact = np.empty(N_test)
err_bound = np.empty(N_test)

err_out = np.empty(N_test)
err_out_bound = np.empty(N_test)

start_time = time.time()
for i in range(N_test):
    u_fom[i] = fom_pd.solve(mu_test[i])
    s_fom[i] = fom_pd.output(mu_test[i])
fom_time = time.time() - start_time
print(f"\nAverage time for FOM solve: {fom_time / N_test:.2e}s")

start_time = time.time()
for i in range(N_test):
    u_rom[i] = rom.solve(mu_test[i])
    s_rom[i] = rom.output(mu_test[i])
rom_time = time.time() - start_time
print(f"Average time for ROM solve: {rom_time / N_test:.2e}s")
print(f"Speedup: {fom_time / rom_time:.1f}x")

start_time = time.time()
for i in range(N_test):
    err_exact[i] = rom.error(mu_test[i], u_fom=u_fom[i])
error_exact_time = time.time() - start_time

err_out = np.abs(s_fom - s_rom)

start_time = time.time()
for i in range(N_test):
    err_bound[i] = rom.error_bound(mu_test[i])
    err_out_bound[i] = rom.output_error_bound(mu_test[i])
error_bound_time = time.time() - start_time   
print(f"Average time for ROM solve + error bound: {(rom_time+error_bound_time) / N_test:.2e}s")
print(f"Speedup: {fom_time / (rom_time+error_bound_time):.1f}x")

print(f"\nMax error of ROM: {np.max(err_exact):.2e}")
print(f"Average error of ROM: {np.mean(err_exact):.2e}")

print(f"\nMax overestimation of error: {np.max(err_bound / err_exact):.1f}")
print(f"Average overestimation of error: {np.mean(err_bound / err_exact):.1f}")

########################################
# PLOT SOLUTION
########################################

fig = plt.figure()
fig.suptitle("Output")
plt.plot(range(N_test), s_fom,label='FOM',linestyle='',marker='o')
plt.plot(range(N_test), s_rom,label='ROM',linestyle='',marker='x')
plt.xlabel('mu')
plt.legend()

fig = plt.figure()
fig.suptitle("Error")
plt.semilogy(range(N_test), err_exact,linestyle=':',marker='o',label='exact')
plt.semilogy(range(N_test), err_bound,linestyle=':',marker='o',label='bound')
plt.semilogy(range(N_test), err_out,linestyle=':',marker='o',label='output exact')
plt.semilogy(range(N_test), err_out_bound,linestyle=':',marker='o',label='output bound')
plt.xlabel('mu')
plt.legend()

plt.show()

print("\nDone.")