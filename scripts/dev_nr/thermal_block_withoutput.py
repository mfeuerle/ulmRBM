import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import time

from ulmRBM.core import NO_MU
from ulmRBM.fenicsx import utils, norms
from ulmRBM.fenicsx.problems import thermal_block_with_output
from ulmRBM.solver import DirectSolver, IterativeSolver
from ulmRBM.fom import GalerkinFOM, FOM
from ulmRBM.rom import *
from ulmRBM.reductors import greedy_rbm

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

B, f, U, V, l, s0 = thermal_block_with_output(n_dofs, nblocks)

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
                    continuity=continuity_analytic,
                    l=l)
else:
    fom = FOM(B, f, U_H10, U_H10, solver=solver_B, 
            stability =coercivity_analytic, 
            continuity=continuity_analytic,
            l=l)
    
    
#########################
# ROM: CONFIG
#########################

mu_train = np.asarray([np.random.uniform(*mu_range, size=nblocks) for _ in range(N_train)])

coercivity = ThetaStability(fom)
continuity = ThetaContinuity(fom)

if galerkin:
    rom = GalerkinROM(fom, stability=ExactStability(fom), continuity=ExactContinuity(fom))
else:
    rom = ROM(fom, stability=coercivity, continuity=continuity)


#########################
# ROM: BUILD
#########################
start_time = time.time()

# greedy_constant_estimator(coercivity, mu_train)
# greedy_constant_estimator(continuity, mu_train)

err_decay = greedy_rbm(rom, mu_train, Nmax, strong=strong, ortho=ortho, use_output=True)[1]

time_buildin_rom = time.time() - start_time
print(f"Time for building ROM: {time_buildin_rom:.2f}s")


print(f"Full-order dimension: {fom.shape}.")
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
    u_fom[i] = fom.solve(mu_test[i])
    s_fom[i] = fom.output(mu_test[i]) + s0(mu_test[i])
fom_time = time.time() - start_time
print(f"\nAverage time for FOM solve: {fom_time / N_test:.2e}s")

start_time = time.time()
for i in range(N_test):
    u_rom[i] = rom.solve(mu_test[i])
    s_rom[i] = rom.output(mu_test[i]) + s0(mu_test[i])
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

print(f"\nMax output error of ROM: {np.max(err_out):.2e}")
print(f"Average output error of ROM: {np.mean(err_out):.2e}")

print(f"\nMax overestimation of output error: {np.max(err_out_bound / err_out):.1f}")
print(f"Average overestimation of output error: {np.mean(err_out_bound / err_out):.1f}")

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
    
# mus = [np.random.uniform(*mu_range, size=nblocks) for i in range(3)]

# for mu in mus:
#     plotter = pv.Plotter(shape=(1, 2), title=f"mu = {mu}")
#     plotter.subplot(0,0)
#     utils.plot_pyvista(U.set_dirichletbcs(mu, fom.solve(mu)), U.space, f"FOM solution", plotter)
#     plotter.subplot(0,1)
#     utils.plot_pyvista(U.set_dirichletbcs(mu, rom.reconstruct(mu)), U.space, f"ROM solution", plotter)
#     plotter.show(interactive_update=True)

# rows = 4    
# cols = np.ceil(rom.shape[1]/4).astype(int)
# plotter = pv.Plotter(shape=(rows,cols) , title=f"reduced basis")
# for i in range(rom.shape[1]):
#     plotter.subplot(np.floor(i/cols).astype(int),np.mod(i,cols,dtype=int))
#     utils.plot_pyvista(U.set_dirichletbcs(NO_MU, rom.U_basis[:,i]), U.space, f"xi_{i}", plotter)
# plotter.show(interactive_update=True)
    
# start = 7
# stop = len(err_decay)-1
# C, b = np.polyfit(range(start,stop), np.log(err_decay[start:stop]), 1)
# y_exp = np.exp(b) * np.exp(C * np.arange(start,stop))

# plt.figure('error decay')
# plt.semilogy(err_decay, label="greedy error decay")
# plt.semilogy(range(start,stop), y_exp, "--", label=fr"$e^{{{C:.3f}x}}$")
# plt.xlabel('greedy iteration')
# plt.ylabel('max. error over training set')
# plt.legend()

plt.show()

print("\nDone.")