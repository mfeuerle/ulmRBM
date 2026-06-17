import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from ulmRBM.affine import AffineObject
from ulmRBM.fenicsx import utils, norms
from ulmRBM.fenicsx.problems import simple_wave, assemble_system
from ulmRBM.solver import DirectSolver, IterativeSolver
from ulmRBM.products import OperatorInnerProduct
from ulmRBM.fom import FOM
from ulmRBM import rom
from ulmRBM.reductors import greedy_algorithm

########################################
# FULL-ORDER MODEL
########################################

solver = DirectSolver()

I = [0.0, 2.0]
Omega = [0.0, 1.0]

mu_range = (1e-4, 1e2)

nx = 10
nt = np.ceil(mu_range[1]*(I[1]-I[0])/((Omega[1]-Omega[0])/(nx+1))).astype('int')  # ensure CFL
print(f"\nnt = {nt}, nx = {nx}, mu_max = {mu_range[1]}")

use_exact = False    # wheter to calculate data from an exact solution
if use_exact:
    exact_sol = lambda x: np.sin(np.pi*x[0])*x[1]  # exact solution at mu=1.0
    B, f, U, V = simple_wave(I, Omega, nt, [nx], exact_sol=exact_sol, exact_mu=1.0)
else:
    f =  AffineObject([1.0], [1.0])   # right-hand side
    u0 = AffineObject([0.0], [1.0])   # initial condition u(0)
    u1 = AffineObject([0.0], [1.0])   # initial velocity u_t(0)
    g = AffineObject([0.0], [1.0])    # boundary condition on IxGamma
    B, f, U, V = simple_wave(I, Omega, nt, [nx], f, u0, u1, g)
    
B, f = assemble_system(B, f, U, V)

print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

V_H10 = norms.h10(V, solver)
U_BiV = OperatorInnerProduct(B, V_H10.dual, solver)
U_H10 = norms.h10(U, solver)

fom_BiV = FOM(B, f, U_BiV, V_H10, stability=1.0, continuity=1.0, solver=solver)
fom_H10 = FOM(B, f, U_H10, V_H10, solver=solver)

print(f"\nFOM dimension: {fom_H10.dim}")

########################################
# REDUCED-ORDER MODEL
########################################

rom_BiV = rom.Trial2TestROM(fom_BiV, stability=rom.StabilityExact())
rom_H10 = rom.Trial2TestROM(fom_H10, stability=rom.StabilityMinTheta())

########################################
# MODEL REDUCTION
########################################

mu_train = np.linspace(mu_range[0], mu_range[1], 1000).tolist()
Nmax = 20
tol = 1e-99

print("\nGreedy algorithm for BiV inner product:")
_, err_BiV, idx_BiV = greedy_algorithm(rom_BiV, mu_train, Nmax, tol)

print("\nGreedy algorithm for H10 inner product:")
_, err_H10, idx_H10 = greedy_algorithm(rom_H10, mu_train, Nmax, tol)


########################################
# PLOT SOLUTION
########################################
    
mus = [np.random.uniform(*mu_range) for i in range(3)]

for mu in mus:
    plotter = pv.Plotter(shape=(1, 3), title=f"mu = {mu:.2e}")
    plotter.subplot(0,0)
    utils.plot_pyvista(U.set_dirichletbcs(mu, fom_H10.solve(mu)), U.space, f"FOM solution (mu={mu:.2f})", plotter)
    plotter.subplot(0,1)
    utils.plot_pyvista(U.set_dirichletbcs(mu, rom_H10.reconstruct(mu)), U.space, f"ROM H10 solution (mu={mu:.2f})", plotter)
    plotter.subplot(0,2)
    utils.plot_pyvista(U.set_dirichletbcs(mu, rom_BiV.reconstruct(mu)), U.space, f"ROM BiV solution (mu={mu:.2f})", plotter)
    plotter.show(interactive_update=True)
    
if use_exact:
    mu = 1.0
    plotter = pv.Plotter(shape=(2, 2))
    plotter.subplot(0,0)
    utils.plot_pyvista(utils.interpolate_function(U.space, exact_sol).x.array, U.space, f"exact solution (mu={1:.2f})", plotter)
    plotter.subplot(0,1)
    utils.plot_pyvista(U.set_dirichletbcs(mu, fom_H10.solve(mu)), U.space, f"FOM solution (mu={mu:.2f})", plotter)
    plotter.subplot(1,0)
    utils.plot_pyvista(U.set_dirichletbcs(mu, rom_H10.reconstruct(mu)), U.space, f"ROM H10 solution (mu={mu:.2f})", plotter)
    plotter.subplot(1,1)
    utils.plot_pyvista(U.set_dirichletbcs(mu, rom_BiV.reconstruct(mu)), U.space, f"ROM BiV solution (mu={mu:.2f})", plotter)
    plotter.show(interactive_update=True)
    
    
plt.figure()
plt.loglog(err_BiV, label='BiV')
plt.loglog(err_H10, label='H10')
plt.xlabel('greedy iteration')
plt.ylabel('max. error over training set')
plt.legend()

plt.figure()
idx = np.zeros(len(mu_train), dtype=bool)
idx[idx_BiV] = True
plt.plot(idx, 'o', label='BiV')
idx = np.zeros(len(mu_train), dtype=bool)
idx[idx_H10] = True
plt.plot(idx, '+', label='H10')
plt.xlabel('greedy iteration')
plt.ylabel('max. error over training set')
plt.legend()

plt.show()

