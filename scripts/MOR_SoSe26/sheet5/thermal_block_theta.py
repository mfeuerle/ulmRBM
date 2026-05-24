import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from ulmRBM.fenicsx import  norms
from ulmRBM.fenicsx.problems import thermal_block, assemble_system
from ulmRBM.solver import DirectSolver
from ulmRBM.fom import FOM, GalerkinFOM
from ulmRBM import rom

nblocks = [2, 2]
mu_range = (0.05, 2.0)
solver = DirectSolver()

galerkin = True
N_train = 100
Nmax = 19
n_plots_per_row = 4

#########################
# FULL-ORDER MODEL
#########################

B, f, U, V = thermal_block([50, 50], nblocks)
B, f = assemble_system(B, f, U, V)

print(f"System matrix shape: {B.shape}")
print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

U_H10 = norms.h10(U, solver)
coercivity_analytic = lambda mu, *args: np.min(mu)
continuity_analytic = lambda mu, *args: np.max(mu)

if galerkin:
    fom = GalerkinFOM(B, f, U_H10, solver=solver, 
                    stability =coercivity_analytic, 
                    continuity=continuity_analytic)
else:
    fom = FOM(B, f, U_H10, U_H10, solver=solver, 
            stability =coercivity_analytic, 
            continuity=continuity_analytic)


#########################
# TRAINING PARAMETERS
#########################

mu_train = np.asarray([np.random.uniform(*mu_range, size=nblocks) for _ in range(N_train)])

coercivity_exact = np.asarray([fom.stability(mu) for mu in mu_train])
continuity_exact = np.asarray([fom.continuity(mu) for mu in mu_train])

idx = np.argsort(coercivity_exact)
mu_train_coercivity = mu_train[idx]
coercivity_exact = coercivity_exact[idx]

idx = np.argsort(continuity_exact)
mu_train_continuity = mu_train[idx]
continuity_exact = continuity_exact[idx]

#########################
# STABILITY ESTIMATE
#########################

coercivity_theta = rom.ThetaStability(fom)

coercivity_LB = []
coercivity_UB = []
def coercivity_callback(estimator, mu):
    coercivity_LB.append([estimator.lower_bound(mu) for mu in mu_train_coercivity])
    coercivity_UB.append([estimator.upper_bound(mu) for mu in mu_train_coercivity])
    
rom.greedy_constant_estimator(coercivity_theta, mu_train_coercivity, Nmax, callback=coercivity_callback)

coercivity_LB = np.asarray(coercivity_LB)
coercivity_UB = np.asarray(coercivity_UB)

#########################
# CONTINUITY ESTIMATE
#########################

continuity_theta = rom.ThetaContinuity(fom)

continuity_LB = []
continuity_UB = []
def continuity_callback(estimator, mu):
    continuity_LB.append([estimator.lower_bound(mu) for mu in mu_train_continuity])
    continuity_UB.append([estimator.upper_bound(mu) for mu in mu_train_continuity])

rom.greedy_constant_estimator(continuity_theta, mu_train_continuity, Nmax, callback=continuity_callback)

continuity_LB = np.asarray(continuity_LB)
continuity_UB = np.asarray(continuity_UB)

########################################
# PLOT SOLUTION
########################################

fig = plt.figure('stability')
rows = np.ceil((len(coercivity_LB)+1)/n_plots_per_row).astype('int')
for i in range(len(coercivity_LB)):
    ax = fig.add_subplot(rows, n_plots_per_row, i+1)
    ax.plot(coercivity_LB[i], 'r')
    ax.plot(coercivity_UB[i], 'g')
    ax.plot(coercivity_exact, 'k--')
    ax.set_title(f'iteration {i+1}')
    ax.set_xlabel('parameter index')
    ax.set_ylabel('stability constant')
ax = fig.add_subplot(rows, n_plots_per_row, i+2)
ax.plot([], 'r', label='lower bound')
ax.plot([], 'g', label='upper bound')
ax.plot([], 'k--', label='exact')
ax.legend()
    
fig = plt.figure('continuity')
rows = np.ceil((len(continuity_LB)+1)/n_plots_per_row).astype('int')
for i in range(len(continuity_LB)):
    ax = fig.add_subplot(rows, n_plots_per_row, i+1)
    ax.plot(continuity_LB[i], 'r')
    ax.plot(continuity_UB[i], 'g')
    ax.plot(continuity_exact, 'k--')
    ax.set_title(f'iteration {i+1}')
    ax.set_xlabel('parameter index')
    ax.set_ylabel('continuity constant')
ax = fig.add_subplot(rows, n_plots_per_row, i+2)
ax.plot([], 'r', label='lower bound')
ax.plot([], 'g', label='upper bound')
ax.plot([], 'k--', label='exact')
ax.legend()

plt.show()