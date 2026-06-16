import numpy as np
import matplotlib.pyplot as plt
import time

from ulmRBM.affine import AffineFunction, AffineObject
from ulmRBM.fenicsx import norms
from ulmRBM.fenicsx.problems import simple_timestepping_heat
from ulmRBM.solver import DirectSolver
from ulmRBM.fom import *
from ulmRBM.rom import *
from ulmRBM.reductors import pod_greedy_rbm

Omega = [0, 1]
nx = 500

# Omega = [[0,0], [1,1]]
# nx = [50,50]

I = [0,1]
K = 2000

mu_range = (0.01, 1.0)

strong = True
ortho = True
N_train = 200
N_test = 20
Nmax = 15

#########################
# FULL-ORDER MODEL
#########################

f = AffineFunction([1.0], [lambda t: lambda x: np.ones(x.shape[1])])
u0 = AffineObject([1.0], [lambda x:  np.sin(np.pi*x[0])])

A, M, f, u0, U = simple_timestepping_heat(Omega, nx, f, u0)
U_H10 = norms.h10(U, DirectSolver(factorize=True))

LI, LE, b, t = crank_nicolson(A, M, f, I, K)

LI = ParametricGalerkinOperator(LI, U_H10, stability='direct', continuity='direct')
LE = ParametricGalerkinOperator(LE, U_H10, stability='direct', continuity='direct')

fom = StationaryTimeSteppingGalerkinFOM(LI, LE, b, u0, t)

print(f"FOM dimension: {fom.n}")
print(f"Number of time steps: {fom.K+1}")
print(f"Number of affine terms in LI: {len(LI.B)}")
print(f"Number of affine terms in LE: {len(LE.B)}")
print(f"Number of affine terms in b: {len(b)}")

def solve_fom(mu):
    u = fom.solve(mu)
    u_full = np.zeros((U.dim, len(u.t)))
    for k in range(len(u.t)):
        u_full[:,k] = U.set_dirichletbcs(mu, u.u[:,k])
    return TimeSteppingSolution(mu, u.t, u_full)


#########################
# ROM: CONFIG
#########################

mu_train = np.asarray([np.random.uniform(*mu_range) for _ in range(N_train)])

LI_coercivity = ThetaStability(LI)
LE_continuity = ThetaContinuity(LE)

rom = StationaryTimeSteppingGalerkinROM(fom, LI_stability=LI_coercivity, LE_continuity=LE_continuity)#

def solve_rom(mu):
    u = rom.reconstruct(mu)
    u_full = np.zeros((U.dim, len(u.t)))
    for k in range(len(u.t)):
        u_full[:,k] = U.set_dirichletbcs(mu, u.u[:,k])
    return TimeSteppingSolution(mu, u.t, u_full)

#########################
# ROM: BUILD
#########################
start_time = time.time()

greedy_constant_estimator(LI_coercivity, mu_train)
greedy_constant_estimator(LE_continuity, mu_train)

err_decay = pod_greedy_rbm(rom, mu_train, Nmax, strong=strong, ortho=ortho)[1]

time_buildin_rom = time.time() - start_time
print(f"Time for building ROM: {time_buildin_rom:.2f}s")


print(f"Full-order dimension: {fom.n}")
print(f"Reduced-order dimension: {rom.n}")


#########################
# TIME AND ERROR
#########################

mu_test = np.asarray([np.random.uniform(*mu_range) for _ in range(N_test)])

u_fom = np.empty(N_test, dtype=object)
u_rom = np.empty(N_test, dtype=object)
err_exact = np.empty((N_test, K+1))
err_bound = np.empty((N_test, K+1))

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

print(f"\nMax overestimation of error: {np.max(err_bound / err_exact):.2e}")
print(f"Average overestimation of error: {np.mean(err_bound / err_exact):.2e}")

########################################
# PLOT SOLUTION
########################################

if np.isscalar(nx) or len(nx) == 1:
    mus = [np.random.uniform(*mu_range) for i in range(3)]

    T, X = np.meshgrid(t,U.space.mesh.geometry.x[:,0])

    for mu in mus:
        u_fom = solve_fom(mu)
        u_rom = solve_rom(mu)

        fig = plt.figure(fr'solution $\mu = {mu:.2f}$')
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        ax.plot_surface(T, X, u_fom.u)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u(t,x)$')
        
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot_surface(T, X, u_rom.u)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u(t,x)$')
        
        ax = fig.add_subplot(1, 3, 3)
        ax.semilogy(t, rom.error(mu), "*-")
        tmp = rom.error_bound(mu)
        idx = tmp < 1e100
        ax.semilogy(t[idx], tmp[idx], "*--")
        ax.legend(['exact error', 'error bound'])
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$\|u_{fom}(t) - u_{rom}(t)\|$')
        
        plt.show(block=False)
    
start = 3
stop = len(err_decay)-1
C, b = np.polyfit(range(start,stop), np.log(err_decay[start:stop]), 1)
y_exp = np.exp(b) * np.exp(C * np.arange(start,stop))

plt.figure('error decay')
plt.semilogy(err_decay, label="POD-greedy error decay")
plt.semilogy(range(start,stop), y_exp, "--", label=fr"$e^{{{C:.3f}x}}$")
plt.xlabel('POD-greedy iteration')
plt.ylabel('max. error over training set')
plt.legend()

plt.show()

print("\nDone.")