import numpy as np
from scipy.sparse.linalg import spsolve
import pyvista as pv
import time
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

from mpi4py import MPI

from dolfinx import mesh, fem
import ufl

from ulmRBM.affine import empirical_interpolation, AffineFunction, AffineObject, AffineLinear
from ulmRBM.fenicsx import utils, interpolate_function_eim
from ulmRBM.fenicsx.problems import thermal_block, assemble_system, assemble_vector

np.set_printoptions(edgeitems=30, linewidth=100000, precision=2)



def func1(mu, x):
    return np.sin(mu*x)

Omega = [0, 2]
P     = [0, 2*np.pi]

Omega_train = np.linspace(Omega[0], Omega[1], 20)
P_train     = np.linspace(P[0], P[1], 20)

Omega_test = np.linspace(Omega[0], Omega[1], 100)
P_test     = np.linspace(P[0], P[1], 100)


print("EIM discrete:")
func1_eim_d = empirical_interpolation(func1, P_train, Omega_train, continuous=False)
FUNC1_d     = np.asarray([func1(mu, Omega_train) for mu in P_test])
FUNC1_eim_d = np.asarray([func1_eim_d(mu) for mu in P_test])
ERR_d       = abs(FUNC1_d-FUNC1_eim_d)
print(f"Error on Omega_train x P_test:\t\t\t\t{np.max(ERR_d):.2e}\n")


print("EIM continuous:")
func1_eim_c = empirical_interpolation(func1, P_train, Omega_train, continuous=True, residual=True)
FUNC1_c     = np.asarray([func1(mu, Omega_test) for mu in P_test])
FUNC1_eim_c = np.asarray([func1_eim_c(mu)(Omega_test) for mu in P_test])
ERR_c       = abs(FUNC1_c-FUNC1_eim_c)
print(f"Error on Omega_test x P_test:\t\t\t\t{np.max(ERR_c):.2e}")


XX_d, PP_d = np.meshgrid(Omega_train, P_test)
XX_c, PP_c = np.meshgrid(Omega_test,  P_test)

fig = plt.figure("discrete")

ax = fig.add_subplot(1,3,1, projection='3d')
surf = ax.plot_surface(XX_d, PP_d, FUNC1_d, cmap='viridis')
ax.set_title('func1(mu, x)')
ax.set_xlabel('x')
ax.set_ylabel('mu')

ax = fig.add_subplot(1,3,2, projection='3d')
surf = ax.plot_surface(XX_d, PP_d, FUNC1_eim_d, cmap='viridis')
ax.set_title('func1_eim(mu, x)')
ax.set_xlabel('x')
ax.set_ylabel('mu')

ax = fig.add_subplot(1,3,3, projection='3d')
surf = ax.plot_surface(XX_d, PP_d, ERR_d, cmap='viridis')
ax.set_title('|func1(mu, x) - func1_eim(mu, x)|')
ax.set_xlabel('x')
ax.set_ylabel('mu')

fig = plt.figure("discrete basis functions")

for i, (theta_q, f_q) in enumerate(func1_eim_d):
    ax = fig.add_subplot(2,int(np.ceil(len(func1_eim_d)/2)),i+1)
    surf = ax.plot(Omega_train, np.asarray(f_q))
    ax.set_title(F"g_{i}(x)")
    ax.set_xlabel('x')



fig = plt.figure("continuous")

ax = fig.add_subplot(1,3,1, projection='3d')
surf = ax.plot_surface(XX_c, PP_c, FUNC1_c, cmap='viridis')
ax.set_title('func1(mu, x)')
ax.set_xlabel('x')
ax.set_ylabel('mu')

ax = fig.add_subplot(1,3,2, projection='3d')
surf = ax.plot_surface(XX_c, PP_c, FUNC1_eim_c, cmap='viridis')
ax.set_title('func1_eim(mu, x)')
ax.set_xlabel('x')
ax.set_ylabel('mu')

ax = fig.add_subplot(1,3,3, projection='3d')
surf = ax.plot_surface(XX_c, PP_c, ERR_c, cmap='viridis')
ax.set_title('|func1(mu, x) - func1_eim(mu, x)|')
ax.set_xlabel('x')
ax.set_ylabel('mu')


fig = plt.figure("continous basis functions")

for i, (theta_q, f_q) in enumerate(func1_eim_c):
    ax = fig.add_subplot(2,int(np.ceil(len(func1_eim_c)/2)),i+1)
    surf = ax.plot(Omega_test, np.asarray(f_q(Omega_test)))
    ax.set_title(F"g_{i}(x)")
    ax.set_xlabel('x')


plt.tight_layout()
plt.show()
