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


def func2(mu, x, idx=None):
    f1 = lambda mu, x: 1/(np.sqrt((x[0]+mu[0])**2 + (x[1]+mu[1])**2))
    f2 = lambda mu, x: np.sin(2*np.pi*mu[0]*x[0]) * x[1]**mu[1]
    
    if idx is None:
        return [[f1(mu, xx), f2(mu, xx)] for xx in x]
    if idx == 0:
        return [f1(mu, xx) for xx in x]
    if idx == 1:
        return [f2(mu, xx) for xx in x]
    raise ValueError("idx must be None, 0 or 1.")



Omega = [[0,0], [1,1]]
P     = [[1e-2,1e-2], [1,1]]

Nmax = 100
tol  = 1e-6

Omega_train = np.meshgrid(np.linspace(Omega[0][0], Omega[1][0], 40), np.linspace(Omega[0][1], Omega[1][1], 40))
P_train     = np.meshgrid(np.linspace(P[0][0], P[1][0], 40), np.linspace(P[0][1], P[1][1], 40))
Omega_train = np.vstack([Omega_train[0].flatten(), Omega_train[1].flatten()]).T
P_train     = np.vstack([P_train[0].flatten(), P_train[1].flatten()]).T

func2_eim = empirical_interpolation(func2, P_train, Omega_train, Nmax=Nmax, tol=tol)
func21_eim = empirical_interpolation(lambda mu,x: func2(mu,x,0), P_train, Omega_train, Nmax=Nmax, tol=tol)
func22_eim = empirical_interpolation(lambda mu,x: func2(mu,x,1), P_train, Omega_train, Nmax=Nmax, tol=tol)