import numpy as np
from scipy.sparse.linalg import spsolve
import pyvista as pv

from ulmRBM.fenicsx import utils
from ulmRBM.fenicsx.problems import thermal_block


nblocks = [2, 3]

B, f, U, V = thermal_block([20, 20], nblocks, plot=True)

def solve(mu):
    u = spsolve(B(mu), f(mu))
    return U.set_dirichletbcs(mu, u)

print(f"System matrix shape: {B.shape}")
print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

########################################
# plott results

# parameter value at which to solve the problem
mus = [
    np.ones(nblocks), 
    np.logspace(0,-1.2,np.prod(nblocks)).reshape(nblocks),
    0.05 + (np.indices(nblocks).sum(axis=0) % 2) * (2 - 0.05)
]  

plotter = pv.Plotter(shape=(1,len(mus)))
for i, mu in enumerate(mus):
    plotter.subplot(0, i)
    utils.plot_pyvista(solve(mu), U.space, f"solution (mu={mu})", plotter)
plotter.show()