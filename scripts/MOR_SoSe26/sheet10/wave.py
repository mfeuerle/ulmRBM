import numpy as np
import pyvista as pv

from ulmRBM.affine import AffineFunction
from ulmRBM.solver import DirectSolver
from ulmRBM.fenicsx import utils, norms
from ulmRBM.fenicsx.problems import simple_wave
from ulmRBM.fom import FOM

K = 50
nx = [50]

# f = AffineFunction([1.0], [lambda tx: np.ones(tx.shape[1])])
# g = AffineFunction([1.0], [lambda tx: tx[0]*tx[1]])
# u0 = AffineFunction([1.0], [lambda tx: np.sin(np.pi*tx[1])])

B, f, U_fnx, V_fnx = simple_wave(K, nx)
U = norms.h10(U_fnx, solver=DirectSolver(factorize=True))
V = norms.h10(V_fnx, solver=DirectSolver(factorize=True))
fom = FOM(B, f, U, V, stability='direct', continuity='direct', solver=DirectSolver())

def solve_fom(mu):
    u = fom.solve(mu)
    return U_fnx.set_dirichletbcs(mu, u)

mu = 1
u = solve_fom(mu)

plotter = pv.Plotter()
utils.plot_pyvista(u, U_fnx.space, f'solution mu = {mu:.2f}', plotter)
plotter.show()

print()