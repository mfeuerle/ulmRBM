import numpy as np
import pyvista as pv

from ulmRBM.affine import AffineFunction
from ulmRBM.solver import DirectSolver
from ulmRBM.fenicsx import utils, norms
from ulmRBM.fenicsx.problems import simple_elliptic
from ulmRBM.fom import GalerkinFOM

n = [50, 50]

# f = AffineFunction([1.0], [lambda x: np.ones(x.shape[1])])
# g = AffineFunction([1.0], [lambda x: x[0]*x[1]])

B, f, U_fnx, V_fnx = simple_elliptic(n)
U = norms.h10(U_fnx, solver=DirectSolver(factorize=True))
fom = GalerkinFOM(B, f, U, stability='direct', continuity='direct', solver=DirectSolver())

def solve_fom(mu):
    u = fom.solve(mu)
    return U_fnx.set_dirichletbcs(mu, u)

mu = 1
u = solve_fom(mu)

plotter = pv.Plotter()
utils.plot_pyvista(u, U_fnx.space, f'solution mu = {mu:.2f}', plotter)
plotter.show()

print()