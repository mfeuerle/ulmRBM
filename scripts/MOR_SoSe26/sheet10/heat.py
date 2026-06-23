import numpy as np
import matplotlib.pyplot as plt

from ulmRBM.affine import AffineFunction
from ulmRBM.solver import DirectSolver
from ulmRBM.fenicsx import norms, SpaceTimeKey
from ulmRBM.fenicsx.problems import simple_heat
from ulmRBM.fom import FOM

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME

K = 50
nx = [50]

# f = AffineFunction([1.0], [lambda t,x: np.ones(x.shape[1])])
# g = AffineFunction([1.0], [lambda t,x: t*x[0]])
# u0 = AffineFunction([1.0], [lambda x: np.sin(np.pi*x[0])])

B, f, U_fnx, V_fnx = simple_heat(K, nx)

U = norms.space_time(U_fnx, 
                     [{TIME: 'l2',  SPACE: 'h10'}, 
                      {TIME: 'h10', SPACE: 'h10 dual'}], 
                     solver=DirectSolver(factorize=True))

V = norms.space_time(V_fnx, 
                     [{TIME: 'l2',  SPACE: 'h10'}], 
                     solver=DirectSolver(factorize=True))

fom = FOM(B, f, U, V, stability='direct', continuity='direct', solver=DirectSolver())

def solve_fom(mu):
    u = fom.solve(mu)
    u = U_fnx.set_dirichletbcs(mu, u, True)
    return u

mu = 1
u = solve_fom(mu)

if np.isscalar(nx) or len(nx) == 1:
    T, X = np.meshgrid(U_fnx.space[TIME].mesh.geometry.x[:,0], U_fnx.space[SPACE].mesh.geometry.x[:,0])

    fig = plt.figure(f'solution mu = {mu:.2f}')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(T, X, u)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(t,x)$')
    
    plt.show()

print()