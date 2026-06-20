import numpy as np
from scipy.sparse.linalg import spsolve
import pyvista as pv
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh

from ulmRBM.affine import AffineObject, AffineFunction
from ulmRBM.solver import DirectSolver
from ulmRBM.fenicsx import utils, SpaceTimeKey
from ulmRBM.fenicsx.problems import simple_heat

SPACE = SpaceTimeKey.SPACE
TIME = SpaceTimeKey.TIME

solver = DirectSolver()

I = [0.0, 1.0]
Omega = [0.0, 1.0]
K = 50
nx = 50
gdim = 1

# for x dependent diffusion, e.g.: AffineFunction([lambda mu: mu], [lambda x: -np.eye(gdim).reshape(-1,1) * np.ones(x.shape[1])])
A = AffineObject([lambda mu: mu], [-np.eye(gdim)])
b = AffineObject([0.0], [np.ones(gdim)])
c = AffineObject([0.0], [1.0])
A = (A, b, c)

f = AffineFunction([1.0], [lambda t,x: np.ones(x.shape[1])])
g = AffineFunction([1.0], [lambda t,x: t*x[0]])
u0 = AffineFunction([1.0], [lambda x: np.sin(np.pi*x[0])])

# f = AffineFunction([1.0], [lambda t,x: np.ones(x.shape[1])])
# g = AffineFunction([1.0], [lambda t,x: np.zeros(x.shape[1])])
# u0 = AffineFunction([1.0], [lambda x: np.sin(np.pi*x[0])])

B, f, U, V = simple_heat(I, Omega, K, nx, A, f, g, u0)

print(f"System matrix shape: {B.shape}")
print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")


mu = 1

u = solver(B(mu).assemble(), f(mu))
u = U.set_dirichletbcs(mu, u, True)

if np.isscalar(nx) or len(nx) == 1:
    T, X = np.meshgrid(U.space[TIME].mesh.geometry.x[:,0],U.space[SPACE].mesh.geometry.x[:,0])

    fig = plt.figure(f'solution mu = {mu:.2f}')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(T, X, u)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(t,x)$')
    
    plt.show()

print()