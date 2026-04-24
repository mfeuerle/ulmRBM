import numpy as np
from scipy.sparse.linalg import spsolve
import pyvista as pv

from mpi4py import MPI
from dolfinx import mesh

from ulmRBM.affine import AffineObject
from ulmRBM.fenicsx import utils
from ulmRBM.fenicsx.problems import weak_problem, assemble_system


Omega = np.array([[0.0, 0.0], [2.0, 2.0]])
msh = mesh.create_rectangle(MPI.COMM_WORLD, Omega, [20, 20])
gdim = msh.geometry.dim

dbdry = [lambda x: utils.isclose(x[0], Omega[:,0])]   # list of dirichlet boundary parts of the trial space U
nbdry = [lambda x: utils.isclose(x[1], Omega[:,1])]   # list of neumann boundary parts of the trial space U

# for x dependent diffusion, e.g.: lambda x: -np.eye(gdim).reshape(-1,1) * np.ones(x.shape[1])
A = AffineObject([lambda mu: mu], [-np.eye(gdim)])
b = AffineObject([1.0], [np.ones(gdim)])
c = AffineObject([1.0], [1.0])

use_exact = True    # wheter to calculate data from an exact solution
if use_exact:
    data = (lambda x: np.sin(np.pi*x[0])*x[1],  # exact solution at one
            1.0)                                # given parameter value
else:
    data = ( AffineObject([1.0], [1.0]),   # f: right-hand side
            [AffineObject([1.0], [1.0])],  # g: list of dirichlet boundary conditions
            [AffineObject([1.0], [1.0])])  # h: list of neumann boundary conditions
    
B, f, U, V = weak_problem(msh, (A,b,c), data, dbdry, nbdry)
B, f = assemble_system(B, f, U, V)

def solve(mu):
    u = spsolve(B(mu), f(mu))
    return U.set_dirichletbcs(mu, u)

print(f"System matrix shape: {B.shape}")
print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

########################################
# plott results

mus = [1.0, 2.0]  # parameter value at which to solve the problem

if use_exact:
    U_ref = utils.change_element(U.space, add_degree=1)
    u_ref = utils.interpolate_function(U_ref, data[0]).x.array
    
    plotter = pv.Plotter(shape=(1,len(mus)+1))
    plotter.subplot(0, 0)
    utils.plot_pyvista(u_ref, U_ref, f"exact solution (mu={data[1]})", plotter)
    for i, mu in enumerate(mus):
        plotter.subplot(0, i+1)
        utils.plot_pyvista(solve(mu), U.space, f"solution (mu={mu})", plotter)
    plotter.show()
else:
    plotter = pv.Plotter(shape=(1,len(mus)))
    for i, mu in enumerate(mus):
        plotter.subplot(0, i)
        utils.plot_pyvista(solve(mu), U.space, f"solution (mu={mu})", plotter)
    plotter.show()