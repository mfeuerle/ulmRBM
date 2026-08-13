
from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh, fem

from ulmRBM.fenicsx import utils

def arrange_dofs_for_plot(U: fem.FunctionSpace, u: np.ndarray) -> np.ndarray:
    degree = U.ufl_element().basix_element.degree
    u = u.copy()
    if degree == 0: raise NotImplementedError("arrange_1D not implemented for degree 0 elements.")
    order = list(range(1,degree)) + [0]
    u[1:] = u[1:].reshape((-1,degree))[:,order].reshape(-1)
    x = np.linspace(0, 1, len(u))
    return x, u

u_func = lambda x: np.sin(np.pi*x[0])

class LVL(IntEnum):
    COARSE = 0
    FINE   = 1

msh = {LVL.COARSE: mesh.create_unit_interval(MPI.COMM_WORLD, 4),
       LVL.FINE:   mesh.create_unit_interval(MPI.COMM_WORLD, 10)}

for lvl in LVL: msh[lvl].topology.create_connectivity(msh[lvl].topology.dim-1, msh[lvl].topology.dim)

U = {LVL.COARSE: fem.functionspace(msh[LVL.COARSE], ("Lagrange", 1)),
     LVL.FINE:   fem.functionspace(msh[LVL.FINE],   ("Lagrange", 2))}

u = {lvl: fem.Function(U[lvl]) for lvl in LVL}
for lvl in LVL: u[lvl].interpolate(u_func)
u = {lvl: u[lvl].x.array.copy() for lvl in LVL}


P_CF, P_FC = utils.projection_matrices(U[LVL.COARSE], U[LVL.FINE], both=True)
Q_FC = utils.projection_matrices(U[LVL.FINE], U[LVL.COARSE])


fig = plt.figure()
fig.suptitle('Projections based on coarse to fine interpolation')
ax = fig.add_subplot(2, 2, 1)
ax.plot(*arrange_dofs_for_plot(U[LVL.COARSE], u[LVL.COARSE]), 'o-', label='coarse')
ax.plot(*arrange_dofs_for_plot(U[LVL.FINE],   P_CF @ u[LVL.COARSE]), 'x--', label='coarse to fine')
ax.set_title('Interpolation from coarse to fine')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x)$')
ax.set_ylim(-0.05,1.05)
ax.legend()

ax = fig.add_subplot(2, 2, 2)
ax.plot(*arrange_dofs_for_plot(U[LVL.FINE],   u[LVL.FINE]), 'o-', label='fine')
ax.plot(*arrange_dofs_for_plot(U[LVL.COARSE], P_FC @ u[LVL.FINE]), 'x--', label='fine to coarse')
ax.set_title('Projection from coarse to fine (via Moore–Penrose pseudoinverse)')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x)$')
ax.set_ylim(-0.05,1.05)
ax.legend()

ax = fig.add_subplot(2, 2, 4)
ax.plot(*arrange_dofs_for_plot(U[LVL.FINE],   u[LVL.FINE]), 'o-', label='fine')
ax.plot(*arrange_dofs_for_plot(U[LVL.COARSE], Q_FC @ u[LVL.FINE]), 'x--', label='fine to coarse')
ax.set_title('Interpolation from coarse to fine')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(x)$')
ax.set_ylim(-0.05,1.05)
ax.legend()

plt.show()