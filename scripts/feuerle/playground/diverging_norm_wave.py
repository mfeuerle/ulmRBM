import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from ulmRBM.fenicsx import utils, norms, SpaceTimeKey
from ulmRBM.fenicsx.problems import simple_wave_hilbert, simple_wave_structured, simple_wave
from ulmRBM.solver import DirectSolver, IterativeSolver
from ulmRBM.products import OperatorInnerProduct

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME

eigenvalues = 'direct'
B_solver = DirectSolver()
U_solver = lambda: DirectSolver()
V_solver = lambda: DirectSolver()

Nx = np.linspace(10, 1010, 5, dtype='int')
Kt = Nx

alpha = 1/6

f  = 0.0
u0 = 0.0
# u1 = lambda x: abs(x[0])**(alpha-1)
u1 = lambda tx: abs(tx[1])**(alpha-1)
g  = 0.0

# u1 = lambda x: utils.isclose(x[0], 0.0, atol=1e-12) * 1.0
# u1 = lambda tx: utils.isclose(tx[1], 0.0, atol=1e-12) * 1.0

# u1 = 0.0
# f = lambda t,x: utils.isclose(x[0], t) * 1.0
# f = lambda tx: np.isclose(tx[1], tx[0]) * 1.0

mu = 1.0

norm_H10 = np.zeros(len(Nx))
norm_BiV = np.zeros(len(Nx))

for i, (nx,kt) in enumerate(zip(Nx, Kt)):
    print(f"nx = {nx}, kt = {kt}")
    
    # B_, f_, U_fnx, V_fnx = simple_wave_structured(kt, nx, f, g, u0, u1)
    B_, f_, U_fnx, V_fnx = simple_wave(kt, nx, f, g, u0, u1)

    print(f"fom dimension: {B_.shape[0]} x {B_.shape[1]}\n")

    # U_H10 = norms.space_time(U_fnx, 
    #                     [{TIME: 'l2',  SPACE: 'h10'},
    #                      {TIME: 'h10', SPACE: 'l2'}],
    #                     solver=U_solver())
    # V_H10 = norms.space_time(V_fnx, 
    #                     [{TIME: 'l2',  SPACE: 'h10'},
    #                      {TIME: 'h10', SPACE: 'l2'}],
    #                     solver=V_solver())
    U_H10 = norms.h10(U_fnx, U_solver())
    V_H10 = norms.h10(V_fnx, V_solver())
    
    U_BiV = OperatorInnerProduct(B_, V_H10.dual, solver=B_solver)
    
    u = B_solver(B_(mu), f_(mu))
    
    norm_H10[i] = U_H10.norm(mu, u)
    norm_BiV[i] = U_BiV.norm(mu, u)
    
    np.savetxt(str(pathlib.Path(__file__).parent.resolve() / 'sol_convergence.csv'), 
               np.column_stack((Nx[:i+1],Kt[:i+1],norm_H10[:i+1],norm_BiV[:i+1])),
               header='Nx, Kt, norm_H10, norm_BiV')
    
    # u_bcs = U_fnx.set_dirichletbcs(mu, u, True)
    
    # T, X = np.meshgrid(U_fnx.space[TIME].mesh.geometry.x[:,0], U_fnx.space[SPACE].mesh.geometry.x[:,0])
    # fig = plt.figure(f'nx = {nx}')
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(T, X, u_bcs, cmap="viridis")
    # ax.set_xlabel('$t$')
    # ax.set_ylabel('$x$')
    # ax.set_zlabel('$u(t,x)$')
    # plt.show(block=False)
    # plt.draw()
    # plt.pause(0.01)
    
    # u_bcs = U_fnx.set_dirichletbcs(mu, u)
    
    # plotter = pv.Plotter()
    # utils.plot_pyvista(u_bcs, U_fnx.space, f"FOM solution (c={mu:.2f})", plotter)    
    # plotter.show(interactive_update=True)
    # plotter.show()


plt.figure()
plt.loglog(Nx, norm_H10, label='H10')
plt.loglog(Nx, norm_BiV, label='BiV')
plt.legend()
plt.xlabel('$n_x$')
plt.ylabel(r'$\|u\|$')
plt.show()
