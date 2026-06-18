import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from dolfinx import mesh
import ufl

from ulmRBM.affine import AffineObject, AffineLinear, wrap_affinelinear
from ulmRBM.fenicsx import utils, norms, utils_nr
from ulmRBM.fenicsx.problems import simple_wave, assemble_system, assemble_vector
from ulmRBM.solver import DirectSolver, IterativeSolver
from ulmRBM.products import OperatorInnerProduct
from ulmRBM.fom import FOM
from ulmRBM import rom
from ulmRBM.reductors import greedy_rbm

########################################
# FULL-ORDER MODEL
########################################

solver = DirectSolver()

I = [0.0, 2.0]
Omega = [0.0, 1.0]

mu_range = (80, 1e2)

nx = 10
nt = np.ceil(mu_range[1]*(I[1]-I[0])/((Omega[1]-Omega[0])/(nx+1))).astype('int')  # ensure CFL
print(f"\nnt = {nt}, nx = {nx}, mu_max = {mu_range[1]}")

use_exact = False    # wheter to calculate data from an exact solution
if use_exact:
    exact_sol = lambda x: np.sin(np.pi*x[0])*x[1]  # exact solution at mu=1.0
    B, f, U, V = simple_wave(I, Omega, nt, [nx], exact_sol=exact_sol, exact_mu=1.0)
else:
    f =  AffineObject([1.0], [1.0])   # right-hand side
    u0 = AffineObject([0.0], [1.0])   # initial condition u(0)
    u1 = AffineObject([0.0], [1.0])   # initial velocity u_t(0)
    g = AffineObject([0.0], [1.0])    # boundary condition on IxGamma
    B, f, U, V = simple_wave(I, Omega, nt, [nx], f, u0, u1, g)
    
B, f = assemble_system(B, f, U, V)

print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

V_H10 = norms.h10(V, solver)
U_BiV = OperatorInnerProduct(B, V_H10.dual, solver)
# U_H10 = norms.h10(U, solver)

def l_matvec(u):
    return utils_nr.eval_int_u_T(U.set_dirichletbcs(np.nan,u.flatten()), 0, U.space)
def l_matmat(U_mat):
    return np.array([utils_nr.eval_int_u_T(U.set_dirichletbcs(np.nan,col_u), 0, U.space) for col_u in U_mat.T])
l_output = LinearOperator((1,B.shape[1]), matvec=l_matvec, matmat=l_matmat)

fom_intpol = FOM(B, f, U_BiV, V_H10, stability=1.0, continuity=1.0, solver=solver, l=l_output)
# fom_H10 = FOM(B, f, U_H10, V_H10, solver=solver)

print(f"\nFOM dimension: {fom_intpol.dim}")

msh = U.space.mesh
tdim = msh.topology.dim
terminal_bdry = [lambda tx: utils.isclose(tx[0], I[1])]
terminal_bdry = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  terminal_bdry]
ds = utils.create_measure("ds", msh, tdim-1, terminal_bdry)
u = ufl.TrialFunction(U.space)
l2 = assemble_vector(u * ds).reshape(-1,1)
l_f = l2[U.dofs].T
l0 = wrap_affinelinear(l2)
s0 = sum([l0.apply2data(lambda lq: lq[bc.dofs]).T @ bc for bc in U.bcs])

fom_dofs = FOM(B, f, U_BiV, V_H10, stability=1.0, continuity=1.0, solver=solver, l=l_f)

mus = [np.random.uniform(*mu_range) for i in range(50)]
mus = [np.random.uniform(*mu_range) for i in range(50)]
mus = np.sort(mus)

print(f"\nTest with {len(mus)} random mus:")

plot_u = False
plot_u_T = False

s_foms = np.zeros_like(mus)
s_foms_2 = np.zeros_like(mus)
s_foms_3 = np.zeros_like(mus)

err = np.zeros_like(mus)
err2 = np.zeros_like(mus)

for i, mu in enumerate(mus):
    print(".", end="",flush=True)
    u_intpol = fom_intpol.solve(mu)
    u_intpol_wb = U.set_dirichletbcs(mu, u_intpol)
    s_foms_3[i] = fom_dofs.output(mu)[0] + s0(mu)[0]
    s_foms[i] = fom_intpol.output(mu, u_intpol)[0]
    s_foms_2[i] = l2.reshape(-1) @ u_intpol_wb

err = np.abs(s_foms-s_foms_2)
err2 = np.abs(s_foms_3-s_foms_2)

print(f"\nmax err 1: {np.max(err)}")
print(f"\nmax err 2: {np.max(err2)}")

fig = plt.figure()
fig.suptitle("Output")
plt.plot(mus, s_foms,label='Intpol',linestyle='',marker='o')
plt.plot(mus, s_foms_2,label='DOFs',linestyle='',marker='x')
plt.xlabel('mu')
plt.legend()
plt.show()

# ########################################
# # REDUCED-ORDER MODEL
# ########################################

# rom_BiV = rom.ROM(fom_BiV, stability=rom.ExactStability(fom_BiV))
# # rom_H10 = rom.Trial2TestROM(fom_H10, stability=rom.StabilityMinTheta())

# ########################################
# # MODEL REDUCTION
# ########################################

# mu_train = np.linspace(mu_range[0], mu_range[1], 1000).tolist()
# Nmax = 100
# tol = 1e-5

# print("\nGreedy algorithm for BiV inner product:")
# _, err_BiV, idx_BiV = greedy_rbm(rom_BiV, mu_train, Nmax, tol)

# # print("\nGreedy algorithm for H10 inner product:")
# # _, err_H10, idx_H10 = greedy_algorithm(rom_H10, mu_train, Nmax, tol)


# ########################################
# # PLOT SOLUTION
# ########################################
    
# mus = [np.random.uniform(*mu_range) for i in range(50)]
# mus = np.sort(mus)

# print(f"\nTest with {len(mus)} random mus:")

# plot_u = False
# plot_u_T = False

# s_foms = np.zeros_like(mus)
# s_roms = np.zeros_like(mus)

# err_state = np.zeros_like(mus)

# for i, mu in enumerate(mus):
#     u_fom = fom_BiV.solve(mu)
#     u_rom = rom_BiV.solve(mu)
#     u_rom_rec = rom_BiV.reconstruct(mu, u_rom)
#     u_fom_wb = U.set_dirichletbcs(mu, u_fom)
#     u_rom_wb = U.set_dirichletbcs(mu, u_rom_rec)          
#     if plot_u_T:
#         fig = plt.figure()
#         fig.suptitle(f"mu={mu:.2f}")
#         utils_nr.plot_endtime_pyvista(u_fom_wb, 0, U.space, "FOM")
#         utils_nr.plot_endtime_pyvista(u_rom_wb, 0, U.space, "ROM")
#         plt.legend()
#     if plot_u:
#         plotter = pv.Plotter(shape=(1, 2), title=f"u(T) for mu = {mu:.2e}")
#         plotter.subplot(0,0)
#         utils.plot_pyvista(u_fom_wb, U.space, f"FOM solution (mu={mu:.2f})", plotter)
#         plotter.subplot(0,1)
#         utils.plot_pyvista(u_rom_wb, U.space, f"ROM BiV solution (mu={mu:.2f})", plotter)
#         plotter.show(interactive_update=True)
#         # plotter.show()
#     s_foms[i] = fom_BiV.output(mu, u_fom)[0]
#     s_roms[i] = rom_BiV.output(mu, u_rom)
#     err_state[i] = fom_BiV.U.norm(mu, u_fom - u_rom_rec)
#     print(".", end="",flush=True)
# print("\n")

# fig = plt.figure()
# fig.suptitle("Output")
# plt.plot(mus, s_foms,label='FOM',linestyle='',marker='o')
# plt.plot(mus, s_roms,label='ROM',linestyle='',marker='x')
# plt.xlabel('mu')
# plt.legend()


# fig = plt.figure()
# fig.suptitle("Error")
# plt.semilogy(mus, err_state,linestyle=':',marker='o',label='state')
# plt.semilogy(mus, np.abs(s_foms-s_roms),linestyle=':',marker='o',label='output')
# plt.xlabel('mu')
# plt.legend()
# plt.show()

    
# if use_exact:
#     mu = 1.0
#     plotter = pv.Plotter(shape=(2, 2))
#     plotter.subplot(0,0)
#     utils.plot_pyvista(utils.interpolate_function(U.space, exact_sol).x.array, U.space, f"exact solution (mu={1:.2f})", plotter)
#     plotter.subplot(0,1)
#     utils.plot_pyvista(U.set_dirichletbcs(mu, fom_BiV.solve(mu)), U.space, f"FOM solution (mu={mu:.2f})", plotter)
#     plotter.subplot(1,0)
#     utils.plot_pyvista(U.set_dirichletbcs(mu, rom_H10.reconstruct(mu)), U.space, f"ROM H10 solution (mu={mu:.2f})", plotter)
#     plotter.subplot(1,1)
#     utils.plot_pyvista(U.set_dirichletbcs(mu, rom_BiV.reconstruct(mu)), U.space, f"ROM BiV solution (mu={mu:.2f})", plotter)
#     plotter.show(interactive_update=True)
    
    
# plt.figure()
# plt.loglog(err_BiV, label='BiV')
# plt.loglog(err_H10, label='H10')
# plt.xlabel('greedy iteration')
# plt.ylabel('max. error over training set')
# plt.legend()

# plt.figure()
# idx = np.zeros(len(mu_train), dtype=bool)
# idx[idx_BiV] = True
# plt.plot(idx, 'o', label='BiV')
# idx = np.zeros(len(mu_train), dtype=bool)
# idx[idx_H10] = True
# plt.plot(idx, '+', label='H10')
# plt.xlabel('greedy iteration')
# plt.ylabel('max. error over training set')
# plt.legend()

# plt.show()

