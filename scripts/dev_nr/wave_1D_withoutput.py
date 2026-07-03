import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from dolfinx import mesh
import ufl

from ulmRBM.affine import AffineObject, AffineLinear, wrap_affinelinear
from ulmRBM.fenicsx import utils, norms
from ulmRBM.fenicsx.problems import simple_wave_with_output
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

mu_range = (60, 1e2)

nx = 10
nt = np.ceil(mu_range[1]*(I[1]-I[0])/((Omega[1]-Omega[0])/(nx+1))).astype('int')  # ensure CFL
print(f"\nnt = {nt}, nx = {nx}, mu_max = {mu_range[1]}")

# use_exact = False    # wheter to calculate data from an exact solution
# if use_exact:
#     exact_sol = lambda x: np.sin(np.pi*x[0])*x[1]  # exact solution at mu=1.0
#     B, f, U, V = simple_wave(I, Omega, nt, [nx], exact_sol=exact_sol, exact_mu=1.0)
# else:
f =  AffineObject([1.0], [1.0])   # right-hand side
u0 = AffineObject([0.0], [1.0])   # initial condition u(0)
u1 = AffineObject([0.0], [1.0])   # initial velocity u_t(0)
g = AffineObject([0.0], [1.0])    # boundary condition on IxGamma
# B, f, U, V = simple_wave(I, Omega, nt, [nx], f, u0, u1, g)
B, f, U, V, l, s0 = simple_wave_with_output(nt, [nx], f, u0, u1, g)
    # compute output
    # msh = U.space.mesh
    # tdim = msh.topology.dim
    # terminal_bdry = [lambda tx: utils.isclose(tx[0], I[1])]
    # terminal_bdry = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  terminal_bdry]
    # ds = utils.create_measure("ds", msh, tdim-1, terminal_bdry)
    # u = ufl.TrialFunction(U.space)
    # l = assemble_vector(u * ds).reshape(1,-1)
    
# B, f, U,= assemble_system(B, f, U, V, l)

print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

V_H10 = norms.h10(V, solver)
U_BiV = OperatorInnerProduct(B, V_H10.dual, solver)

# msh = U.space.mesh
# tdim = msh.topology.dim
# terminal_bdry = [lambda tx: utils.isclose(tx[0], I[1])]
# terminal_bdry = [mesh.locate_entities_boundary(msh, tdim-1, bdry) for bdry in  terminal_bdry]
# ds = utils.create_measure("ds", msh, tdim-1, terminal_bdry)
# u = ufl.TrialFunction(U.space)
# l = assemble_vector(u * ds).reshape(1,-1)
# l0 = wrap_affinelinear(l)
# s0 = sum([l0.apply2data(lambda lq: lq[:,bc.dofs]) @ bc for bc in U.bcs])

fom_wave = FOM(B, f, U_BiV, V_H10, stability=1.0, continuity=1.0, solver=solver, l=l)

########################################
# REDUCED-ORDER MODEL
########################################

rom_wave = rom.ROM(fom_wave, stability=rom.ExactStability(fom_wave))

########################################
# MODEL REDUCTION
########################################

mu_train = np.linspace(mu_range[0], mu_range[1], 1000).tolist()
Nmax = 100
tol = 1e-5

print("\nGreedy algorithm for BiV inner product:")
_, err_BiV, idx_BiV = greedy_rbm(rom_wave, mu_train, Nmax, tol)


########################################
# PLOT SOLUTION
########################################
    
mus = [np.random.uniform(*mu_range) for i in range(50)]
mus = np.sort(mus)

print(f"\nTest with {len(mus)} random mus:")

plot_u = False

s_foms = np.zeros_like(mus)
s_roms = np.zeros_like(mus)

err_state = np.zeros_like(mus)

for i, mu in enumerate(mus):
    u_fom = fom_wave.solve(mu)
    u_rom = rom_wave.solve(mu)
    u_rom_rec = rom_wave.reconstruct(mu, u_rom)
    u_fom_wb = U.set_dirichletbcs(mu, u_fom)
    u_rom_wb = U.set_dirichletbcs(mu, u_rom_rec)          
    if plot_u:
        plotter = pv.Plotter(shape=(1, 2), title=f"u(T) for mu = {mu:.2e}")
        plotter.subplot(0,0)
        utils.plot_pyvista(u_fom_wb, U.space, f"FOM solution (mu={mu:.2f})", plotter)
        plotter.subplot(0,1)
        utils.plot_pyvista(u_rom_wb, U.space, f"ROM BiV solution (mu={mu:.2f})", plotter)
        plotter.show(interactive_update=True)
    s_foms[i] = fom_wave.output(mu, u_fom) + s0(mu)
    s_roms[i] = rom_wave.output(mu, u_rom) + s0(mu)
    err_state[i] = fom_wave.U.norm(mu, u_fom - u_rom_rec)
    print(".", end="",flush=True)
print("\n")

fig = plt.figure()
fig.suptitle("Output")
plt.plot(mus, s_foms,label='FOM',linestyle='',marker='o')
plt.plot(mus, s_roms,label='ROM',linestyle='',marker='x')
plt.xlabel('mu')
plt.legend()

fig = plt.figure()
fig.suptitle("Error")
plt.semilogy(mus, err_state,linestyle=':',marker='o',label='state')
plt.semilogy(mus, np.abs(s_foms-s_roms),linestyle=':',marker='o',label='output')
plt.xlabel('mu')
plt.legend()
plt.show()