import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from ulmRBM.affine import AffineObject
from ulmRBM.fenicsx import utils, norms, SpaceTimeKey
from ulmRBM.fenicsx.problems import simple_wave, simple_wave_structured, simple_wave_hilbert
from ulmRBM.fom import FOM
from ulmRBM.solver import DirectSolver

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME

########################################
# Model Parameters
########################################

eigenvalues = 'direct'
B_solver = lambda: DirectSolver()
U_solver = lambda: DirectSolver(factorize=True)
V_solver = lambda: DirectSolver(factorize=True)

c = 1.0    # wave speed

nx = [20] * 1
K = np.ceil(max(1,c) * nx[0]).astype('int')  # ensure CFL
# K = 10    # violate the CFL condition

print(f"\nK = {K}, nx = {nx[0]}, c = {max(1,c)}")

f  = 1.0   # right-hand side
u0 = 0.0   # initial condition u(0)
u1 = 0.0   # initial velocity u_t(0)
g  = 0.0   # boundary condition on IxGamma

# f = lambda t,x: np.sin(np.pi*t)*np.sin(np.pi*x[0])
# u0 = lambda x: np.sin(np.pi * x[0])
# u1 = lambda x: np.ones(x[0].shape)
# g = lambda t,x: t*x[0]


########################################
# Unstructured Wave discretization
########################################

def unstructured(K, nx, f, g, u0, u1):
    
    def merge_variables(f, space:bool, time:bool):
        # convert a function f(t,x), f(t) or f(x) into a function f(tx) where tx = (t,x)
        def merge(f):
            if not callable(f): return f
            if space and time:  return lambda tx: f(tx[0], tx[1:])
            elif space:         return lambda tx: f(tx[1:])
            elif time:          return lambda f: lambda tx: f(tx[0])
            else: raise ValueError("Huh?")
        return f.apply2data(merge) if isinstance(f, AffineObject) else merge(f)
    
    f  = merge_variables(f,  space=True, time=True)
    u0 = merge_variables(u0, space=True, time=False)
    u1 = merge_variables(u1, space=True, time=False)
    g  = merge_variables(g,  space=True, time=True)
    
    B, f, U_fnx, V_fnx = simple_wave(K, nx, f, g, u0, u1)
    V_H10 = norms.h10(V_fnx, V_solver())
    U_H10 = norms.h10(U_fnx, U_solver())
    
    fom = FOM(B, f, U_H10, V_H10, stability=eigenvalues, continuity=eigenvalues, solver=B_solver())
    set_dbcs = lambda u: U_fnx.set_dirichletbcs(None, u)
    
    return fom, set_dbcs, U_fnx, V_fnx


def structured(K, nx, f, g, u0, u1):
    
    B, f, U_fnx, V_fnx = simple_wave_structured(K, nx, f, g, u0, u1)
    U_H10 = norms.space_time(U_fnx, 
                        [{TIME: 'l2',  SPACE: 'h10'},
                         {TIME: 'h10', SPACE: 'l2'}],
                        solver=U_solver())
    V_H10 = norms.space_time(V_fnx, 
                        [{TIME: 'l2',  SPACE: 'h10'},
                         {TIME: 'h10', SPACE: 'l2'}],
                        solver=V_solver())
    
    fom = FOM(B, f, U_H10, V_H10, stability=eigenvalues, continuity=eigenvalues, solver=B_solver())
    set_dbcs = lambda u: U_fnx.set_dirichletbcs(None, u, True)
    
    return fom, set_dbcs, U_fnx, V_fnx


def hilbert(K, nx, f, g, u0, u1):
    
    B, f, U_fnx, V_fnx = simple_wave_hilbert(K, nx, f, g, u0, u1)
    U_H10 = norms.space_time(U_fnx, 
                        [{TIME: 'l2',  SPACE: 'h10'},
                         {TIME: 'h10', SPACE: 'l2'}],
                        solver=U_solver())
    V_H10 = norms.space_time(V_fnx, 
                        [{TIME: 'l2',  SPACE: 'h10'},
                         {TIME: 'h10', SPACE: 'l2'}],
                        solver=V_solver())
    
    fom = FOM(B, f, U_H10, V_H10, stability=eigenvalues, continuity=eigenvalues, solver=B_solver())
    set_dbcs = lambda u: U_fnx.set_dirichletbcs(None, u, True)
    
    return fom, set_dbcs, U_fnx, V_fnx

########################################
# Assemble discrete problems
########################################

PROBLEMS = ['unstructured', 'structured', 'hilbert']

fom, set_dbcs, U_fnx, V_fnx = {}, {}, {}, {}
for P in PROBLEMS:
    fom[P], set_dbcs[P], U_fnx[P], V_fnx[P] = eval(P)(K, nx, f, g, u0, u1)
    
u = {P: fom[P].solve(c) for P in PROBLEMS}
u_bcs = {P: set_dbcs[P](u[P]) for P in PROBLEMS}

print("Stability constants:")
print("".join(f"{P}: {fom[P].stability(c)}\n" for P in PROBLEMS))

print("Continuity constants:")
print("".join(f"{P}: {fom[P].continuity(c)}\n" for P in PROBLEMS))

print("Norms of solutions:")
print("".join(f"{P}: {fom[P].U.norm(c, u[P])}\n" for P in PROBLEMS))


########################################
# Plot results
########################################

plotter = pv.Plotter()
utils.plot_pyvista(u_bcs['unstructured'], U_fnx['unstructured'].space, f"FOM solution (c={c:.2f})", plotter)    
plotter.show(interactive_update=True)

if np.isscalar(nx) or len(nx) == 1:
    T, X = np.meshgrid(U_fnx['structured'].space[TIME].mesh.geometry.x[:,0], U_fnx['structured'].space[SPACE].mesh.geometry.x[:,0])

    fig = plt.figure(f'solution c = {c:.2f}')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(T, X, u_bcs['structured'], cmap="viridis")
    ax.set_title('Tensor product discretization')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(t,x)$')
    
    T, X = np.meshgrid(U_fnx['hilbert'].space[TIME].mesh.geometry.x[:,0], U_fnx['hilbert'].space[SPACE].mesh.geometry.x[:,0])
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(T, X, u_bcs['hilbert'], cmap="viridis")
    ax.set_title('Hilbert tensor product discretization')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u(t,x)$')
    
    plt.show()


plotter.show()