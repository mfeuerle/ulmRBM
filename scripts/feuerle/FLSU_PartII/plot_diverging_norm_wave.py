import pathlib

import numpy as np
import matplotlib.pyplot as plt

from ulmRBM.fenicsx import  norms, SpaceTimeKey
from ulmRBM.fenicsx.problems import simple_wave
from ulmRBM.solver import DirectSolver
from ulmRBM.products import OperatorInnerProduct

SPACE = SpaceTimeKey.SPACE
TIME  = SpaceTimeKey.TIME

eigenvalues = 'direct'
B_solver = DirectSolver()
V_solver = DirectSolver()

Nx = np.linspace(10, 4010, 41, dtype='int')
Kt = Nx

alpha = 1/32

f  = 0.0
u0 = 0.0
u1 = lambda tx: abs(tx[1])**(alpha-1)   # tx[0] = t, tx[1] = x
g  = 0.0

mu = 1.0

norm_H10 = np.zeros(len(Nx))
norm_BiV = np.zeros(len(Nx))

for i, (nx,kt) in enumerate(zip(Nx, Kt)):
    print(f"nx = {nx}, kt = {kt}")
    
    B_, f_, U_fnx, V_fnx = simple_wave(kt, nx, f, g, u0, u1)

    print(f"fom dimension: {B_.shape[0]} x {B_.shape[1]}\n")
    
    U_H10 = norms.h10(U_fnx)
    V_H10 = norms.h10(V_fnx, V_solver)
    U_BiV = OperatorInnerProduct(B_, V_H10.dual, solver=B_solver)
    
    u = B_solver(B_(mu), f_(mu))
    
    norm_H10[i] = U_H10.norm(mu, u)
    norm_BiV[i] = U_BiV.norm(mu, u)
    
    np.savetxt(str(pathlib.Path(__file__).parent.resolve() / 'plot_diverging_norm_wave_data.txt'), 
               np.column_stack((Nx[:i+1],Kt[:i+1],norm_H10[:i+1],norm_BiV[:i+1])),
               header='Nx, Kt, norm_H10, norm_BiV')

plt.figure()
plt.loglog(Nx, norm_H10, label='H10')
plt.loglog(Nx, norm_BiV, label='BiV')
plt.legend()
plt.xlabel('$n_x$')
plt.ylabel(r'$\|u\|$')
plt.show()
