import numpy as np
import time

from ulmRBM.fenicsx import norms
from ulmRBM.fenicsx.problems import thermal_block, assemble_system
from ulmRBM.solver import DirectSolver
from ulmRBM.fom import FOM, GalerkinFOM

nblocks = [2, 3]
mu_range = (0.05, 2.0)
solver = DirectSolver()

B, f, U, V = thermal_block([15, 15], nblocks)
B, f = assemble_system(B, f, U, V)

print(f"System matrix shape: {B.shape}")
print(f"Number of affine terms in B: {len(B)}")
print(f"Number of affine terms in f: {len(f)}")
print(f"Number of affine terms in bcs: {sum(len(bc) for bc in U.bcs)}")

U_H10 = norms.h10(U, solver)
V_H10 = norms.h10(V, solver)

fom_pg = FOM(B, f, U_H10, V_H10, solver=solver)
fom_g  = GalerkinFOM(B, f, U_H10, solver=solver)

continuity_exact = lambda mu: np.max(mu)
coercivity_exact = lambda mu: np.min(mu)

mus = [np.random.uniform(*mu_range, size=nblocks) for _ in range(500)]


print(f"Calculating constants for {len(mus)} random parameters...")
start = time.time()
err_continuity_pg = [fom_pg.continuity(mu) - continuity_exact(mu) for mu in mus]
print(f"Max. error in continuity constant (PG): {np.max(np.abs(err_continuity_pg)):.2e}\t time taken: {time.time() - start:.2f} seconds")

start = time.time()
err_continuity_g = [fom_g.continuity(mu) - continuity_exact(mu) for mu in mus]
print(f"Max. error in continuity constant  (G): {np.max(np.abs(err_continuity_g)):.2e}\t time taken: {time.time() - start:.2f} seconds")

start = time.time()
err_stability_pg  = [fom_pg.stability(mu)  - coercivity_exact(mu) for mu in mus]
print(f"Max. error in stability constant (PG): {np.max(np.abs(err_stability_pg)):.2e}\t\t time taken: {time.time() - start:.2f} seconds")

start = time.time()
err_stability_g  = [fom_g.stability(mu)  - coercivity_exact(mu) for mu in mus]
print(f"Max. error in stability constant  (G): {np.max(np.abs(err_stability_g)):.2e}\t\t time taken: {time.time() - start:.2f} seconds")
