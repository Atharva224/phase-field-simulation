import numpy as np
import matplotlib.pyplot as plt
from fipy import Grid2D, CellVariable, TransientTerm, DiffusionTerm

# --- Parameters ---
Vm = 1e-5
f0 = 1e6
Kc = 8.11e-7
M = 1e-17
dx = 1e-8
L = 1e-6
nx = ny = int(L / dx)
dt = 1.0
steps = 100
output_interval = 20

# Equilibrium concentrations
c_gamma_e = 0.16
c_gamma_dash_e = 0.23
c_mean = (c_gamma_e + c_gamma_dash_e) / 2

# --- Mesh and Variables ---
mesh = Grid2D(dx=dx, dy=dx, nx=nx, ny=ny)
c = CellVariable(name="concentration", mesh=mesh, hasOld=True)
mu = CellVariable(name="chemical_potential", mesh=mesh)

# --- Initial Condition: random noise around mean ---
np.random.seed(42)
noise = 0.8 * (np.random.rand(nx * ny) - 0.5)
c[:] = c_mean + noise

# --- Bulk chemical potential ---
def compute_mu_bulk(c):
    return -2 * (f0/Vm) * (c - c_gamma_e) * (c_gamma_dash_e - c) * (c_gamma_dash_e - 2*c + c_gamma_e)

# --- Time loop ---
for step in range(steps + 1):
    c.updateOld()
    mu.setValue(compute_mu_bulk(c) - Kc * c.faceGrad.divergence)
    eq = TransientTerm() == DiffusionTerm(coeff=M)
    eq.solve(var=c, dt=dt)

    if step % output_interval == 0:
        print(f"Step {step} — Time = {step * dt:.1f} s")

        # --- Sharp, grid-mapped plot ---
        plt.figure(figsize=(6, 6), dpi=300)
        plt.imshow(
            c.value.reshape((nx, ny)),
            cmap='jet',                # High-contrast colormap
            interpolation='none',     # No smoothing
            origin='lower',           # (0,0) at bottom-left
            extent=[0, nx, 0, ny]     # Grid index units
        )
        plt.title(f"Time step: {step}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="Concentration")
        plt.tight_layout()
        plt.savefig(f"spinodal_pattern_t{step}.png", dpi=300)
        plt.close()
