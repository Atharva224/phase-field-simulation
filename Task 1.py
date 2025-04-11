from fipy import *
import numpy as np
import matplotlib.pyplot as plt

# Define the computational grid
dx = 0.5  # Grid spacing
Lx = 100.0  # Total length of the domain
nx = int(Lx / dx)  # Number of grid points
mesh = Grid1D(dx=dx, nx=nx)

# Define initial condition function
def initialize_phi():
    phi = CellVariable(mesh=mesh, hasOld=True)
    phi.setValue(1.0, where=mesh.cellCenters[0] < Lx / 2)
    phi.setValue(0.0, where=mesh.cellCenters[0] >= Lx / 2)
    return phi

# Parameter sets to test
param_sets = [
    {"f0": 1.0, "K_phi": 1.0, "L": 1.0},
    {"f0": 2.0, "K_phi": 1.0, "L": 1.0},
    {"f0": 1.0, "K_phi": 5.0, "L": 1.0},
    {"f0": 1.0, "K_phi": 1.0, "L": 5.0},
    {"f0": 2.0, "K_phi": 5.0, "L": 5.0},
]

# Loop over different parameter sets
for i, params in enumerate(param_sets):
    f0 = params["f0"]
    K_phi = params["K_phi"]
    L = params["L"]

    # Initialize φ
    phi = initialize_phi()

    # Define governing equation
    bulk_energy_density = f0 * phi * (1 - phi) * (1 - 2 * phi)
    gradient_energy = DiffusionTerm(coeff=K_phi)
    eq = TransientTerm() == -L * (bulk_energy_density - gradient_energy)

    # Boundary conditions
    phi.faceGrad.constrain(0.0, where=mesh.facesLeft | mesh.facesRight)

    # Time-stepping parameters
    dt = 0.1  # Time step
    steps = 100  # Number of iterations

    # Arrays to store energy evolution
    total_energy = []

    # Time evolution loop
    for step in range(steps):
        phi.updateOld()
        res = 1e5
        while res > 1e-3:  # Iterative solver until convergence
            res = eq.sweep(var=phi, dt=dt)

        # Compute total free energy F at each time step
        bulk_energy = (f0 * phi * (1 - phi) ** 2).value.sum() * dx
        grad_energy = (K_phi * 0.5 * (phi.grad.mag ** 2)).value.sum() * dx
        total_energy.append(bulk_energy + grad_energy)

    # Compute energy densities for the final state
    bulk_energy_density_final = f0 * phi * (1 - phi) ** 2
    grad_energy_density_final = K_phi * 0.5 * (phi.grad.mag ** 2)
    total_energy_density_final = bulk_energy_density_final + grad_energy_density_final

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Order Parameter Distribution
    axs[0].plot(mesh.cellCenters[0], phi.value, label=r"$\phi(x)$")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel(r"$\phi(x)$")
    axs[0].set_title(f"Order Parameter (f0={f0}, K_phi={K_phi}, L={L})")
    axs[0].legend()
    axs[0].grid()

    # Total Free Energy Evolution
    axs[1].plot(np.arange(steps) * dt, total_energy, label=r"$F$")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("F")
    axs[1].set_title(f"Total Energy Evolution (f0={f0}, K_phi={K_phi}, L={L})")
    axs[1].legend()
    axs[1].grid()

    # Energy Density Distribution
    axs[2].plot(mesh.cellCenters[0], bulk_energy_density_final, label=r"$f_{bulk}(x)$", color="blue")
    axs[2].plot(mesh.cellCenters[0], grad_energy_density_final, label=r"$f_{grad}(x)$", color="red")
    axs[2].plot(mesh.cellCenters[0], total_energy_density_final, label=r"$f_{total}(x)$", color="black")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Energy Density")
    axs[2].set_title(f"Energy Density (f0={f0}, K_phi={K_phi}, L={L})")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"phase_field_case_{i+1}.png")
    plt.close()

print("All plots generated and saved as PNG files.")
