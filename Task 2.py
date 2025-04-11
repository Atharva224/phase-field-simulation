import numpy as np
import matplotlib.pyplot as plt

# Given values
f0 = 9.99e7  # Computed f0 (J/mol)
c_e_gamma = 0.16  # Equilibrium composition for γ phase
c_e_gamma_prime = 0.23  # Equilibrium composition for γ' phase

# Define composition range
c_values = np.linspace(0.15, 0.24, 100)  # Slightly extended range for smooth curve

# Compute bulk energy density function
f_bulk = f0 * (c_e_gamma_prime - c_values)**2 * (c_values - c_e_gamma)**2

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(c_values, f_bulk, label=r"$f_{\text{bulk}}(c) = f_0 (c^{e}_{\gamma'} - c)^2 (c - c^{e}_{\gamma})^2$", color="red")
plt.axvline(x=c_e_gamma, linestyle="--", color="blue", label=r"$c^{e}_{\gamma}$")
plt.axvline(x=c_e_gamma_prime, linestyle="--", color="green", label=r"$c^{e}_{\gamma'}$")
plt.axvline(x=0.195, linestyle="--", color="black", label=r"$c_{\text{peak}}$")

# Labels and title
plt.xlabel("Composition (c)")
plt.ylabel(r"Bulk Energy Density $f_{\text{bulk}}$ (J/mol)")
plt.title("Verification of Bulk Energy Density Function")
plt.legend(fontsize=8, loc="upper right", framealpha=0.8, labelspacing=0.5)
plt.grid()

# Save the figure
plt.savefig("bulk_energy_verification.png")
plt.show()
