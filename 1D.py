import numpy as np
import fipy as fp
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
import matplotlib.pyplot as plt

# Given parameters from the task
Vm = 1e-5         # Molar volume (m³/mol)
M = 1e-17         # Interface mobility (mol² J⁻¹ m⁻¹ s⁻¹)
dt = 1            # Time scale (s)
dx = 1e-8         # Grid spacing (m)
c_gamma_e = 0.16  # Lower equilibrium concentration
c_gamma_dash_e = 0.23  # Upper equilibrium concentration
f0 = 1e6         # Free energy coefficient (J/m³) - REDUCED from 1e8 to 1e6

# Experimental interface energy range
F_inte_exp_min = 5e-3   # J/m²
F_inte_exp_max = 50e-3  # J/m²

# Function to run simulation and calculate interface energies for a given Kc
def calculate_interface_energy(Kc):
    # Simulation domain 
    L = 1e-6     # Domain length (m)
    nx = int(L / dx)
    mesh = Grid1D(dx=dx, nx=nx)

    # Initialize concentration field with a smooth interface
    c = CellVariable(name="concentration", mesh=mesh, value=c_gamma_e)
    x = mesh.cellCenters[0]
    c.setValue(c_gamma_e + (c_gamma_dash_e - c_gamma_e) * 0.5 * (1 + np.tanh((x - L/2) / (5 * dx))))
    
    # Define the free energy derivative and calculate energy
    def calculate_energy_components():
        # Get bulk values for reference
        bulk_low = c_gamma_e
        bulk_high = c_gamma_dash_e
        c_mid = (c_gamma_e + c_gamma_dash_e) / 2
        
        # Calculate bulk free energy at equilibrium concentrations
        f_bulk_low = f0 / Vm * (bulk_low - c_gamma_e)**2 * (bulk_low - c_gamma_dash_e)**2
        f_bulk_high = f0 / Vm * (bulk_high - c_gamma_e)**2 * (bulk_high - c_gamma_dash_e)**2
        
        # Bulk free energy term
        f_bulk = (f0 / Vm) * (c - c_gamma_e)**2 * (c - c_gamma_dash_e)**2
        
        # Gradient energy term 
        f_grad = Kc/2 * c.grad.mag**2
        
        # Create variables for energy components
        F_bulk = CellVariable(name="bulk_free_energy", mesh=mesh, value=f_bulk)
        F_grad = CellVariable(name="gradient_free_energy", mesh=mesh, value=f_grad)
        F_total = CellVariable(name="total_free_energy", mesh=mesh, value=f_bulk + f_grad)
        
        # Calculate interface width more precisely
        c_delta = c_gamma_dash_e - c_gamma_e
        interface_mask = (c.value > (c_gamma_e + 0.1*c_delta)) & (c.value < (c_gamma_dash_e - 0.1*c_delta))
        interface_width = np.sum(interface_mask) * dx  # in meters
        
        # Calculate excess free energy more accurately
        excess_bulk = np.zeros_like(f_bulk)
        for i in range(len(excess_bulk)):
            if c[i] < c_mid:
                excess_bulk[i] = f_bulk[i] - f_bulk_low
            else:
                excess_bulk[i] = f_bulk[i] - f_bulk_high
                
        # Only consider positive excess energy contributions
        excess_bulk[excess_bulk < 0] = 0
        
        # Create excess energy variable
        F_excess = CellVariable(name="excess_free_energy", mesh=mesh, value=excess_bulk)
        F_excess_total = F_excess + CellVariable(name="grad_energy", mesh=mesh, value=f_grad)
        
        # Calculate interface energy properly (integral of excess energy)
        interface_energy = float((F_excess_total * mesh.cellVolumes).sum())
        
        # Interface energy per unit area (for 1D model, divide by cross-sectional area of 1m²)
        interface_energy_density = interface_energy
        
        # Better interface width calculation
        grad_mag = np.array(c.grad.mag)
        if np.max(grad_mag) > 0:
            # Calculate characteristic width based on maximum gradient
            max_grad_idx = np.argmax(grad_mag)
            max_grad = grad_mag[max_grad_idx]
            interface_width_grad = c_delta / max_grad
        else:
            interface_width_grad = 0
        
        return {
            "F_bulk": F_bulk,
            "F_grad": F_grad,
            "F_total": F_total,
            "F_excess": F_excess_total,
            "total_F_bulk": float((F_bulk * mesh.cellVolumes).sum()),
            "total_F_grad": float((F_grad * mesh.cellVolumes).sum()),
            "total_F_interface": float((F_total * mesh.cellVolumes).sum()),
            "interface_energy": interface_energy,
            "interface_energy_density": interface_energy_density,
            "interface_width": interface_width,
            "interface_width_grad": interface_width_grad
        }
    
    # Create a chemical potential variable
    mu = CellVariable(name="chemical_potential", mesh=mesh)
    
    # Set up the equations for Cahn-Hilliard
    # First equation: μ = δF/δc = -2f0/Vm*(c-c_γe)(c_γe'-c)(c_γe'-2c+c_γe) - Kc∇²c
    # Second equation: ∂c/∂t = M∇²μ
    
    # Step 1: Define the function to calculate the bulk contribution of chemical potential
    def calculate_mu_bulk():
        return -2 * f0 / Vm * (c - c_gamma_e) * (c_gamma_dash_e - c) * (c_gamma_dash_e - 2*c + c_gamma_e)
    
    # Step 2: Calculate the chemical potential for the current concentration field
    def update_mu():
        mu_bulk = calculate_mu_bulk()
        # Calculate Laplacian of c 
        c_laplacian = c.faceGrad.divergence
        # Complete chemical potential: μ = μ_bulk - Kc∇²c
        mu.setValue(mu_bulk - Kc * c_laplacian)
    
    # Step 3: Define the equation for concentration evolution
    # ∂c/∂t = M∇²μ
    equation = TransientTerm() == DiffusionTerm(coeff=M)
    
    # Evolve the system for a number of time steps
    elapsed_time = 0
    total_steps = 300  
    
    for step in range(total_steps):
        if step > 50:  # After some initial steps
            old_c = c.value.copy()
            
            # Update the chemical potential
            update_mu()
            
            # Solve for the new concentration
            equation.solve(var=mu, dt=dt)
            
            # Check if system is close to equilibrium
            change = np.max(np.abs(c.value - old_c))
            if change < 1e-8:  # Adjust threshold as needed
                print(f"Equilibrium reached at step {step}")
                break
            
            elapsed_time += dt
        else:
            # Regular evolution for initial steps
            update_mu()
            equation.solve(var=mu, dt=dt)
            elapsed_time += dt
        
        # Print progress
        if step % 20 == 0:  
            print(f"Step {step}, Time: {elapsed_time} s")
    
    # After simulation has reached equilibrium, calculate the energy components
    energy_data = calculate_energy_components()
    
    # Add concentration field and mesh to the results
    energy_data["c"] = c
    energy_data["mesh"] = mesh
    
    return energy_data

# Wider range of Kc values with focus on higher values
Kc_values = np.logspace(-6.5, -5, 12)
results = []

for Kc in Kc_values:
    print(f"Testing Kc = {Kc:.3e}")
    result = calculate_interface_energy(Kc)
    results.append({
        "Kc": Kc,
        "interface_energy_density": result["interface_energy_density"],
        "interface_width": result["interface_width"],
        "interface_width_grad": result["interface_width_grad"],
        "total_F_bulk": result["total_F_bulk"],
        "total_F_grad": result["total_F_grad"]
    })
    print(f"Kc = {Kc:.3e}, Interface Energy = {result['interface_energy_density']:.6e} J/m², " +
          f"Width = {result['interface_width']:.2e} m, Width (grad) = {result['interface_width_grad']:.2e} m")
    print(f"Bulk Energy = {result['total_F_bulk']:.6e} J, Gradient Energy = {result['total_F_grad']:.6e} J")

# Find the Kc values that give interface energy in the expected range
valid_Kc = [r["Kc"] for r in results if F_inte_exp_min <= r["interface_energy_density"] <= F_inte_exp_max]

if valid_Kc:
    print("\nValid Kc values within the experimental interface energy range (5-50 × 10^-3 J/m²):")
    for Kc in valid_Kc:
        idx = [r["Kc"] for r in results].index(Kc)
        print(f"Kc = {Kc:.6e}, Interface Energy = {results[idx]['interface_energy_density']:.6e} J/m²")
        print(f"  Bulk/Gradient Energy Ratio: {results[idx]['total_F_bulk']/results[idx]['total_F_grad']:.2f}")
    
    # Use the middle valid Kc value for detailed analysis
    optimal_Kc = valid_Kc[len(valid_Kc) // 2]
else:
    # If no exact matches, find the closest value
    distances = [abs(r["interface_energy_density"] - (F_inte_exp_min + F_inte_exp_max)/2) for r in results]
    closest_idx = distances.index(min(distances))
    optimal_Kc = results[closest_idx]["Kc"]
    print(f"\nNo exact matches. Closest Kc value: {optimal_Kc:.6e}")

# Detailed analysis with the chosen Kc
print(f"\nDetailed analysis with Kc = {optimal_Kc:.6e}:")
detailed_result = calculate_interface_energy(optimal_Kc)

# Print the detailed results
print(f"F_bulk = {detailed_result['total_F_bulk']:.6e} J")
print(f"F_grad = {detailed_result['total_F_grad']:.6e} J")
print(f"F_inte = {detailed_result['interface_energy']:.6e} J")
print(f"Interface width = {detailed_result['interface_width']:.6e} m")
print(f"Interface width (by gradient) = {detailed_result['interface_width_grad']:.6e} m")
print(f"Interface energy density = {detailed_result['interface_energy_density']:.6e} J/m²")
print(f"Bulk/Gradient Energy Ratio: {detailed_result['total_F_bulk']/detailed_result['total_F_grad']:.2f}")

# Create plots
plt.figure(figsize=(12, 10))

# Plot the concentration profile
plt.subplot(2, 1, 1)
plt.plot(detailed_result["mesh"].cellCenters[0] * 1e9, detailed_result["c"].value, 'b-')
plt.xlabel('x (nm)')
plt.ylabel('Concentration (c)')
plt.title(f'Concentration Profile (Kc = {optimal_Kc:.3e})')
plt.grid(True)

# Plot the free energy components
plt.subplot(2, 1, 2)
x_values = detailed_result["mesh"].cellCenters[0] * 1e9  # Convert to nm for plotting
plt.plot(x_values, detailed_result["F_bulk"].value, 'r-', label='Bulk Energy')
plt.plot(x_values, detailed_result["F_grad"].value, 'g-', label='Gradient Energy')
plt.plot(x_values, detailed_result["F_excess"].value, 'b-', label='Excess Energy')
plt.xlabel('x (nm)')
plt.ylabel('Free Energy Density (J/m³)')
plt.title('Free Energy Components')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"interface_energy_analysis_Kc_{optimal_Kc:.3e}.png")
plt.show()

# Plot showing Kc vs Interface Energy
plt.figure(figsize=(10, 6))
Kc_array = np.array([r["Kc"] for r in results])
energy_array = np.array([r["interface_energy_density"] for r in results])
plt.loglog(Kc_array, energy_array, 'bo-')
plt.axhline(y=F_inte_exp_min, color='r', linestyle='--', label=f'Min Target: {F_inte_exp_min}')
plt.axhline(y=F_inte_exp_max, color='g', linestyle='--', label=f'Max Target: {F_inte_exp_max}')
plt.xlabel('Gradient Energy Coefficient Kc (J/m)')
plt.ylabel('Interface Energy Density (J/m²)')
plt.title('Interface Energy vs Gradient Energy Coefficient')
plt.grid(True)
plt.legend()
plt.savefig("interface_energy_vs_kc.png")
plt.show()

# Plot showing ratio of bulk to gradient energy
plt.figure(figsize=(10, 6))
bulk_array = np.array([r["total_F_bulk"] for r in results])
grad_array = np.array([r["total_F_grad"] for r in results])
ratio_array = bulk_array / grad_array
plt.semilogx(Kc_array, ratio_array, 'ro-')
plt.xlabel('Gradient Energy Coefficient Kc (J/m)')
plt.ylabel('Bulk/Gradient Energy Ratio')
plt.title('Energy Component Ratio vs Gradient Energy Coefficient')
plt.grid(True)
plt.savefig("energy_ratio_vs_kc.png")
plt.show()