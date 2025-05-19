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

# Keeping the original work safe. Email atharvasinnarkar@gmail.com for the full code and mention the proper usecase.
