# phase-field-simulation
Phase-field simulations of solidification and spinodal decomposition using FiPy. Includes 1D/2D models, bulk and gradient energy coefficient determination, and visualization. Based on the Foundations of Materials Simulation course at FAU.


---

## 📑 Contents

- `Task_1.py` – 1D single-component solidification 
- `Task_2.py` – Bulk energy coefficient determination
- `1D.py`, `2D.py` – Cahn-Hilliard equation simulations
- `Phase_field_method_Report.pdf` – Detailed report with derivations, results & figures
- `PF_practical.pdf` – Original project sheet
- `spinodal_pattern_t*.png` – 2D concentration profiles at various timesteps

---

## 🚀 Requirements

- Python 3.8+
- [FiPy](https://www.ctcms.nist.gov/fipy/), NumPy, Matplotlib

```bash
pip install fipy numpy matplotlib
