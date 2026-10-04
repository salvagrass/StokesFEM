# StokesFEM

**A modular, research-oriented Finite Element framework for incompressible Stokes flow, built on FEniCS.**

This repository provides a mathematically rigorous computational environment designed to analyze, discretize, and solve steady incompressible Stokes flow problems. Built with high software standards (SOLID principles, Factory patterns, zero non-essential dependencies), the framework enables systematic comparisons across different finite element spaces, automated $h$-refinement studies with online Experimental Order of Convergence (EOC) tracking, and reproducible post-processing.

---

## Mathematical Formulations & Discretizations

The incompressible Stokes problem governs creeping flow for viscous fluids:

$$
\begin{aligned}
-\nabla \cdot \boldsymbol{\sigma}(\boldsymbol{u}, p) &= \boldsymbol{f} \quad \text{in } \Omega, \\
\nabla \cdot \boldsymbol{u} &= 0 \quad \text{in } \Omega,
\end{aligned}
$$

where $\boldsymbol{u}$ is the fluid velocity, $p$ is the pressure, and $\boldsymbol{\sigma} = 2\mu \boldsymbol{\varepsilon}(\boldsymbol{u}) - p\boldsymbol{I}$ is the Cauchy stress tensor.

Because the system is an operator saddle-point problem, compatibility between discrete velocity and pressure spaces is governed by the **Ladyzhenskaya-Babuška-Brezzi (LBB / inf-sup)** condition. The framework natively implements and benchmarks multiple discretization strategies:

* **Primal Formulations:**
  * Taylor-Hood ($\mathbb{P}_k / \mathbb{P}_{k-1}$)
  * MINI Element 
  * Kouhia-Stenberg (KS)

* **Mixed Formulations:**
  * Arnold-Falk-Winther (AFW)
  * Raviart-Thomas / Continuous Galerkin (RT-CG)
  * Hybridized Mixed Methods

* **Augmented Formulations**

---
## Project Architecture & Repository Structure

The framework is built with a strong emphasis on modern Python software engineering principles (SOLID). Mathematical logic, numerical resolution, metrics calculation, and visualization are strictly decoupled.

```text
StokesFEM/
├── problems/                    # Domain geometries, boundary conditions, and exact solutions
│   ├── base.py                  # Base data structures (e.g., StokesProblemData)
│   ├── analytical.py            # Method of Manufactured Solutions (MMS) definitions
│   ├── cavity.py                # Lid-driven cavity benchmark setups
│   └── obstacle.py              # Flow past an obstacle scenarios
│
├── solvers/                     # FEM Solver implementations
│   ├── base.py                  # Abstract interfaces and StokesSolution definitions
│   ├── factory.py               # SolverFactory for dynamic solver instantiation
│   ├── primal.py                # Velocity-pressure formulations (Taylor-Hood, MINI, KS)
│   ├── mixed.py                 # Dual/tensor formulations (AFW, RT-CG, Hybrid)
│   └── augmented.py             # Stabilized equal-order formulations
│
├── postprocessing/              # Analysis, metrics, and visualization
│   ├── metrics.py               # ConvergenceStudy: O(1) online EOC calculation
│   ├── plots.py                 # General field and domain plotting utilities
│   └── convergence_plotter.py   # Object-oriented convergence visualization
│
├── benchmarks/                  # Reproducible scientific scripts (Use cases)
│   ├── run_analytical.py        # Automated h-refinement loops and EOC validation
│   ├── run_cavity.py            # Lid-driven cavity simulations
│   └── run_obstacle.py          # Physical flow past a cylinder simulations
│
└── results/                     # Output directory for CSVs, LaTeX tables, and plots
```

---
## Usage Examples

### 1. Flow Past an Obstacle: Swapping Formulations
```python
# 1. Generate the physical domain and boundaries
problem_data = generate_problem(n=30)
viscosity = Constant(2.0)

# 2. Seamlessly swap between formulations using the Factory
methods = ["th", "mini", "ks", "afw", "rt-cg", "hybrid"]

for method in methods:
    # Instantiate the specific solver dynamically
    solver = SolverFactory.create_solver(method, fluid_model=None, degree=2)
    
    # Solve the saddle-point system
    solution = solver.solve(problem_data, viscosity)
    
    # Visualize velocity streamlines and pressure fields
    fig, _ = plot_combined(solution, mode='streamlines', title_prefix=f'{method.upper()} - ')
    fig.savefig(f"obstacle_{method}.png")
```
### 2 Automated Convergence Study
```python
plotter = ConvergencePlotter(mode='h', prefix="Analytical MMS", normalized=True)
methods = ["th", "mini", "ks"]

for method in methods:
    solver = SolverFactory.create_solver(method, degree=2)
    study = ConvergenceStudy(method_name=method)
    
    # h-refinement loop
    for n in [8, 16, 32, 64]:
        h = 1.0 / n
        problem_data = generate_problem(n=n)
        
        solution = solver.solve(problem_data, Constant(2.0))
        
        # Compute L2/H1 velocity and L2 pressure errors
        errors = compute_errors(problem_data, solution, degree_rise=3)
        
        # O(1) online EOC tracking
        study.add_error_data(errors, mesh_h=h)
        
    # Print terminal-friendly tables (or export to LaTeX/CSV)
    study.print_table()
    
    # Automatically feed the study data to the plotter
    plotter.plot_from_study(study)

# Add theoretical reference slopes to the plots
plotter.add_reference_slope('u_l2', order=2)
plotter.add_reference_slope('u_l2', order=3)

plotter.fig.savefig("convergence_rates.png")
```
---

### To be implemented

- [ ] **Discontinuous Galerkin (DG) Methods:** Interior Penalty DG (IPDG) and hybridizable DG (HDG) discretizations for arbitrary polygonal meshes and strict locally mass-conserving velocity fields.
- [ ] **Non-Linear Fluid Rheology:** Generalization from Newtonian fluids ($\mu = \text{const}$) to generalized non-Newtonian shear-thinning and viscoplastic models:
  - Power-Law (Ostwald-de Waele)
  - Carreau-Yasuda
  - Bingham / Herschel-Bulkley (via augmented regularized formulations)
--- 
