# Stokes Flow Discretization Suite in FEniCS

A modular, extensible Finite Element framework developed in legacy FEniCS (2019.1.0) to simulate and benchmark various discretizations for 2D incompressible Stokes flow problems.

The suite supports both standard **Primal formulations** and dual **Mixed formulations** with weak stress symmetry

---

## Supported Discretizations

### Primal Formulations (Velocity–Pressure)
* **Taylor-Hood (`th`)**: Stable $\boldsymbol{P}_k / P_{k-1}$ mixed formulation (default $k=2$).
* **MINI Element (`mini`)**: Linear continuous velocity enriched with bubble functions and linear pressure ($\boldsymbol{P}_1\text{b} / P_1$).
* **Kouhia-Stenberg (`ks`)**: Reduced-order formulation decoupling velocity components into continuous and discontinuous components.

### Mixed Formulations (Stress–Velocity Dual with Weak Symmetry)
Dual formulation solving for the row-wise Cauchy stress tensor ($\boldsymbol{\sigma}$), velocity ($\boldsymbol{u}$), and an asymmetry multiplier ($q$), with a-posteriori pressure recovery:
* **AFW (`afw`)**: Arnold-Falk-Winther method using $\text{BDM}_k \times [\text{DG}_{k-1}]^2 \times \text{DG}_{k-1}$.
* **RT-CG (`rt-cg`)**: Raviart-Thomas stress fields with continuous asymmetry multiplier.
* **Hybrid (`hybrid`)**: Hybrid stress approximation ($\text{RT}_1 \times \text{BDM}_1$) coupled with discontinuous velocity and continuous multiplier.

---

## Directory Structure

```text
.
├── problems/
│   ├── base.py           # StokesProblemData container
│   └── obstacle.py       # Flow past an obstacle benchmark generator
├── solvers/
│   ├── base.py           # StokesSolver base class & StokesSolution dataclass
│   ├── factory.py        # SolverFactory registry
│   ├── mixed.py          # Dual mixed solvers (AFW, RT-CG, Hybrid)
│   └── primal.py         # Primal solvers (Taylor-Hood, MINI, KS)
├── test_obstacle.py      # Automated benchmark across registered solvers
├── .gitignore
└── README.md