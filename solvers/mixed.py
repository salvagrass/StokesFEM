from abc import abstractmethod
from typing import Any, Optional

from fenics import *
from ufl.coefficient import Coefficient
from ufl.core.expr import Expr

from problems import StokesProblemData
from common import MethodType
from solvers.base import StokesSolution, StokesSolver


class MixedSolver(StokesSolver):
    """Abstract base class for dual mixed stress-velocity Stokes formulations.

    This family of solvers formulates the Stokes system using the stress tensor sigma and
    the velocity field u.
    """

    def __init__(self, fluid_model: Optional[Any] = None) -> None:
        """Initialize the mixed stress-velocity solver.

        Args:
            fluid_model (Optional[Any], optional): Rheological model for non-Newtonian
                or variable-viscosity fluids. Defaults to None.
        """
        super().__init__(fluid_model)

    @abstractmethod
    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        """Construct the mixed finite element function space on the computational mesh.

        Args:
            mesh (Mesh): Computational triangulation of the domain.

        Returns:
            FunctionSpace: Mixed space comprising stress rows, velocity, and rotation multiplier.
        """
        pass

    def solve(self, data: StokesProblemData, mu: Coefficient) -> StokesSolution:
        """Assemble and solve the saddle-point dual mixed Stokes system.

        Args:
            data (StokesProblemData): Problem specifications including geometry,
                boundary facets, boundary tractions/velocities, and source term.
            mu (Coefficient): Dynamic fluid viscosity.

        Returns:
            StokesSolution: Extracted discrete velocity and post-processed pressure
                fields along with the full mixed solution.
        """
        X = self._construct_spaces(data.mesh)

        # Trial and test functions for [stress_row_1, stress_row_2, velocity, asymmetry_multiplier]
        s1, s2, u, q = TrialFunctions(X)
        t1, t2, v, r = TestFunctions(X)

        # Impose essential boundary conditions (Neumann tractions become Dirichlet on H(div) stress)
        bcs = []
        for bnd_id, bc_value in data.mixed_neumann_bcs.items():
            bcs.append(
                DirichletBC(X.sub(0), bc_value[0], data.boundary_markers, bnd_id)
            )
            bcs.append(
                DirichletBC(X.sub(1), bc_value[1], data.boundary_markers, bnd_id)
            )

        sigma = as_tensor([s1, s2])
        tau = as_tensor([t1, t2])

        # Skew-symmetric Levi-Civita alternator tensor in 2D for weak symmetry enforcement
        eps = as_tensor([[0, 1], [-1, 0]])

        # Bilinear form of the dual mixed system with deviatoric compliance and symmetry penalty
        lhs = (
            dot(div(sigma), v) * dx
            + inner(dev(sigma), dev(tau)) / (2 * mu) * dx
            + dot(div(tau), u) * dx
            + inner(sigma, eps) * r * dx
            + inner(tau, eps) * q * dx
        )

        # Linear functional: volumetric body force
        rhs = -dot(data.forcing_term, v) * dx

        # Boundary integration measures and outward unit facet normal
        ds = Measure(
            "ds", subdomain_data=data.boundary_markers, domain=data.mesh
        )
        n = FacetNormal(data.mesh)

        # Natural boundary conditions (prescribed Dirichlet velocities enter through boundary integrals)
        for bnd_id, bc_value in data.dirichlet_bcs.items():
            rhs += dot(dot(tau, n), bc_value) * ds(bnd_id)

        # Solve linear saddle-point algebraic system
        sol = Function(X)
        solve(lhs == rhs, sol, bcs)
        s1_res, s2_res, u_res, _ = sol.split(deepcopy=True)

        # Post-process hydrodynamic pressure from the trace of the stress tensor: p = -1/2 * tr(sigma)
        sigma_res = as_tensor([s1_res, s2_res])
        p_expr = -0.5 * tr(sigma_res)

        p_res = self._extract_pressure(data.mesh, p_expr)
        return StokesSolution(
            u=u_res, p=p_res, T=sigma_res, raw_solution=sol, method_name=self.method_name
        )

    @abstractmethod
    def _extract_pressure(self, mesh: Mesh, p_expr: Expr) -> Function:
        """Project the continuous pressure algebraic expression onto the target finite element space.

        Args:
            mesh (Mesh): Computational domain mesh.
            p_expr (Expr): Algebraic UFL expression for pressure (-0.5 * tr(sigma)).

        Returns:
            Function: Projected discrete pressure field.
        """
        pass


class AFWSolver(MixedSolver):
    """Arnold-Falk-Winther (AFW) mixed stress-velocity solver.

    Discretizes stress rows with Brezzi-Douglas-Marini (BDM_k) elements, velocity
    with discontinuous polynomials (DG_{k-1}), and the asymmetry multiplier with DG_{k-1}.
    """

    method_name = MethodType.AFW

    def __init__(self, fluid_model: Optional[Any] = None, degree: int = 1) -> None:
        if degree < 1:
            raise ValueError("Degree in AFW can't be less than one")
        super().__init__(fluid_model)
        self.degree = degree

    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        cell = mesh.ufl_cell()
        S1 = FiniteElement("BDM", cell, self.degree)
        S2 = FiniteElement("BDM", cell, self.degree)
        U = VectorElement("DG", cell, self.degree - 1)
        Q = FiniteElement("DG", cell, self.degree - 1)

        return FunctionSpace(mesh, MixedElement([S1, S2, U, Q]))

    def _extract_pressure(self, mesh: Mesh, p_expr: Expr) -> Function:
        p_space = FunctionSpace(mesh, "DG", self.degree - 1)
        return project(p_expr, p_space)


class RTCGSolver(MixedSolver):
    """Raviart-Thomas / Continuous Galerkin (RT-CG) mixed solver.

    Discretizes stress rows with Raviart-Thomas (RT_k) elements, velocity with
    discontinuous polynomials (DG_{k-1}), and the asymmetry multiplier with continuous
    Lagrange elements (CG_{k-1}).
    """

    method_name = MethodType.RTCG

    def __init__(self, fluid_model: Optional[Any] = None, degree: int = 2) -> None:
        if degree < 2:
            raise ValueError("Degree in RT-CG can't be less than two")
        super().__init__(fluid_model)
        self.degree = degree

    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        cell = mesh.ufl_cell()
        S1 = FiniteElement("RT", cell, self.degree)
        S2 = FiniteElement("RT", cell, self.degree)
        U = VectorElement("DG", cell, self.degree - 1)
        Q = FiniteElement("CG", cell, self.degree - 1)

        return FunctionSpace(mesh, MixedElement([S1, S2, U, Q]))

    def _extract_pressure(self, mesh: Mesh, p_expr: Expr) -> Function:
        p_space = FunctionSpace(mesh, "DG", self.degree)
        return project(p_expr, p_space)


class HybridSolver(MixedSolver):
    """Hybridized mixed formulation combining RT_1 and BDM_1 stress components.

    Discretizes stress rows with an asymmetric RT_1 / BDM_1 coupling, piecewise constant
    velocity (DG_0), and affine continuous rotation Lagrange multipliers (CG_1).
    """

    method_name = MethodType.HYBRID

    def __init__(self, fluid_model: Optional[Any] = None) -> None:
        super().__init__(fluid_model)

    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        cell = mesh.ufl_cell()
        S1 = FiniteElement("RT", cell, 1)
        S2 = FiniteElement("BDM", cell, 1)
        U = VectorElement("DG", cell, 0)
        Q = FiniteElement("CG", cell, 1)

        return FunctionSpace(mesh, MixedElement([S1, S2, U, Q]))

    def _extract_pressure(self, mesh: Mesh, p_expr: Expr) -> Function:
        p_space = FunctionSpace(mesh, "DG", 1)
        return project(p_expr, p_space)