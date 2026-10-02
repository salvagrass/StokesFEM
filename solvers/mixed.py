from solvers.base import StokesSolution,StokesSolver
from problems.base import StokesProblemData
from abc import abstractmethod
from typing import Any
from fenics import *

class MixedSolver(StokesSolver):
    def __init__(self,fluid_model: Any):
        super().__init__(fluid_model)

    @abstractmethod
    def _construct_spaces(self, mesh: Mesh) -> Any:
        pass

    def solve(self,data:StokesProblemData,mu: Function) -> StokesSolution:

        X = self._construct_spaces(data.mesh)
        
        s1, s2, u, q = TrialFunctions(X)
        t1, t2, v, r = TestFunctions(X)
        
        bcs = []
        for id, bc_value in data.neumann_bcs.items():
            bcs.append(DirichletBC(X.sub(0), bc_value, data.boundary_markers, id))
            bcs.append(DirichletBC(X.sub(1), bc_value, data.boundary_markers, id))
            
        sigma = as_tensor([s1, s2])
        tau = as_tensor([t1, t2])
        eps = as_tensor([[0, 1], [-1, 0]])

        lhs = (
            dot(div(sigma), v) * dx
            + inner(dev(sigma), dev(tau)) / (2 * mu) * dx
            + dot(div(tau), u) * dx
            + inner(sigma, eps) * r * dx
            + inner(tau, eps) * q * dx
        )

        rhs = -dot(data.forcing_term,v)*dx

        ds = Measure("ds", subdomain_data=data.boundary_markers, domain=data.mesh)
        n = FacetNormal(data.mesh)

        for id, bc_value in data.dirichlet_bcs.items():
            rhs += dot(dot(tau, n), bc_value) * ds(id) 

        sol = Function(X)
        solve(lhs == rhs, sol, bcs)
    
        return self._extract_solution(data.mesh,sol)

    @abstractmethod
    def _extract_solution(self,mesh: Mesh, sol: Function) -> StokesSolution:
        pass

class AFWSolver(MixedSolver):
    def __init__(self, fluid_model: Any, degree: int = 1):
        super().__init__(fluid_model)
        self.degree = degree

    def _construct_spaces(self, mesh: Mesh) -> Function:
        S1 = FiniteElement('BDM', mesh.ufl_cell(), self.degree)
        S2 = FiniteElement('BDM', mesh.ufl_cell(), self.degree)
        U = VectorElement('DG', mesh.ufl_cell(), self.degree-1)
        Q = FiniteElement('DG', mesh.ufl_cell(), self.degree-1)

        return FunctionSpace(mesh, MixedElement([S1, S2, U, Q]))

    def _extract_solution(self,mesh: Mesh, sol: Function) -> StokesSolution:
        s1_res, s2_res, u_res, q_res = sol.split(deepcopy=True)
        
        sigma_res = as_tensor([s1_res, s2_res])
        p_expr = -0.5 * tr(sigma_res)
        
        P_space = FunctionSpace(mesh, "DG", self.degree-1)
        p_res = project(p_expr, P_space)
        
        return StokesSolution(u_res,p_res,sol)