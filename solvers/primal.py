from solvers.base import StokesSolver,StokesSolution
from problems.base import StokesProblemData
from typing import Any
from abc import abstractmethod
from fenics import *

class StandardPrimalSolver(StokesSolver):
    def __init__(self, fluid_model: Any):
        super().__init__(fluid_model)

    @abstractmethod
    def _construct_spaces(self,mesh: Mesh) -> Any:
        pass

    def solve(self,data: StokesProblemData,mu: Function) -> StokesSolution:
        X = self._construct_spaces(data.mesh)
        bcs = []
        for id,bc_values in data.dirichlet_bcs.items():
            bcs.append(DirichletBC(X.sub(0),bc_values,data.boundary_markers,id))
        
        u,p = TrialFunctions(X)
        v,q = TestFunctions(X)
            
        sol = Function(X)
        
        lhs = mu*(inner(grad(u),grad(v))*dx) - div(v)*p*dx + div(u)*q*dx
        rhs = dot(data.forcing_term,v)*dx
        
        ds = Measure('ds',subdomain_data=data.boundary_markers,domain=data.mesh)
        
        for id,bc_value in data.neumann_bcs.items():
                rhs += dot(bc_value,v)*ds(id)
        solve(lhs==rhs,sol,bcs)
        sol_u, sol_p = sol.split(deepcopy=True)
        return StokesSolution(sol_u,sol_p,sol)


class TaylorHoodSolver(StandardPrimalSolver):
    def __init__(self, fluid_model: Any, degree: int = 2):
        super().__init__(fluid_model)
        self.degree = degree

    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        V = VectorElement("CG", mesh.ufl_cell(), self.degree)
        Q = FiniteElement("CG", mesh.ufl_cell(), self.degree - 1)
        return FunctionSpace(mesh, MixedElement((V,Q)))

class MINISolver(StandardPrimalSolver):
    def _construct_spaces(self, mesh):
        P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
        B = FiniteElement('Bubble', mesh.ufl_cell(), mesh.topology().dim() + 1)
        V = VectorElement(NodalEnrichedElement(P1, B))
        Q = FiniteElement('CG',mesh.ufl_cell(),1)
        return FunctionSpace(mesh,MixedElement((V,Q)))

class KSSolver(StandardPrimalSolver):
    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        V1 = FiniteElement("CR", mesh.ufl_cell(), 1)
        V2 = FiniteElement("CG", mesh.ufl_cell(), 1)
        V = MixedElement([V1, V2])
        Q = FiniteElement("DG", mesh.ufl_cell(), 0)
        
        return FunctionSpace(mesh, MixedElement([V, Q]))

class SIPGSolver(StokesSolver):
    def __init__(self, fluid_model,degree: int = 1, alpha_coeff : float = 6.1, beta_coeff: float = 2.1 ):
        super().__init__(fluid_model)
        self.degree = degree
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff
    
    def solve(self, data: StokesProblemData,mu: Function):
        V = VectorElement("DG",data.mesh.ufl_cell(),self.degree)
        Q = VectorElement("DG",data.mesh.ufl_cell(),self.degree)
        X = FunctionSpace(data.mesh,MixedElement((V,Q)))

        u,p = TrialFunctions(X)
        v,q = TestFunctions(X)
        n = FacetNormal(data.mesh)
        h = CellDiameter(data.mesh)
        h_avg = (h('+') + h('-'))/2

        u_terms = (
            (mu*inner(grad(u),grad(v)))*dx 
            - (mu*inner(avg(grad(u)),jump(v,n)))*dS 
            - (mu*inner(avg(grad(v)),jump(u,n)))*dS 
            + (mu*alpha/h_avg*inner(jump(u,n),jump(v,n)))*dS
            # TODO
            )
        pass
        

