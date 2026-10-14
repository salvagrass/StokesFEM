from problems import StokesProblemData
from commons import MethodType
from solvers.base import StokesSolver,StokesSolution
from typing import Any
from abc import abstractmethod
from fenics import *

class StandardPrimalSolver(StokesSolver):
    """Abstract base class for standard mixed velocity-pressure primal Stokes formulations.

    Implements the template method pattern: subclasses define specific inf-sup
    stable discrete function spaces (_construct_spaces), while this class
    assembles and solves the continuous Galerkin variational problem.
    """
    def __init__(self, fluid_model: Any):
        super().__init__(fluid_model)

    @abstractmethod
    def _construct_spaces(self,mesh: Mesh) -> Any:
        pass

    def solve(self,data: StokesProblemData) -> StokesSolution:
        """Assemble and solve the continuous Galerkin mixed Stokes variational problem.

        Args:
            data (StokesProblemData): Computational domain, boundary markers, and data.

        Returns:
            StokesSolution: Solution container holding velocity, pressure, and the raw state.
        """
        mu = self.fluid_model.mu
        X = self._construct_spaces(data.mesh)
        x = Function(X)
        
        bcs = []
        for marker_id,bc_values in data.dirichlet_bcs.items():
            bcs.append(DirichletBC(X.sub(0),bc_values,data.boundary_markers,marker_id))

        init_value = self.fluid_model.get_init_mu()

        if init_value is not None:
            self._solve_linear_system(data,init_value,bcs,x)
        
        self._solve_linear_system(data,self.fluid_model.mu,bcs,x)

        sol_u, sol_p = x.split(deepcopy=True)
        I = Identity(data.mesh.geometric_dimension())
        sol_T = -sol_p*I + 2*mu(sol_u)*sym(grad(sol_u))
        return StokesSolution(sol_u,sol_p,sol_T,x,self.method_name)


    def _solve_linear_system(self,data: StokesProblemData,mu,bcs,x):
        """Solves the linear system, used for both newtonian and non newtonian fluids"""

        u,p = split(x)
        v,q = TestFunctions(x.function_space())
        lhs = 2*mu(u)*inner(sym(grad(u)),sym(grad(v)))*dx - div(v)*p*dx + div(u)*q*dx
        rhs = dot(data.forcing_term,v)*dx
        
        ds = Measure('ds',subdomain_data=data.boundary_markers,domain=data.mesh)
        
        for marker_id,bc_value in data.neumann_bcs.items():
                rhs += dot(bc_value,v)*ds(marker_id)
        solve(lhs-rhs == 0,x,bcs)
        




class TaylorHoodSolver(StandardPrimalSolver):
    method_name = MethodType.TH
    def __init__(self, fluid_model: Any, degree: int = 2):
        super().__init__(fluid_model)
        self.degree = degree

    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        V = VectorElement("CG", mesh.ufl_cell(), self.degree)
        Q = FiniteElement("CG", mesh.ufl_cell(), self.degree - 1)
        return FunctionSpace(mesh, MixedElement((V,Q)))

class MINISolver(StandardPrimalSolver):
    method_name = MethodType.MINI
    def _construct_spaces(self, mesh):
        P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
        B = FiniteElement('Bubble', mesh.ufl_cell(), mesh.topology().dim() + 1)
        V = VectorElement(NodalEnrichedElement(P1, B))
        Q = FiniteElement('CG',mesh.ufl_cell(),1)
        return FunctionSpace(mesh,MixedElement((V,Q)))

class KSSolver(StandardPrimalSolver):
    method_name = MethodType.KS
    def _construct_spaces(self, mesh: Mesh) -> FunctionSpace:
        V1 = FiniteElement("CR", mesh.ufl_cell(), 1)
        V2 = FiniteElement("CG", mesh.ufl_cell(), 1)
        V = MixedElement([V1, V2])
        Q = FiniteElement("DG", mesh.ufl_cell(), 0)
        
        return FunctionSpace(mesh, MixedElement([V, Q]))

