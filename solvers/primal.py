from problems import StokesProblemData
from commons import MethodType,FluidModel
from solvers.base import StokesSolver,StokesSolution
from solvers.state import State
from solvers.strategies import SolverStrategy
from typing import Any
from abc import abstractmethod
from fenics import *

class StandardPrimalSolver(StokesSolver):
    """Abstract base class for standard mixed velocity-pressure primal Stokes formulations.

    Implements the template method pattern: subclasses define specific inf-sup
    stable discrete function spaces (_construct_spaces), while this class
    assembles and solves the continuous Galerkin variational problem.
    """
    def __init__(self, fluid_model: FluidModel):
        super().__init__(fluid_model)

    @abstractmethod
    def _construct_spaces(self,mesh: Mesh) -> Any:
        pass

    def solve(self,data: StokesProblemData,strategy: SolverStrategy = None) -> StokesSolution:
        """Assemble and solve the continuous Galerkin mixed Stokes variational problem.

        Args:
            data (StokesProblemData): Computational domain, boundary markers, and data.

        Returns:
            StokesSolution: Solution container holding velocity, pressure, and the raw state.
        """
        mu = self.fluid_model.mu
        self.data = data
        X = self._construct_spaces(data.mesh)
        x = Function(X)

        bcs = []
        for marker_id,bc_values in data.dirichlet_bcs.items():
            bcs.append(DirichletBC(X.sub(0),bc_values,data.boundary_markers,marker_id))

        strategy.solve_system(self.build_system,PrimalState(x,mu),bcs)

        sol_u, sol_p = x.split(deepcopy=True)
        I = Identity(data.mesh.geometric_dimension())
        sol_T = -sol_p*I + 2*mu(sym(grad(sol_u)))*sym(grad(sol_u))
        return StokesSolution(sol_u,sol_p,sol_T,x,self.method_name)


    def build_system(self, x: Function,mu_func,linear: bool = False):
        
        X = x.function_space()
        u,p = TrialFunctions(X) if linear else split(x)
        v,q = TestFunctions(X)


        lhs = 2*mu_func(sym(grad(u)))*inner(sym(grad(u)),sym(grad(v)))*dx - div(v)*p*dx + div(u)*q*dx
        rhs = dot(self.data.forcing_term,v)*dx
                
        ds = Measure('ds',subdomain_data=self.data.boundary_markers,domain=self.data.mesh)
        for marker_id,bc_value in self.data.neumann_bcs.items():
            rhs += dot(bc_value,v)*ds(marker_id)
        return (lhs,rhs)

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



class PrimalState(State):
    def __init__(self, x, mu_func):
        super().__init__(x, mu_func)

    def compute_viscosity(self):
        u = split(self.x)[0]
        return self.mu_func(sym(grad(u)))
        