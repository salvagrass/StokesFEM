from problems import StokesProblemData
from commons import MethodType
from solvers.base import StokesSolution,StokesSolver
from solvers.state import State
from solvers.strategies import SolverStrategy

from fenics import *
from ufl.core.expr import Expr

from abc import abstractmethod
from typing import Tuple

class AugmentedSolver(StokesSolver):
    """Base class for augmented mixed Stokes solvers.

    Solves for stress T, auxiliary gradient R, and velocity u.
    """

    def __init__(self, fluid_model):
        """Initializes the augmented solver.

        Args:
            fluid_model: Rheology model defining viscosity μ.
        """
        super().__init__(fluid_model)

    def solve(self,data: StokesProblemData, strategy: SolverStrategy) -> StokesSolution:
        """Solves the augmented Stokes problem.

        Args:
            data: Problem setup with mesh, markers, and boundary data.
            strategy: Solver strategy (e.g., Picard, Newton, or linear).

        Returns:
            StokesSolution containing computed u, p, T.
        """
        self.data = data

        mu = self.fluid_model.mu
        X = self._construct_spaces(data.mesh)
        x = Function(X)

        # Setting up the Neumann Boundary
        bcs = []
        for bnd_id, bc_value in data.mixed_neumann_bcs.items():
            bcs.append(
                DirichletBC(X.sub(0), bc_value[0], data.boundary_markers, bnd_id)
            )
            bcs.append(
                DirichletBC(X.sub(1), bc_value[1], data.boundary_markers, bnd_id)
            )

        
        # Solving the system and splitting the data:
        strategy.solve_system(self.build_system,AugmentedState(x,mu),bcs)

        t1_sol,t2_sol,_,_,u_sol = x.split(deepcopy=True)
        T_sol = as_tensor([t1_sol,t2_sol])

        p_expr = -0.5*tr(T_sol)
        p_space = FunctionSpace(data.mesh,"DG",self.degree-1)
        p_sol = project(p_expr,p_space)

        return StokesSolution(u_sol,p_sol,T_sol,x,self.method_name)


    def build_system(self, x:Function,mu,linear: bool = False) -> Tuple[Expr,Expr]:
        """Assembles variational forms (lhs, rhs) for the augmented system.

        Args:
            x: Current solution state vector.
            mu: Apparent viscosity function μ(D).
            linear: If True, uses TrialFunctions instead of split(x).

        Returns:
            Tuple of UFL forms (lhs, rhs).
        """

        X = x.function_space()
        # Setting up our elements
        t1,t2,r1,r2,u = TrialFunctions(X) if linear else split(x)
        s1,s2,e1,e2,v =TestFunctions(X)
        
        T = as_tensor([t1,t2])
        R = as_tensor([r1,r2])

        sigma = as_tensor([s1,s2])
        eta = as_tensor([e1,e2])

        # Constructing the formulation
        lhs = (
            -2*mu(sym(R))*inner(sym(R),sym(eta))*dx
            + inner(dev(T),dev(eta))* dx
            + inner(dev(sigma),dev(R)) * dx
            + dot(div(sigma),u) * dx
            + dot(div(T),v) * dx
        )

        rhs = -dot(self.data.forcing_term,v)*dx

        # Adding boundary terms
        ds = Measure(
            "ds", subdomain_data=self.data.boundary_markers, domain=self.data.mesh
        )
        n = FacetNormal(self.data.mesh)

        for bnd_id, bc_value in self.data.dirichlet_bcs.items():
            rhs += dot(dot(sigma, n), bc_value) * ds(bnd_id)
        
        return (lhs,rhs)
        

    @abstractmethod
    def _construct_spaces(self,mesh: Mesh):
        pass


class AUG_BDM_Solver(AugmentedSolver):
    """Augmented solver using Brezzi-Douglas-Marini (BDM) elements."""

    method_name = MethodType.AUG_BDM
    def __init__(self, fluid_model, degree: int = 1):
        """Initializes the BDM augmented solver.

        Args:
            fluid_model: Fluid rheology model.
            degree: Polynomial degree (must be >= 1).
        """
        if(degree < 1):
            raise ValueError("Degree too low for the method")
        super().__init__(fluid_model)
        self.degree = degree

    def _construct_spaces(self,mesh):
        cell = mesh.ufl_cell()
        S1 = FiniteElement("BDM",cell,degree=self.degree)
        S2 = FiniteElement("BDM",cell,degree=self.degree)
                
        #! Doesn't work with DG elements with degree - 1
        N1 = FiniteElement("BDM",cell,degree=self.degree)
        N2 = FiniteElement("BDM",cell,degree=self.degree)
                
        U = VectorElement("DG",cell, self.degree-1)

        # Constructing global space
        return FunctionSpace(mesh,MixedElement([S1,S2,N1,N2,U]))

class AUG_RT_Solver(AugmentedSolver):
    """Augmented solver using Raviart-Thomas (RT) elements."""
    
    method_name = MethodType.AUG_RT
    def __init__(self, fluid_model, degree: int = 1):
        """Initializes the BDM augmented solver.

        Args:
            fluid_model: Fluid rheology model.
            degree: Polynomial degree (must be >= 1).
        """
        if(degree < 1):
            raise ValueError("Degree too low for the method")
        super().__init__(fluid_model)
        self.degree = degree

    def _construct_spaces(self,mesh):
        cell = mesh.ufl_cell()
        S1 = FiniteElement("RT",cell,degree=self.degree)
        S2 = FiniteElement("RT",cell,degree=self.degree)
                
        #! Doesn't work with DG elements with degree - 1
        N1 = FiniteElement("RT",cell,degree=self.degree)
        N2 = FiniteElement("RT",cell,degree=self.degree)
                
        U = VectorElement("DG",cell, self.degree-1)

        # Constructing global space
        return FunctionSpace(mesh,MixedElement([S1,S2,N1,N2,U]))


class AugmentedState(State):
    def __init__(self, x, mu_func):
        super().__init__(x, mu_func)

    def compute_viscosity(self):
        _,_,r1,r2,_ = split(self.x)
        R = as_tensor([r1,r2])

        return self.mu_func(R)