from problems import StokesProblemData
from commons import MethodType,FluidModel
from solvers.strategies import SolverStrategy
from dataclasses import dataclass
from fenics import Function
from ufl.coefficient import Coefficient
from ufl.core.expr import Expr
from abc import ABC,abstractmethod
from typing import Callable,Tuple


class StokesSolver(ABC):
    method_name: MethodType
    """Abstract base class defining the interface for all Stokes flow solvers.

    Subclasses implement specific mixed finite element formulations (e.g.,
    Taylor-Hood, MINI, stress-velocity, hybridized) for Newtonian or
    non-Newtonian rheological regimes.

    Attributes:
        fluid_model (FluidModel): Constitutive rheology model governing fluid behavior
            (e.g., Newtonian constant viscosity, Power-Law, Carreau).
    """

    def __init__(self, fluid_model: FluidModel):
        """Initialize the Stokes solver with an optional constitutive fluid model.

        Args:
            fluid_model: FluidModel: Rheological model describing
                viscous behavior.
        """
        self.fluid_model = fluid_model

    @abstractmethod
    def solve(self, data: StokesProblemData,strategy: SolverStrategy) -> "StokesSolution":
        """Solve the Stokes boundary value problem on the given computational domain.

        Args:
            data (StokesProblemData): Container with mesh, boundary markers,
                boundary data mappings, and volumetric body forces.
            strategy (SolverStrategy): the strategy used to solve the system
                (e.g, LinearStrategy,NewtonStrategy,PicardStrategy)

        Returns:
            StokesSolution: Standardized output container holding extracted
                velocity and pressure fields alongside raw mixed solution data.
        """
        pass

    @abstractmethod
    def build_system(self, x: Function,mu_func: Callable[[Expr],Expr] , linear: bool = False) -> Tuple[Expr,Expr]:
        pass
    
@dataclass
class StokesSolution:
    """Standardized solution container holding discrete fields from a Stokes solver.

    Attributes:
        u (Coefficient): Discrete velocity field extracted from the mixed system.
        p (Coefficient): Discrete pressure field extracted from the mixed system.
        raw_solution (Coefficient): Full discrete solution vector across the combined
            mixed function space before component sub-function extraction.
        method_name (MethodType): Discretization scheme utilized to produce the solution.
    """
    u: Coefficient
    p: Coefficient
    T: Coefficient
    raw_solution: Expr
    method_name: MethodType