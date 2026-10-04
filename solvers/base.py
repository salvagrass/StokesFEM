from problems.base import StokesProblemData
from dataclasses import dataclass
from ufl.coefficient import Coefficient
from abc import ABC,abstractmethod
from enum import Enum
from typing import Any


class MethodType(str, Enum):
    """Enumeration of supported mixed finite element spatial discretizations.

    Inherits from (str, Enum) to ensure seamless serialization, terminal output,
    and dictionary key usage across metrics and plotting modules.

    Attributes:
        TH: Taylor-Hood mixed element family (P_{k} / P_{k-1}).
        MINI: MINI element with cubic bubble enrichment on velocity (P_1^b / P_1).
        KS: Kouhia-Stenberg formulation with asymmetric stress/velocity fields.
        AFW: Arnold-Falk-Winther stable mixed stress-velocity pair.
        RTCG: Raviart-Thomas velocity with discontinuous pressure (RT_k / DG_{k-1}).
        HYBRID: Hybridized mixed formulation with face Lagrange multipliers.
    """

    TH = "Taylor-Hood"
    MINI = "Mini"
    KS = "Kouhia-Stenberg"
    AFW = "Arnold-Falk-Winther"
    RTCG = "Raviart-Thomas"
    HYBRID = "Hybrid"

    def __str__(self) -> str:
        return self.value



class StokesSolver(ABC):
    method_name: MethodType
    """Abstract base class defining the interface for all Stokes flow solvers.

    Subclasses implement specific mixed finite element formulations (e.g.,
    Taylor-Hood, MINI, stress-velocity, hybridized) for Newtonian or
    non-Newtonian rheological regimes.

    Attributes:
        fluid_model (Any): Constitutive rheology model governing fluid behavior
            (e.g., Newtonian constant viscosity, Power-Law, Carreau).
    """

    def __init__(self, fluid_model: Any):
        """Initialize the Stokes solver with an optional constitutive fluid model.

        Args:
            fluid_model (Optional[Any], optional): Rheological model describing
                viscous behavior. Defaults to None.
        """
        self.fluid_model = fluid_model

    @abstractmethod
    def solve(self, data: StokesProblemData,mu: Coefficient) -> "StokesSolution":
        """Solve the Stokes boundary value problem on the given computational domain.

        Args:
            data (StokesProblemData): Container with mesh, boundary markers,
                boundary data mappings, and volumetric body forces.
            mu (Coefficient): Dynamic viscosity coefficient (Constant, Function,
                or Expression).

        Returns:
            StokesSolution: Standardized output container holding extracted
                velocity and pressure fields alongside raw mixed solution data.
        """
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
    raw_solution: Coefficient
    method_name: MethodType