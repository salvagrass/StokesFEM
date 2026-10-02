from abc import ABC,abstractmethod
from typing import Any
from problems.base import StokesProblemData
from dataclasses import dataclass
from fenics import Function

class StokesSolver(ABC):
    def __init__(self, fluid_model: Any):
        self.fluid_model = fluid_model

    @abstractmethod
    def solve(self, problem: StokesProblemData,mu: Function) -> Any:
        pass


@dataclass
class StokesSolution:
    u: Function
    p: Function
    raw_solution: Function

