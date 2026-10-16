from abc import ABC,abstractmethod
from ufl.core.expr import Expr
from fenics import Function
from typing import Callable


class State(ABC):
    def __init__(self, x: Function, mu_func: Callable[[Expr],Expr]):
        self.x = x
        self.mu_func = mu_func

    @abstractmethod
    def compute_viscosity(self) -> Expr:
        pass