from abc import ABC,abstractmethod
from typing import Any

from fenics import Constant,inner,sym,grad,DOLFIN_EPS
from ufl.core.expr import Expr


class FluidModel(ABC):    
    @abstractmethod
    def get_init_mu(self):
        pass

    @abstractmethod
    def mu(self,u: Expr):
        pass

class LinearModel(FluidModel):
    def __init__(self, mu: float):
        self.mu = Constant(mu)
        super().__init__()

    def mu(self,u: Expr = None):
        return self.mu
    
    def get_init_mu(self):
        return None

class PowerLawModel(FluidModel):
    def __init__(self,
                 K: float = 0.5,
                 n: float = 3,
                 init_mu: float = 0.2
                 ):
        self.mu_0 = Constant(K)
        self.r = Constant(n)
        self.init_mu = Constant(init_mu)

    def mu(self,u: Expr):
        strain_rate = inner(sym(grad(u)),sym(grad(u))) + DOLFIN_EPS
        return self.mu_0*pow(strain_rate,(self.r-2.0)/2.0)

    def get_init_mu(self):
        return lambda u: self.init_mu