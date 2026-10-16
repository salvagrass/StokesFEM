from abc import ABC,abstractmethod
from typing import Any

from fenics import Constant,inner,sym,grad,DOLFIN_EPS
from ufl.core.expr import Expr


class FluidModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def mu(self,u: Expr):
        pass

    def mu_tau(self,tau: Expr):
        raise NotImplementedError(
            "This rheology can't be inverted to be used in mixed/augumented formulations"
        )

class LinearModel(FluidModel):
    def __init__(self, mu: float):
        self.mu = Constant(mu)

    def mu(self,u: Expr = None):
        return self.mu

    def mu_tau(self, tau: Expr = None):
        return self.mu

class PowerLawModel(FluidModel):
    def __init__(self,
                 K: float = 0.5,
                 r: float = 2.0,
                 ):
        self.K = Constant(K)
        self.r = Constant(r)

    def mu(self,D: Expr):
        strain_rate = 2.0*inner(D,D) + DOLFIN_EPS
        return self.K*pow(strain_rate,(self.r-2.0)/2.0)

    def mu_tau(self, tau):
        tau_sq = 0.5*inner(tau,tau) + DOLFIN_EPS
        return pow(self.K,1/(self.r-1))*pow(tau_sq,(self.r-2.0)/(2.0*(self.r-1)))



class CarreauModel(FluidModel):
    def __init__(self,
                 mu_0: float = 0.5,
                 mu_inf: float = 0.01,
                 K: float = 1.0,
                 r: float = 2.0
                 ):
        self.mu_0 = Constant(mu_0)
        self.mu_inf = Constant(mu_inf)
        self.K = Constant(K)
        self.r = Constant(r)


    def mu(self,D: Expr):
        strain_rate = 2.0*inner(D,D)
        return self.mu_inf + (self.mu_0 - self.mu_inf)*pow((1+self.K**2*strain_rate),(self.r-2.0)/2.0)