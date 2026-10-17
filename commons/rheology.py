from abc import ABC,abstractmethod

from fenics import Constant,inner,sym,grad,DOLFIN_EPS
from ufl.core.expr import Expr


class FluidModel(ABC):
    """Abstract base class representing a fluid rheology model.

    Defines the interface for computing apparent viscosity as a function of
    strain rate or stress tensor.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def mu(self,u: Expr) -> Expr:
        """Computes the apparent dynamic viscosity given the strain rate tensor.

        Args:
            D: Symmetric strain-rate tensor.

        Returns:
            Viscosity expression as a UFL `Expr`.
        """
        pass

    def mu_tau(self,tau: Expr) -> Expr:
        """Computes the inverse viscosity given the stress tensor.
        Used primarily in mixed formulations.

        Args:
            tau: Deviatoric stress tensor.

        Returns:
            Viscosity expression as a UFL `Expr`.

        Raises:
            NotImplementedError: If the rheological model does not support
                or implement an analytical inverse.
        """
        raise NotImplementedError(
            "This rheology can't be inverted to be used in mixed/augumented formulations"
        )

class LinearModel(FluidModel):
    """Newtonian (constant viscosity) fluid model.

    Attributes:
        mu_val: Constant dynamic viscosity.
    """
    def __init__(self, mu: float):
        """Initializes the linear fluid model with a constant viscosity.

        Args:
            mu: Dynamic viscosity value.
        """
        self.mu = Constant(mu)

    def mu(self,u: Expr = None) -> Expr:
        """Returns the constant dynamic viscosity.

        Args:
            D: Optional strain-rate tensor (ignored since constant).

        Returns:
            Constant viscosity expression.
        """
        return self.mu

    def mu_tau(self, tau: Expr = None) -> Expr:
        """Returns the constant dynamic viscosity.

        Args:
            D: Optional strain-rate tensor (ignored since constant).

        Returns:
            Constant viscosity expression.
        """
        return self.mu

class PowerLawModel(FluidModel):
    """Ostwald-de Waele Power-Law non-Newtonian fluid model.

    Formula: μ = K · (2 D : D)^((r - 2) / 2)

    Attributes:
        K: multiplicative constant.
        r: Flow behavior parameter (2 for newtonian fluids, <2 shear thickening, >2 shear thinning).
    """
    def __init__(self,
                 K: float = 0.5,
                 r: float = 2.0,
                 ):
        self.K = Constant(K)
        self.r = Constant(r)

    def mu(self,D: Expr) -> Expr:
        """Computes the power-law viscosity from the strain-rate tensor.

        Args:
            D: Symmetric strain-rate tensor D(u) = sym(∇u).

        Returns:
            viscosity of the fluid as a UFL Expr
        """
        strain_rate = 2.0*inner(D,D) + DOLFIN_EPS
        return self.K*pow(strain_rate,(self.r-2.0)/2.0)

    def mu_tau(self, tau):
        """Computes the inverted viscosity from the deviatoric stress tensor.

        Args:
            tau: Deviatoric stress tensor τ.

        Returns:
            Inverted viscosity expression.
        """
        tau_sq = 0.5*inner(tau,tau) + DOLFIN_EPS
        return pow(self.K,1/(self.r-1))*pow(tau_sq,(self.r-2.0)/(2.0*(self.r-1)))



class CarreauModel(FluidModel):
    """Carreau non-Newtonian fluid model.

    Formula: μ = μ_∞ + (μ_0 - μ_∞) · (1 + K² · (2 · D : D))^((r - 2) / 2)

    Attributes:
        mu_0: Zero-shear-rate viscosity plateau μ_0 as a FEniCS Constant.
        mu_inf: Infinite-shear-rate viscosity plateau μ_∞ as a FEniCS Constant.
        K: Relaxation time / characteristic time parameter K as a FEniCS Constant.
        r: Power-law index parameter r as a FEniCS Constant.
    """
    def __init__(self,
                 mu_0: float = 0.5,
                 mu_inf: float = 0.01,
                 K: float = 1.0,
                 r: float = 2.0
                 ):
        """Initializes the CarreauModel.

        Args:
            mu_0: Zero-shear viscosity plateau μ_0. Defaults to 0.5.
            mu_inf: Infinite-shear viscosity plateau μ_∞. Defaults to 0.01.
            K: Relaxation time parameter. Defaults to 1.0.
            r: Power-law index parameter. Defaults to 2.0.
        """
        self.mu_0 = Constant(mu_0)
        self.mu_inf = Constant(mu_inf)
        self.K = Constant(K)
        self.r = Constant(r)


    def mu(self,D: Expr) -> Expr:
        """Computes Carreau apparent viscosity from the strain-rate tensor.

        Args:
            D: Symmetric strain-rate tensor D(u) = sym(∇u).

        Returns:
            Viscosity of the fluid as a UFL Expression.
        """
        strain_rate = 2.0*inner(D,D)
        return self.mu_inf + (self.mu_0 - self.mu_inf)*pow((1+self.K**2*strain_rate),(self.r-2.0)/2.0)