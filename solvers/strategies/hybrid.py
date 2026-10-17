from .base import SolverStrategy
from .picard import PicardStrategy
from .newton import NewtonStrategy


class HybridStrategy(SolverStrategy):
    """Combines initial Picard iterations with subsequent Newton-Raphson steps.

    Uses a few Picard fixed-point steps to move the solution into the
    basin of attraction, then switches to Newton-Raphson for quadratic convergence.
    """
    def __init__(self,
                 mu_guess: float = 1.0,
                 quad_points: int = 6,
                 relaxation: float = 1.0,
                 num_iter_p: int = 10,
                 num_iter_n: int = 50,
                 abs_tol: float = 1e-10,
                 rel_tol: float = 1e-9
                 ):
        """Initializes the hybrid Picard-Newton solver strategy.

        Args:
            mu_guess: Initial viscosity guess for Picard iterations.
            quad_points: Quadrature degree for numerical integration.
            relaxation: Under-relaxation parameter for Newton iterations.
            num_iter_p: Maximum iterations for the Picard preconditioning stage.
            num_iter_n: Maximum iterations for the Newton-Raphson stage.
            abs_tol: Absolute residual tolerance for Newton convergence.
            rel_tol: Relative residual tolerance for Newton convergence.
        """
        self.mu_guess = float(mu_guess)
        self.quad_points = int(quad_points)
        self.relaxation = float(relaxation)
        self.num_iter_p = int(num_iter_p)
        self.num_iter_n = int(num_iter_n)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)

        self.picard = PicardStrategy(self.mu_guess,self.quad_points,self.num_iter_p,raise_error=False)
        self.newton = NewtonStrategy(quad_points=self.quad_points,
                                     relaxation=self.relaxation,
                                     max_iteration=self.num_iter_n,
                                     abs_tol=self.abs_tol,
                                     rel_tol=self.rel_tol,
                                     not_guess=True)

    def solve_system(self, build_system, state, bcs) -> None:
        """Executes Picard pre-steps followed by Newton-Raphson solve.

        Args:
            build_system: Callback returning (lhs, rhs) variational forms.
            state: Solution state container updated in-place.
            bcs: Dirichlet boundary conditions enforced on the system.
        """
        self.picard.solve_system(build_system,state,bcs)
        self.newton.solve_system(build_system,state,bcs)