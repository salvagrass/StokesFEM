from .base import SolverStrategy
from fenics import Constant,solve

class NewtonStrategy(SolverStrategy):
    """Solves nonlinear variational systems via the Newton-Raphson method.

    Optionally computes an initial guess by first solving a linearized
    system with a constant viscosity estimate.
    """
    def __init__(self, 
                 mu_guess: float = 1,
                 quad_points: int = 6, 
                 relaxation: float = 0.8,
                 max_iteration: int = 50,
                 abs_tol: float = 1e-10,
                 rel_tol: float = 1e-9,
                 not_guess: bool = False
                 ):
        """Initializes the Newton solver strategy.

        Args:
            mu_guess: Viscosity value for the optional initial linear guess.
            quad_points: Quadrature degree for numerical integration.
            relaxation: Under-relaxation parameter (0 < relaxation <= 1).
            max_iteration: Maximum number of Newton iterations allowed.
            abs_tol: Absolute residual convergence tolerance.
            rel_tol: Relative residual convergence tolerance.
            not_guess: If True, skips the initial linear solve step.
        """
        super().__init__()
        # Forcing types for Fenics solver
        self.mu_guess = Constant(mu_guess)
        self.quad_points = int(quad_points)
        self.relaxation = float(relaxation)
        self.max_iter = int(max_iteration)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.not_guess = not_guess

    def solve_system(self, build_system,state,bcs) -> None:
        """Executes the Newton-Raphson solver on the discrete variational form.

        Args:
            build_system: Callback returning (lhs, rhs) for a given state.
            state: Solution state container updated in-place.
            bcs: Dirichlet boundary conditions enforced on the system.
        """

        if not self.not_guess:
            mu = lambda u: self.mu_guess
            lhs,rhs = build_system(state.x,mu,True)
            solve(lhs==rhs,
                    state.x,
                    bcs,
                    form_compiler_parameters = {"quadrature_degree" : self.quad_points}
                    )

        lhs,rhs = build_system(state.x,state.mu_func)
        solve(lhs-rhs==0,
              state.x,
              bcs,
              form_compiler_parameters = {"quadrature_degree" : self.quad_points},
              solver_parameters={
                "newton_solver": {
                    "linear_solver": "umfpack",
                    "relaxation_parameter": self.relaxation, # Setting the learning rate lower than one to avoid non convergence 
                    "maximum_iterations": self.max_iter,
                    "relative_tolerance": self.rel_tol,
                    "absolute_tolerance": self.abs_tol
                    }
                }
            )