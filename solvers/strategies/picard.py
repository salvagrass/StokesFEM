from .base import SolverStrategy
from fenics import solve

class PicardStrategy(SolverStrategy):
    """Solves nonlinear systems via fixed-point Picard iteration."""
    def __init__(self,
                 mu_guess: float = 1.0,
                 quad_points: int = 6,
                 num_iter: int = 100,
                 abs_tol: float = 1e-00,
                 rel_tol: float = 1e-2,
                 raise_error: bool = True               
                 ):
        """Initializes the Picard solver strategy.

        Args:
            mu_guess: Constant viscosity value for the initial guess.
            quad_points: Quadrature degree for numerical integration.
            num_iter: Maximum number of Picard iterations.
            abs_tol: Absolute l2 error convergence tolerance.
            rel_tol: Relative l2 error convergence tolerance.
            raise_error: If True, raises RuntimeError when iterations exceed num_iter.
        """
        self.mu_guess = float(mu_guess)
        self.quad_points = int(quad_points)
        self.num_iter = int(num_iter)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.raise_error = raise_error



    def solve_system(self, build_system,state,bcs) -> None:
        """Executes fixed-point Picard iterations on the variational system.

        Args:
            build_system: Callback returning (lhs, rhs) forms given state and viscosity.
            state: Solution state container updated in-place.
            bcs: Dirichlet boundary conditions enforced on the system.

        Raises:
            RuntimeError: If convergence criteria are not satisfied within num_iter.
        """
        
        # Solving with initial guess
        mu_init = lambda u: self.mu_guess
        lhs, rhs = build_system(state.x,mu_init,True)
        solve(lhs==rhs,state.x,bcs)

        # At each step k we have to solve the problem with mu_{k-1}
        for i in range(0,self.num_iter):

            x_old_vec = state.x.vector().copy()

            mu_old = lambda u: state.compute_viscosity()
            lhs,rhs = build_system(state.x,mu_old,True)
            solve(lhs==rhs,state.x,bcs)

            diff_vec = state.x.vector() - x_old_vec
            abs_err = diff_vec.norm("l2")

            # Compute the relative error
            x_norm = state.x.vector().norm("l2")
            rel_err = abs_err/x_norm if x_norm > 1e-14 else abs_err

            # We ignore the stopping criteria if set to zero
            converged_abs = (self.abs_tol > 0 and abs_err < self.abs_tol)
            converged_rel = (self.rel_tol > 0 and rel_err < self.rel_tol)

            print(f"Picard iter {i+1:2d} | Abs err: {abs_err:.2e} (Tol: {self.abs_tol:.2e}) | Rel err: {rel_err:.2e} (Tol: {self.rel_tol:.2e})")

            if converged_abs or converged_rel:
                print(f"--- Picard converged after {i+1} iterations reached) ---")
                return
        if(self.raise_error):
            msg = f"*** Error: Picard solver failed to converge after {self.num_iter} iterations."
            raise RuntimeError(msg)


        