from .base import SolverStrategy
from fenics import solve


class LinearStrategy(SolverStrategy):
    """Direct solver strategy for linear variational problems."""
    def __init__(self):
        super().__init__()

    def solve_system(self,build_system,state,bcs) -> None:
        """Assembles and solves the linear system a(u, v) = L(v).

        Args:
            build_system: Callback returning (lhs, rhs) forms with linear=True.
            state: Solution state container updated in-place.
            bcs: Dirichlet boundary conditions enforced on the system.
        """
        lhs,rhs = build_system(state.x,state.mu_func,True)
        solve(lhs == rhs,state.x,bcs)
        