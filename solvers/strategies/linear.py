from .base import SolverStrategy
from fenics import solve


class LinearStrategy(SolverStrategy):
    def __init__(self):
        super().__init__()

    def solve_system(self,build_system,state,bcs):
        lhs,rhs = build_system(state.x,state.mu_func,True)
        solve(lhs == rhs,state.x,bcs)
        