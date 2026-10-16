from .base import SolverStrategy
from .picard import PicardStrategy
from .newton import NewtonStrategy


class HybridStrategy(SolverStrategy):
    def __init__(self,
                 mu_guess: float = 1.0,
                 quad_points: int = 6,
                 relaxation: float = 1.0,
                 num_iter_p: int = 10,
                 num_iter_n: int = 50,
                 abs_tol: float = 1e-10,
                 rel_tol: float = 1e-9
                 ):
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

    def solve_system(self, build_system, state, bcs):
        self.picard.solve_system(build_system,state,bcs)
        self.newton.solve_system(build_system,state,bcs)