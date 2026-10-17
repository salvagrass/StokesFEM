from solvers.state import State
from abc import ABC,abstractmethod

class SolverStrategy(ABC):
    """Abstract base class defining the strategy for solving discrete systems."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def solve_system(self,build_system,x: State,bcs) -> None:
        """Solves the variational system and updates the state.

        Args:
            build_system: Callback returning the variational forms (lhs, rhs).
            x: Current solution state container.
            bcs: List of Dirichlet boundary conditions to enforce.
        """
        pass    