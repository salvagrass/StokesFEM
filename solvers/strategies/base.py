from solvers.state import State
from abc import ABC,abstractmethod

class SolverStrategy(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def solve_system(self,build_system,x: State,bcs):
        pass    