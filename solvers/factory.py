
from solvers.primal import TaylorHoodSolver,MINISolver,KSSolver
from solvers.mixed import AFWSolver

class SolverFactory:
    registry = {
        "th": lambda fluid, **kw: TaylorHoodSolver(fluid, degree=kw.get("degree", 2)),
        "mini":        lambda fluid, **kw: MINISolver(fluid),
        "ks":          lambda fluid, **kw: KSSolver(fluid),
        "afw": lambda fluid,**kw: AFWSolver(fluid,degree=kw.get("degree",1))
    }

    @classmethod
    def create_solver(cls, method: str,fluid_model,**kwargs):
        builder = cls.registry.get(method)
        if not builder:
            raise ValueError("Method not supported")
        return builder(fluid_model,**kwargs)