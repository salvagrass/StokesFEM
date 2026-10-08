from typing import Optional,Any
from common import MethodType
from solvers.base import StokesSolver
from solvers.mixed import AFWSolver, HybridSolver, RTCGSolver
from solvers.primal import KSSolver, MINISolver, TaylorHoodSolver
from solvers.augmented import AugmentedSolver


class SolverFactory:
    """Factory dispatching concrete Stokes solver instances based on MethodType."""
    registry = {
        MethodType.TH: lambda fluid, **kw: TaylorHoodSolver(
            fluid, degree=kw.get("degree", 2)
        ),
        MethodType.MINI: lambda fluid, **kw: MINISolver(fluid),
        MethodType.KS: lambda fluid, **kw: KSSolver(fluid),
        MethodType.AFW: lambda fluid, **kw: AFWSolver(
            fluid, degree=kw.get("degree", 1)
        ),
        MethodType.RTCG: lambda fluid, **kw: RTCGSolver(
            fluid, degree=kw.get("degree", 2)
        ),
        MethodType.HYBRID: lambda fluid, **kw: HybridSolver(fluid),
        MethodType.AUG : lambda fluid,**kw: AugmentedSolver(fluid,degree=kw.get("degree",2))
    }

    @classmethod
    def create_solver(
        cls,
        method: MethodType,
        fluid_model: Optional[Any] = None,
        **kwargs: Any,
    ) -> StokesSolver:
        """Instantiate and configure a StokesSolver subclass.

        Args:
            method (MethodType): Desired discretization scheme from MethodType.
            fluid_model (Optional[Any], optional): Rheological model for non-Newtonian
                or variable-viscosity fluids. Defaults to None.
            **kwargs: Additional formulation parameters passed to the solver constructor
                (e.g., polynomial degree).

        Returns:
            StokesSolver: An instantiated solver configured for the selected scheme.

        Raises:
            ValueError: If the requested method is not present in the registry.
        """
        builder = cls.registry.get(method)
        if not builder:
            valid_methods = [m.value for m in cls.registry.keys()]
            raise ValueError(
                f"Method '{method}' is not supported. Valid options: {valid_methods}"
            )

        return builder(fluid_model, **kwargs)