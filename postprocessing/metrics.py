from problems.base import StokesProblemData
from solvers import StokesSolution,MethodType

import numpy as np
from fenics import errornorm

from typing import List,TypedDict,Optional


class StokesError(TypedDict):
    u_l2: float
    u_h1: float
    p_l2: float


class ErrorMetrics(TypedDict):
    u_l2: List[Optional[float]]
    u_h1: List[Optional[float]]
    p_l2: List[Optional[float]]

def compute_errors(data: StokesProblemData,solution: StokesSolution, degree_rise: int = 3) -> StokesError :
    """Compute numerical discretization errors against exact analytical fields.

    Calculates the velocity L2 norm, the velocity H1-seminorm (H1_0), and the
    pressure L2 norm using higher-order quadrature interpolation.

    Args:
        data (StokesProblemData): Problem data container providing exact solutions.
        solution (StokesSolution): Computed discrete velocity and pressure fields.
        degree_rise (int, optional): Polynomial degree elevation used for accurate
            quadrature integration of the error. Defaults to 3.

    Returns:
        StokesError: Calculated error values in L2(u), H1_0(u), and L2(p).

    Raises:
        ValueError: If exact analytical solutions are not defined on the problem data.
    """
    if data.u_exact is None or data.p_exact is None:
        raise ValueError("Analytical solution not available")
    u_l2 = float(errornorm(data.u_exact, solution.u,norm_type='L2',degree_rise=degree_rise))
    u_h1 = float(errornorm(data.u_exact,solution.u,norm_type='H10',degree_rise=degree_rise))
    p_l2 = float(errornorm(data.p_exact,solution.p,norm_type='L2',degree_rise=degree_rise))
    return StokesError(u_l2=u_l2,u_h1=u_h1,p_l2=p_l2)


class ConvergenceStudy:
    """Manages error histories and computes online Experimental Orders of Convergence (EOC).

    Tracks errors across successive h-refinement steps and formats results
    for console printing or rich HTML rendering in Jupyter Notebooks.
    """
    def __init__(self,method_name: MethodType):
        """Initialize empty error history lists and record the method identifier.

        Args:
            method_name (MethodType): Discretization scheme identifier.
        """
        self.errors: ErrorMetrics = {'u_l2': [],'u_h1': [],'p_l2': []}
        self.eocs: ErrorMetrics = {'u_l2': [],'u_h1': [],'p_l2': []}
        self.refs: List[float] = []

        self.method_name = method_name

    def add_error_data(self,errors: StokesError,mesh_h: float):
        """Append error metrics for a given refinement level and compute online EOC.

        Args:
            errors (StokesError): Error dictionary for the current mesh resolution.
            mesh_h (float): Characteristic mesh cell size (h).
        """
        # Computing the EOC online
        if self.refs:
            for name in errors.keys():
                eoc = np.log(self.errors[name][-1]/errors[name])/np.log(self.refs[-1]/mesh_h)
                self.eocs[name].append(float(eoc))
        else:
            for name in errors.keys():
                self.eocs[name].append(None)
        
        for name,value in errors.items():
            self.errors[name].append(value)

        self.refs.append(mesh_h)

    def _repr_html_(self) -> str:
        """Render the convergence study as an HTML table in Jupyter Notebooks."""
        title = (
            f"Convergence Study: {self.method_name.upper()}"
            if getattr(self, "method_name", None)
            else "Convergence Study"
        )

        html = [
            f"<div style='font-family: sans-serif; margin-bottom: 10px;'><strong>{title}</strong></div>",
            "<table style='border-collapse: collapse; text-align: center;'>",
            "<thead>",
            "<tr style='border-bottom: 2px solid black;'>",
            "<th style='padding: 8px;'>$h$</th>",
            "<th style='padding: 8px;'>$\\|u-u_h\\|_{L^2}$</th><th style='padding: 8px;'>EOC</th>",
            "<th style='padding: 8px;'>$\\|u-u_h\\|_{H^1}$</th><th style='padding: 8px;'>EOC</th>",
            "<th style='padding: 8px;'>$\\|p-p_h\\|_{L^2}$</th><th style='padding: 8px;'>EOC</th>",
            "</tr>",
            "</thead>",
            "<tbody>",
        ]

        # Iterate over all recorded refinement levels
        for i in range(len(self.refs)):
            h_str = f"{self.refs[i]:.4e}"
            row = [
                f"<tr style='border-bottom: 1px solid #ddd;'><td style='padding: 8px;'>{h_str}</td>"
            ]

            # Enforce deterministic column ordering
            for metric in ["u_l2", "u_h1", "p_l2"]:
                err = self.errors[metric][i]
                eoc = self.eocs[metric][i]

                err_str = f"{err:.2e}"
                eoc_str = f"{eoc:.2f}" if eoc is not None else "-"

                row.append(
                    f"<td style='padding: 8px;'>{err_str}</td><td style='padding: 8px;'>{eoc_str}</td>"
                )

            row.append("</tr>")
            html.append("".join(row))

        html.append("</tbody></table>")
        return "".join(html)

    def print_table(self) -> None:
        """Print the convergence study table to standard output."""
        title = (
            f"Convergence Study: {self.method_name.upper()}"
            if getattr(self, "method_name", None)
            else "Convergence Study"
        )

        print(f"\n{title}")
        print("=" * 86)
        print(
            f"{'h':<10} | "
            f"{'L2(u)':<11} {'EOC':<6} | "
            f"{'H1(u)':<11} {'EOC':<6} | "
            f"{'L2(p)':<11} {'EOC':<6}"
        )
        print("-" * 86)

        for i in range(len(self.refs)):
            h_str = f"{self.refs[i]:.4e}"
            cols = []

            for metric in ["u_l2", "u_h1", "p_l2"]:
                err = self.errors[metric][i]
                eoc = self.eocs[metric][i]

                err_str = f"{err:.2e}"
                eoc_str = f"{eoc:.2f}" if eoc is not None else "-"
                cols.append(f"{err_str:<11} {eoc_str:<6}")

            print(f"{h_str:<10} | {' | '.join(cols)}")

        print("=" * 86)
        
