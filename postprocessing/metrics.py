from typing import Dict, List, Optional
import numpy as np

from common import MethodType, ErrorType
from problems import StokesProblemData
from solvers import StokesSolution


class ConvergenceStudy:
    """Manages error histories and computes online Experimental Orders of Convergence (EOC).

    Tracks errors across successive h-refinement steps and formats results
    for console printing or rich HTML rendering in Jupyter Notebooks.
    """

    def __init__(self, method: MethodType):
        """Initialize empty error history lists dynamically based on the method.

        Args:
            method (MethodType): Discretization scheme identifier containing accepted norms.
        """
        self.method = method
        self.norms: List[ErrorType] = method.accepted_norms
        self.refs: List[float] = []

        # Initialize lists dynamically for exact required norms
        self.errors: Dict[ErrorType, List[float]] = {norm: [] for norm in self.norms}
        self.eocs: Dict[ErrorType, List[Optional[float]]] = {norm: [] for norm in self.norms}

    def add_error_data(self, errors: Dict[ErrorType, float], mesh_h: float) -> None:
        """Append error metrics for a given refinement level and compute online EOC.

        Args:
            errors (Dict[ErrorType, float]): Error dictionary for the current mesh resolution.
            mesh_h (float): Characteristic mesh cell size (h).
        """
        # 1. Computing the EOC online (Iterating over self.norms ensures deterministic order)
        if self.refs:
            last_h = self.refs[-1]
            for norm in self.norms:
                prev_err = self.errors[norm][-1]
                curr_err = errors[norm]
                
                # EOC = log(e_prev / e_curr) / log(h_prev / h_curr)
                eoc = np.log(prev_err / curr_err) / np.log(last_h / mesh_h)
                self.eocs[norm].append(float(eoc))
        else:
            for norm in self.norms:
                self.eocs[norm].append(None)
        
        # 2. Store error values and refinement level
        for norm in self.norms:
            self.errors[norm].append(float(errors[norm]))

        self.refs.append(mesh_h)

    def add_step(
        self, data: StokesProblemData, solution: StokesSolution, mesh_h: float, **kw
    ) -> Dict[ErrorType, float]:
        """Convenience wrapper to compute and log errors in a single step.
        
        Args:
            data: The analytical problem definition.
            solution: The discrete solver output.
            mesh_h: Characteristic mesh size.
            **kw: Forwarded arguments (like degree_rise) for integration.
        """
        from postprocessing.errors import compute_errors
        
        vals = compute_errors(data, solution, self.method, **kw)
        self.add_error_data(vals, mesh_h)
        return vals

    

    def _repr_markdown_(self) -> str:
        """Render the convergence study as a dynamic Markdown table in Jupyter Notebooks."""

        # Helper function to convert Python scientific notation (1.2e-03) 
        # to LaTeX scientific notation (1.2 \times 10^{-3})
        def to_latex_sci(val: float, precision: int) -> str:
            if val == 0.0:
                return "$0$"
                
            # Format with standard 'e' and split mantissa and exponent
            m, e = f"{val:.{precision}e}".split("e")
            
            # Convert exponent to int to remove '+' and leading zeros (e.g., '-03' -> '-3')
            exp = int(e)
            
            # Use triple braces {{{ }}} to render literal curly braces in the f-string
            return f"${m} \\times 10^{{{exp}}}$"

        title = f"**Convergence Study: {self.method.value.upper()}**\n\n"

        # 1. Build the Markdown table headers and alignment row
        header = "| $h$ |"
        separator = "|:---:|"  # Colons indicate centered alignment

        for norm in self.norms:
            tex_label = norm.label[0]
            header += f" ${tex_label}$ | EOC |"
            separator += ":---:|:---:|"
            
        markdown_lines = [title, header, separator]

        # 2. Build the data rows
        for i, h in enumerate(self.refs):
            # Format 'h' with 2 decimals in LaTeX
            h_str = to_latex_sci(h, 2)
            row = f"| {h_str} |"
            
            for norm in self.norms:
                err = self.errors[norm][i]
                eoc = self.eocs[norm][i]
                
                # Format 'error' with 2 decimals in LaTeX
                err_str = to_latex_sci(err, 2)
                eoc_str = f"{eoc:.2f}" if eoc is not None else "-"
                
                row += f" {err_str} | {eoc_str} |"
                
            markdown_lines.append(row)
        # 3. Join everything with newlines
        return "\n".join(markdown_lines)

    def print_table(self) -> None:
        """Print the dynamic convergence study table to standard output."""
        title = f"Convergence Study: {self.method.value.upper()}"
        
        # Build headers dynamically
        headers = ["h"]
        for norm in self.norms:
            # norm.label[1] gives the plain CLI string (e.g., 'L2(u)')
            headers.extend([norm.label[1], "EOC"])

        # Calculate width to format separators dynamically
        header_str = " | ".join(f"{h:<10}" for h in headers)
        sep = "-" * len(header_str)

        print(f"\n{title}")
        print("=" * len(header_str))
        print(header_str)
        print(sep)

        for i, h in enumerate(self.refs):
            row = [f"{h:.4e}"]
            for norm in self.norms:
                err = self.errors[norm][i]
                eoc = self.eocs[norm][i]
                eoc_str = f"{eoc:.2f}" if eoc is not None else "-"
                
                row.extend([f"{err:.2e}", eoc_str])
                
            print(" | ".join(f"{item:<10}" for item in row))

        print("=" * len(header_str))