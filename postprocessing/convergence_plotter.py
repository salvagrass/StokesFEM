from postprocessing.metrics import ConvergenceStudy
from solvers.base import MethodType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Literal,Optional,Sequence



class ConvergencePlotter:
    """Visualizes finite element convergence rates across multiple error norms.

    Generates a 3-panel figure comparing velocity L2, velocity H1-seminorm,
    and pressure L2 errors. Supports both h-refinement (log-log scale) and
    p-refinement (semi-logarithmic scale) studies with optional error normalization
    and theoretical reference slopes.
    """

    STYLE_MAP= {
        MethodType.TH:   ("tab:blue",   "o"),  
        MethodType.MINI: ("tab:orange", "s"),  
        MethodType.KS:   ("tab:green",  "^"),  
        MethodType.AFW:  ("tab:red",    "D"),  
        MethodType.RTCG:   ("tab:purple", "v"),  
        MethodType.HYBRID:   ("tab:brown",  "p"),  
    }

    ORDER_STYLE_MAP = {
        1: ":",                     # Dotted 
        2: "--",                    # Dashed
        3: "-.",                    # Dash-dot
        4: (0, (5, 1)),             # Densely dashed
        5: (0, (3, 1, 1, 1)),       # Densely dash-dot
        6: (0, (3, 1, 1, 1, 1, 1))  # Densely dash-dot-dot
    }

    ERROR_DICT = {
        "u_l2": 0,
        "u_h1": 1,
        "p_l2": 2
    }

    def __init__(self, mode: Literal['p','h'], prefix: Optional[str] = None, normalized: bool = True):
        """Initialize the convergence plotting canvas and subplots.

        Args:
            mode ('p', 'h'): Discretization study mode ('h' for mesh refinement,
                'p' for polynomial degree elevation).
            prefix (str, optional): Optional text prepended to subplot titles.
                Defaults to None.
            normalized (bool, optional): If True, normalizes error sequences by their initial
                value (e_i / e_0). Defaults to True.
        """
        self.mode = mode
        self.fig, self.axes = plt.subplots(1,3,figsize=(20,6))
        self.normalized = normalized

        prefix = f"{prefix} - " if prefix else ""

        self.axes[0].set_title(rf"{prefix}Velocity $\|u - u_h\|_{{L^2}}$")
        self.axes[1].set_title(rf"{prefix}Velocity $\|u - u_h\|_{{H^1_0}}$")
        self.axes[2].set_title(rf"{prefix}Pressure $\|p - p_h\|_{{L^2}}$")

        x_label = r"Mesh size $h$" if mode == "h" else r"Polynomial degree $k$"
        for ax in self.axes:
            ax.set_xlabel(x_label, fontsize=11)
            y_label = "Normalized Error" if normalized else "Error"
            ax.set_ylabel(y_label, fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.5)
            if mode == "h":
                ax.invert_xaxis()

                
    def add_error_line(self,
                       error_type: Literal['u_l2','u_h1','p_l2'],
                       method_label: str,
                       err_vals : Sequence[float],
                       h_vals: Sequence[float],
                       color: Optional[str] = None,
                       marker: Optional[str] = None
                       ):
        """Plot a single error curve on the specified metric panel.

        Args:
            error_type: Metric identifier targeting the subplot.
            method_label: Scheme identifier used for labeling and styling.
            err_vals: Recorded error values across refinement levels.
            h_vals: Mesh steps (h) or polynomial degrees (p).
            color: Custom line and marker color. Defaults to None.
            marker: Custom data marker symbol. Defaults to None.
        """
        idx = ConvergencePlotter.ERROR_DICT[error_type]
        color = color or ConvergencePlotter.STYLE_MAP[method_label][0]
        marker = marker or ConvergencePlotter.STYLE_MAP[method_label][1]

        ax : Axes = self.axes[idx]

        err_arr = np.array(err_vals,dtype=float)
        if self.normalized:
            err_arr = err_arr/err_arr[0]
        if self.mode == 'h':
            ax.loglog(h_vals,err_arr,color=color,marker=marker,label=method_label,linewidth=1.8)
        else:
            ax.semilogy(h_vals, err_arr, color=color, marker=marker, label=method_label, linewidth=1.8)
        ax.legend(fontsize=10)


    def plot_from_study(self, study: ConvergenceStudy):
        """Extract and plot all error trajectories from a ConvergenceStudy instance.

        Args:
            study (ConvergenceStudy): Completed convergence study containing
                recorded errors and discretization coordinates.
        """
        for name in self.ERROR_DICT.keys():
            self.add_error_line(name,study.method_name,study.errors[name],study.refs)


    def add_reference_slope(self, error_type: Literal['u_l2', 'u_h1', 'p_l2'], h_vals: Sequence[float], order: int):
        """Add an asymptotic convergence reference line O(h^order) to a subplot.

        Args:
            error_type (Literal['u_l2', 'u_h1', 'p_l2']): Metric panel to draw the reference slope on.
            h_vals (Sequence[float]): Sequence of discretization mesh sizes.
            order (int): Convergence order exponent (e.g., 1, 2, 3).
        """
        if self.mode == 'p':
            return

        idx = ConvergencePlotter.ERROR_DICT[error_type]
        ax: Axes = self.axes[idx]
        
        h_arr = np.array(h_vals)

        if self.normalized:
            anchor_err = 1.0
        else:        
            lines = ax.get_lines()
            if lines:
                all_starting_y = [line.get_ydata()[0] for line in lines]
                anchor_err = min(all_starting_y) * 0.5
            else:
                anchor_err = 1.0 
                
        ref_line = anchor_err * (h_arr / h_arr[0]) ** order
        
        ls = ConvergencePlotter.ORDER_STYLE_MAP.get(order, "--")
        ax.loglog(h_arr, ref_line, color="black", linestyle=ls, alpha=0.6, label=rf"$\mathcal{{O}}(h^{order})$")
        ax.legend(fontsize=10)
    


        