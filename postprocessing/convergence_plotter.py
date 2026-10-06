from typing import Literal, Optional, Sequence, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from common import MethodType, ErrorType
from postprocessing.metrics import ConvergenceStudy


class ConvergencePlotter:
    """Visualizes finite element convergence rates across multiple error norms.

    Generates a dynamic multi-panel figure based on the norms active across 
    the provided discretization schemes. Supports both h-refinement (log-log scale) 
    and p-refinement (semi-logarithmic scale) studies with optional error normalization 
    and theoretical reference slopes.
    """

    ORDER_STYLE_MAP = {
        1: ":",                     # Dotted
        2: "--",                    # Dashed
        3: "-.",                    # Dash-dot
        4: (0, (5, 1)),             # Densely dashed
        5: (0, (3, 1, 1, 1)),       # Densely dash-dot
        6: (0, (3, 1, 1, 1, 1, 1))  # Densely dash-dot-dot
    }

    # Canonical order ensuring consistent visual layout across varying methods
    CANONICAL_ORDER = [
        ErrorType.U_L2,
        ErrorType.U_H1,
        ErrorType.P_L2,
        ErrorType.T_L2,
        ErrorType.T_HDIV
    ]

    def __init__(
        self,
        methods: Sequence[MethodType],
        mode: Literal['p', 'h'] = 'h',
        prefix: Optional[str] = None,
        normalized: bool = False
    ):
        """Initialize the convergence plotting canvas and build the dynamic layout.

        Args:
            methods (Sequence[MethodType]): List of methods to be analyzed and plotted.
            mode ('p', 'h'): Discretization study mode ('h' for mesh refinement,
                'p' for polynomial degree elevation). Defaults to 'h'.
            prefix (str, optional): Text prepended to subplot titles. Defaults to None.
            normalized (bool, optional): If True, scales error sequences by their initial
                value (e_i / e_0). Defaults to False.

        Raises:
            ValueError: If no active norms are found across the provided methods.
        """
        self.mode = mode
        self.normalized = normalized
        self.prefix = f"{prefix} - " if prefix else ""

        # 1. Collect all distinct active norms across all target methods
        active_set = set()
        for method in methods:
            active_set.update(method.accepted_norms)

        # 2. Filter and sort active norms according to the canonical order
        self.active_norms: List[ErrorType] = [
            norm for norm in self.CANONICAL_ORDER if norm in active_set
        ]

        # 3. Dynamic figure and grid geometry allocation
        n_plots = len(self.active_norms)
        if n_plots == 0:
            raise ValueError("No active norms found in the provided methods.")

        figsize = (18, 10) if n_plots > 3 else (6 * n_plots, 6)
        self.fig = plt.figure(figsize=figsize)

        # Mapping: ErrorType -> corresponding matplotlib Axes
        self.axes_map: Dict[ErrorType, Axes] = {}
        axes_list = self._create_gridspec_layout(n_plots)

        # 4. Configure individual subplots
        x_label = r"Mesh size $h$" if mode == "h" else r"Polynomial degree $k$"
        y_label = "Normalized Error" if normalized else "Error"

        for norm, ax in zip(self.active_norms, axes_list):
            self.axes_map[norm] = ax

            # Retrieve LaTeX representation directly from the ErrorType definition
            ax.set_title(rf"{self.prefix}${norm.label[0]}$", fontsize=14)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, which="both", ls="--", alpha=0.5)

            if mode == "h":
                ax.invert_xaxis()

    def _create_gridspec_layout(self, n_plots: int) -> List[Axes]:
        """Construct the optimal subplot grid geometry depending on metric count.

        Args:
            n_plots (int): Total number of subplots to place.

        Returns:
            List[Axes]: Flattened list of allocated Axes instances.
        """
        axes = []
        if n_plots == 5:
            # 3 subplots on top row, 2 centered subplots on bottom row (6 virtual columns)
            gs = GridSpec(2, 6, figure=self.fig)
            axes.append(self.fig.add_subplot(gs[0, 0:2]))
            axes.append(self.fig.add_subplot(gs[0, 2:4]))
            axes.append(self.fig.add_subplot(gs[0, 4:6]))
            axes.append(self.fig.add_subplot(gs[1, 1:3]))
            axes.append(self.fig.add_subplot(gs[1, 3:5]))
        elif n_plots == 4:
            # Standard balanced 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            axes = [self.fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        else:
            # Single-row layout (1, 2, or 3 panels)
            gs = GridSpec(1, n_plots, figure=self.fig)
            axes = [self.fig.add_subplot(gs[0, i]) for i in range(n_plots)]

        return axes

    def add_error_line(
        self,
        error_type: ErrorType,
        method: MethodType,
        err_vals: Sequence[float],
        h_vals: Sequence[float],
        color: Optional[str] = None,
        marker: Optional[str] = None
    ) -> None:
        """Plot a single error curve on the designated metric panel.

        Args:
            error_type (ErrorType): Target metric subplot.
            method (MethodType): Discretization scheme identifier for styling.
            err_vals (Sequence[float]): Error values across refinement levels.
            h_vals (Sequence[float]): Mesh resolutions (h) or polynomial degrees (p).
            color (str, optional): Custom line color. Defaults to method style.
            marker (str, optional): Custom marker symbol. Defaults to method style.
        """
        if error_type not in self.axes_map:
            return  # Skip silently if metric is not part of this layout

        ax = self.axes_map[error_type]
        color = color or method.style[0]
        marker = marker or method.style[1]

        err_arr = np.array(err_vals, dtype=float)
        if self.normalized:
            err_arr = err_arr / err_arr[0]

        if self.mode == 'h':
            ax.loglog(h_vals, err_arr, color=color, marker=marker, label=method.value, linewidth=1.8)
        else:
            ax.semilogy(h_vals, err_arr, color=color, marker=marker, label=method.value, linewidth=1.8)

        ax.legend(fontsize=10)

    def plot_from_study(self, study: ConvergenceStudy) -> None:
        """Extract and plot all error trajectories from a ConvergenceStudy instance.

        Args:
            study (ConvergenceStudy): Completed study with populated error histories.
        """
        for norm in study.norms:
            self.add_error_line(norm, study.method, study.errors[norm], study.refs)

    def add_reference_slope(self, error_type: ErrorType, h_vals: Sequence[float], order: int) -> None:
        """Draw an asymptotic theoretical slope O(h^order) on a specific panel.

        Args:
            error_type (ErrorType): Target metric subplot for the reference line.
            h_vals (Sequence[float]): Discretization parameter values (h).
            order (int): Asymptotic convergence order exponent.
        """
        if self.mode == 'p' or error_type not in self.axes_map:
            return

        ax = self.axes_map[error_type]
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
        ls = self.ORDER_STYLE_MAP.get(order, "--")

        ax.loglog(h_arr, ref_line, color="black", linestyle=ls, alpha=0.6, label=rf"$\mathcal{{O}}(h^{order})$")
        ax.legend(fontsize=10)