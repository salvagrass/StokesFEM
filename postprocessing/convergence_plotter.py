from typing import Literal, Optional, Sequence, List, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

    def __init__(
        self,
        mode: Literal['p', 'h'] = 'h',
        prefix: Optional[str] = None,
        normalized: bool = False
    ):
        """Initialize the convergence plotting canvas and build the dynamic layout.

        Args:
            mode ('p', 'h'): Discretization study mode ('h' for mesh refinement,
                'p' for polynomial degree elevation). Defaults to 'h'.
            prefix (str, optional): Text prepended to subplot titles. Defaults to None.
            normalized (bool, optional): If True, scales error sequences by their initial
                value (e_i / e_0). Defaults to False.
        """
        self.mode = mode
        self.prefix = f"{prefix} - " if prefix else ""
        self.normalized = normalized


    def add_error_line(
        self,
        ax: Axes,
        label: str,
        err_vals: Sequence[float],
        h_vals: Sequence[float],
        color: str,
        marker: str 
    ) -> None:
        """Plot a single convergence error curve onto a Matplotlib Axes.

        Applies normalization by the initial error value if ``self.normalized``
        is True. Plots on a log-log scale for h-refinement (``self.mode == 'h'``)
        or a semi-log y scale for p-refinement, then updates the axis legend.

        Args:
            ax (Axes): Target Matplotlib axes where the curve will be drawn.
            label (str): Label displayed in the plot legend.
            err_vals (Sequence[float]): Error values across refinement levels.
            h_vals (Sequence[float]): Mesh resolutions (h) or polynomial degrees (p).
            color (str): Line and marker color specifier.
            marker (str): Matplotlib marker style string (e.g., 'o', 's', '^').
        """

        err_arr = np.array(err_vals, dtype=float)
        if self.normalized:
            err_arr = err_arr / err_arr[0]

        if self.mode == 'h':
            ax.loglog(h_vals, err_arr, color=color, marker=marker, label=label, linewidth=1.8)
        else:
            ax.semilogy(h_vals, err_arr, color=color, marker=marker, label=label, linewidth=1.8)

        ax.legend(fontsize=10)
    

    def add_reference_slope(self, ax: Axes, h_vals: Sequence[float], order: int) -> None:
        """Draw an asymptotic theoretical slope O(h^order) on a specific panel.

        Args:
            ax (Axes): Target Matplotlib axes where the reference line will be drawn
            h_vals (Sequence[float]): Discretization parameter values (h).
            order (int): Asymptotic convergence order exponent.
        """
        if self.mode == 'p':
            return

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



    def plot_from_studies(self, 
                               studies: Union[ConvergenceStudy,List[ConvergenceStudy]],
                               figsize: Optional[Tuple[float,float]] = (10,6),
                               reference_orders: Optional[Tuple[int]] = None,
                               show: Optional[bool] = False
                               ) -> Dict[ErrorType,Figure]:
        """Extract and plot error trajectories across multiple studies into separate figures.

        Iterates over all unique error norms accepted by the provided numerical methods,
        creating an independent figure for each norm and plotting the convergence curves
        of every compatible study.

        Args:
            studies (Union[ConvergenceStudy, List[ConvergenceStudy]]): Single study 
                or list of completed convergence studies containing error histories.
            figsize (Tuple[float, float], optional): Dimensions (width, height)
                in inches for each generated figure. Defaults to (10, 6).
            show (bool, optional): If True, leaves the figure open for display;
                if False, closes the figure to avoid GUI window clutter and free memory.
                Defaults to False.

        Returns:
            Dict[ErrorType, Figure]: Dictionary mapping each analyzed norm to its 
            corresponding Matplotlib Figure object.
        """
        studies_list = studies if isinstance(studies,(list,tuple)) else [studies]
    
        # Extract the set of all the available norms in the studies and order them
        available_norms = {norm_type for study in studies_list for norm_type in study.method.accepted_norms}
        ordered_norms = [norm for norm in ErrorType if norm in available_norms]

        figures = {}
        for norm in ordered_norms:
            fig,ax = plt.subplots(figsize=figsize)
            figures[norm] = fig
            for study in studies_list:
                if norm in study.method.accepted_norms:
                    self.add_error_line(ax,
                                        study.method,
                                        study.errors[norm],
                                        study.refs,
                                        study.method.style[0],
                                        study.method.style[1]
                                        )
            if reference_orders:
                for degree in reference_orders:
                    self.add_reference_slope(ax,study.refs,degree)

            ax.set_title(rf"{self.prefix}Convergence: ${norm.label[0]}$ norm", fontsize=14)
            ax.set_ylabel(rf"Error (${norm.label[0]}$)", fontsize=12)
            xlabel = "h refinements" if self.mode=="h" else "p"
            ax.set_xlabel(xlabel=xlabel)
            if not show:
                plt.close(fig)
        return figures