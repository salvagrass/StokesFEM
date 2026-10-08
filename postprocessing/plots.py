import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from fenics import plot
from solvers.base import StokesSolution

from typing import Tuple,Literal
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_velocity(solution: StokesSolution,
                  ax: Axes = None,
                  title: str = 'Velocity Plot',
                  cmap: str = 'viridis',
                  wireframe: bool = False
                 ) -> Axes:
    """Plots the velocity field of a Stokes solution.

    Args:
        solution (StokesSolution): The data object containing the solved velocity 
            function
        ax (Optional[Axes], optional): Matplotlib axes where the plot will be drawn. 
            If None, a new figure and axes are created. Defaults to None.
        title (str, optional): The title of the plot. Defaults to 'Velocity Plot'.
        cmap (str, optional): The colormap used for the velocity magnitude. 
            Defaults to 'viridis'.
        wireframe (bool, optional): If True, overlays the underlying finite element 
            mesh grid on top of the plot. Defaults to False.

    Returns:
        Axes: The matplotlib axes containing the generated plot.
    """
    if ax is None:
        _,ax = plt.subplots(figsize=(6,5))
    plt.sca(ax)

    c = plot(solution.u,cmap = cmap)
    if wireframe:
        mesh = solution.u.function_space().mesh()
        plot(mesh,color='black',linewidth=0.8,alpha=0.4)
    ax.set_title(title,fontsize=11)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(c,cax=cax)
    return ax

def plot_pressure(
        solution: StokesSolution,
        ax: Axes = None,
        title:str = 'Pressure Plot',
        cmap: str = 'coolwarm',
        wireframe: bool = False
        ) -> Axes:
    """Plots the pressure field of a Stokes solution.

    Args:
        solution (StokesSolution): The data object containing the solved pressure 
            function
        ax (Optional[Axes], optional): Matplotlib axes where the plot will be drawn. 
            If None, a new figure and axes are created. Defaults to None.
        title (str, optional): The title of the plot. Defaults to 'Pressure Plot'.
        cmap (str, optional): The colormap used for the scalar pressure values. 
            Defaults to 'coolwarm'.
        wireframe (bool, optional): If True, overlays the underlying finite element 
            mesh grid on top of the plot. Defaults to False.

    Returns:
        Axes: The matplotlib axes containing the generated plot.
    """
    if ax is None:
        _,ax = plt.subplots(figsize=(6,5))
    plt.sca(ax)
    c = plot(solution.p,cmap = cmap)

    if wireframe:
        mesh = solution.p.function_space().mesh()
        plot(mesh,color='black',linewidth=0.8,alpha=0.4)
    ax.set_title(title,fontsize=11)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(c,cax=cax)
    return ax

def plot_combined(solution: StokesSolution,
                  mode: Literal['quiver','streamlines'] = 'quiver',
                  title_prefix: str = '', 
                  layout: Literal['horizontal','vertical'] = 'horizontal',
                  cmaps: Tuple[str,str] = ('viridis','coolwarm'),
                  wireframe: bool = False
                  ):
    """Creates a multi-panel figure displaying both velocity and pressure fields.

    Args:
        solution (StokesSolution): The data object containing both velocity 
            and pressure solved functions.
        mode (Literal['quiver', 'streamlines'], optional): The visualization mode 
            for the velocity field. 'quiver' plots vectors, 'streamlines' plots 
            flow trajectories. Defaults to 'quiver'.
        title_prefix (str, optional): A string prefix added to the default subplot 
            titles (e.g., method name). Defaults to ''.
        layout (Literal['horizontal', 'vertical'], optional): Subplot arrangement. 
            'horizontal' places plots side-by-side; 'vertical' stacks them. 
            Defaults to 'horizontal'.
        cmaps (Tuple[str, str], optional): A tuple of two colormap strings for 
            velocity and pressure, respectively. Defaults to ('viridis', 'coolwarm').
        wireframe (bool, optional): If True, overlays the mesh on both plots. 
            Defaults to False.

    Returns:
        Tuple[Figure, Union[Axes, np.ndarray]]: A tuple containing the main 
        matplotlib Figure object and the array of generated Axes.
    """
    nrows, ncols = (1, 2) if layout == "horizontal" else (2, 1)
    figsize = (12, 5) if layout == "horizontal" else (6, 5)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if mode == 'quiver':
        plot_velocity(solution, ax=axes[0], title=f"{title_prefix} Velocity Plot",cmap=cmaps[0],wireframe=wireframe)
    elif mode == 'streamlines':
        plot_streamlines(solution,ax=axes[0],title=f"{title_prefix} Streamlines",cmap=cmaps[0])
    plot_pressure(solution, ax=axes[1], title=f"{title_prefix} Pressure Plot",cmap=cmaps[1],wireframe=wireframe)
    if layout=='horizontal':
        fig.tight_layout(w_pad=3.0)
    else:
        fig.tight_layout(h_pad=3.0)
    return fig,axes

def plot_streamlines(
        solution: StokesSolution,
        ax: Axes = None,
        title: str = "Streamlines Plot",
        cmap: str = "viridis",
        resolution: int = 100,
        density: float = 1.2
        ) -> Axes:
    """Plots the flow streamlines of the velocity field.

    Args:
        solution (StokesSolution): The data object containing the solved velocity.
        ax (Optional[Axes], optional): Matplotlib axes where the plot will be drawn. 
            If None, a new figure and axes are created. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Streamlines Plot".
        cmap (str, optional): Colormap mapped to the velocity magnitude along 
            the streamlines. Defaults to "viridis".
        resolution (int, optional): The number of points along each axis for the 
            interpolation grid. Higher values yield smoother streamlines but take 
            longer to compute. Defaults to 100.
        density (float, optional): Controls the closeness of streamlines. 
            Defaults to 1.2.

    Returns:
        Axes: The matplotlib axes containing the generated streamlines.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    plt.sca(ax)

    X, Y, U, V = _construct_grid(solution, resolution)
    speed = np.sqrt(U**2 + V**2)

    strm = ax.streamplot(
        X, Y, U, V,
        color=speed,
        cmap=cmap,
        density=density
    )

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(strm.lines, cax=cax)

    return ax



def _construct_grid(solution: StokesSolution, resolution: int):
    mesh = solution.u.function_space().mesh()
    coords = mesh.coordinates()

    mins = np.min(coords,axis=0)
    maxs = np.max(coords,axis=0)

    vec_x = np.linspace(mins[0],maxs[0],resolution)
    vec_y = np.linspace(mins[1],maxs[1],resolution)

    X,Y = np.meshgrid(vec_x,vec_y)
    U = np.full(X.shape,np.nan)
    V = np.full(Y.shape,np.nan)

    for (i,j),x_val in np.ndenumerate(X):
        y_val = Y[i,j]
        try:
            value = solution.u((x_val,y_val))
            U[i,j], V[i,j] = value[0],value[1]
        except RuntimeError:
            pass
    return X,Y,U,V
