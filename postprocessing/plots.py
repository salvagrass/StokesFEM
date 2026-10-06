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
                 ):
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

def plot_pressure(solution: StokesSolution,ax: Axes = None,title:str = 'Pressure Plot',cmap: str = 'coolwarm',wireframe: bool = False):
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
