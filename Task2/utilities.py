import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typing as ty
from MD import *

def plot_velocity(
        velocities: np.ndarray,
        plot_filename: ty.Union[None, str] = None
):
    plt.style.use("../utils/plotstyle.mplstyle")
    row = 1
    col = 1
    fig = plt.figure(figsize=(3.37*col, 1.89*row), dpi=300,facecolor='white')
    gs = fig.add_gridspec(row,col)
    ax  = fig.add_subplot(gs[0])
    y, x = np.histogram(velocities.flatten(), bins=50)
    ax.plot(x[:-1], y, label='velocity distribution')
    ax.set_xlabel('velocity')
    ax.set_ylabel('counts')
    ax.legend()
    if plot_filename:
        fig.savefig(plot_filename)
    fig.show()

def plot_data(
        results: npt.ArrayLike,
        plot_filename: ty.Union[None, str] = None
):
    plt.style.use("../utils/plotstyle.mplstyle")
    row = 1
    col = 1
    fig = plt.figure(figsize=(3.37*col, 1.89*row), dpi=300,facecolor='white')
    gs = fig.add_gridspec(row,col)
    ax  = fig.add_subplot(gs[0])
    for key, val in results.items():
        if key == 'nsteps':
            continue
        ax.plot(results['nsteps'], val, label=key)
    ax.set_xlabel('steps')
    ax.set_ylabel('quantity')
    ax.legend()
    if plot_filename:
        fig.savefig(plot_filename)
    fig.show()