import typing as ty
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def fit_line(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[npt.ArrayLike, float]:
    """Fit a 1st-order polynomial (line) `f(x) = a * x + b` to `y(x)`.

    Args:
        x (npt.ArrayLike): Independent variable (e.g. time)
        y (npt.ArrayLike): Dependent variable (e.g. energy)

    Returns:
        tuple[npt.ArrayLike, float]: Tuple of (f(x), a)
    """
    # Fit a first-order polynomial, p are the coefficients
    # f(x) = p[0] * x**(deg) + p[1] * x**(deg - 1) + ... + p[deg]
    p = np.polyfit(x, y, deg=1)
    # Evaluate the polynomial at each value in t
    g = np.polyval(p, x)
    # Return fitted function evalution and slope
    return (g, p[0])

def plot_solution(
    exact: dict[str, npt.ArrayLike],
    approximate: ty.Union[None, dict[str, npt.ArrayLike]] = None,
    solver_name: ty.Union[None, str] = None,
    plot_filename: ty.Union[None, str] = None
):
    """Plot the solution from a solver against the exact solution. Exact values are plotted
    as solid lines while the solver's output is plotted using points. Position and velocity
    share the left-hand y-axis, and energy uses the right-hand y-axis.

    Args:
        exact (dict[str, npt.ArrayLike]): Exact results.
        approximate (ty.Union[None, dict[str, npt.ArrayLike]]): Approximate results.
        solver_name (ty.Union[None, str]): Name of the approximate solver (used for plot
            title).
        plot_filename (ty.Union[None, str]): Filename for saving the plot. Should end in
            ".png", ".jpg", ".pdf", or another common image file extension.
    """
    # Define some constants
    position_color = 'tab:red'
    velocity_color = 'tab:green'
    energy_color = 'tab:blue'
    approximate_marker = '.'
    # Make a new figure and axis
    fig, axis = plt.subplots(figsize=(10,5))
    trajectory_ax = axis  # We'll use the main axis to plot positions and velocities
    # Make a twin axis sharing the same x-axis for the energies
    energy_ax = trajectory_ax.twinx()
    # Plot the exact solution
    trajectory_ax.plot(exact['t'], exact['x'], c=position_color, label='exact x(t)')
    trajectory_ax.plot(exact['t'], exact['v'], c=velocity_color, label='exact v(t)')
    energy_ax.plot(exact['t'], exact['E'], c=energy_color, label='exact E(t)')
    # Plot the approximate solution if given
    if approximate is not None:
        trajectory_ax.scatter(approximate['t'], approximate['x'],
                              c=position_color, marker=approximate_marker,
                              label='approx. x(t)')
        trajectory_ax.scatter(approximate['t'], approximate['v'],
                              c=velocity_color, marker=approximate_marker,
                              label='approx. v(t)')
        energy_ax.scatter(approximate['t'], approximate['E'],
                          c=energy_color, marker=approximate_marker,
                          label='approx. E(t)')
    # Fix up and label the axes
    trajectory_ax.set_xlabel('time')
    trajectory_ax.set_ylabel('position,velocity')
    trajectory_ax.autoscale(axis='x', tight=True)
    energy_ax.set_xlabel('time')
    energy_ax.set_ylabel('energy')
    energy_ax.autoscale(axis='x', tight=True)
    # Add a figure title and legend
    title = solver_name.upper() if solver_name is not None else "EXACT"
    fig.suptitle(title)
    fig.legend(loc='upper center', ncols=6, bbox_to_anchor=(0.5, 0.95))
    if plot_filename:
        fig.savefig(plot_filename)
    plt.show()
