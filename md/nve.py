from numba import jit, float64, int64
import numpy as np
import matplotlib.pyplot as plt

from distance import compute_squared_distances
from lennard_jones import compute_forces, compute_e_pot
from integrators import verlet
from measures import compute_e_kin, compute_temp, compute_msd, compute_vacf

@jit((float64[:,:], float64[:,:], float64, float64, float64, int64, int64, float64), nopython=True)
def run_nve(
    pos: np.ndarray,
    vel: np.ndarray,
    box_len: float,
    r_cut: float,
    dt: float,
    n_steps: int,
    log_every: int = 1,
    target_temp: float = -1.0
):
    """Run an MD simulation in the microcanonical (NVE) ensemble using Verlet integration.

    The positions, velocities, and other derived quantities will be recorded every
    `log_every` steps, including the first step. Internally, `n_steps + 1` steps are truly
    run so that, for example, `n_steps = 200` and `log_every = 100` will return logs before
    the first Verlet step, after the 100th step, and after the 200th step.

    Args:
        pos (np.ndarray): (n_atoms x 3) array of positions.
        vel (np.ndarray): (n_atoms x 3) array of velocities.
        box_len (float): Edge length of the cubic supercell.
        r_cut (float): Radial cutoff for inter-atomic potential evaluation.
        dt (float): Time step.
        n_steps (int): Number of time steps to simulate.
        log_every (int, optional): Log every this many steps. Defaults to 1.
        target_temp (float, optional): Target temperature for applying velocity rescaling.
            Defaults to -1.0.

    Returns:
        history (dict):
            'iter': (n_logs,) vector of iteration indices.
            'time': (n_logs,) vector of times.
            'e_pot': (n_logs,) vector of potential energies.
            'e_kin': (n_logs,) vector of total energies.
            'temp': (n_logs,) vector of temperatures.
            'vacf': (n_logs,) vector of velocity autocorrelation values.
            'msd': (n_logs,) vector of mean squared displacement values.
        trajectory (dict):
            'pos': (n_logs x n_atoms x 3) position history array.
            'vel': (n_logs x n_atoms x 3) velocity history array.

    """
    pos_init = pos.copy()
    vel_init = vel.copy()

    dist2 = compute_squared_distances(pos, box_len)
    forces = compute_forces(pos, dist2, box_len, r_cut)
    shift = np.zeros_like(pos)

    i_log = 0
    n_log = n_steps // log_every + 1
    _iter = np.empty(n_log)
    time = np.empty(n_log)
    e_pot = np.empty(n_log)
    e_kin = np.empty(n_log)
    temp = np.empty(n_log)
    vacf = np.empty(n_log)
    msd = np.empty(n_log)
    pos_traj = np.empty((n_log, pos.shape[0], pos.shape[1]))
    vel_traj = np.empty((n_log, vel.shape[0], vel.shape[1]))

    for i_iter in range(n_steps + 1):
        if i_iter % log_every == 0:
            _iter[i_log] = i_iter
            time[i_log] = i_iter * dt
            e_pot[i_log] = compute_e_pot(pos, dist2, r_cut)
            e_kin[i_log] = compute_e_kin(vel)
            temp[i_log] = compute_temp(vel)
            vacf[i_log] = compute_vacf(vel, vel_init)
            msd[i_log] = compute_msd(pos, pos_init, shift, box_len)
            pos_traj[i_log, :, :] = pos
            vel_traj[i_log, :, :] = vel
            i_log += 1

        pos, vel, forces, shift, dist2 = verlet(pos, vel, forces, shift, box_len, r_cut, dt)

        if target_temp > 0.0:
            vel *= np.sqrt(3 * pos.shape[0] * target_temp / (2 * compute_e_kin(vel)))

    history = {
        'iter': _iter,
        'time': time,
        'e_pot': e_pot,
        'e_kin': e_kin,
        'temp': temp,
        'vacf': vacf,
        'msd': msd,
    }

    trajectory = {
        'pos': pos_traj,
        'vel': vel_traj,
    }

    return (history, trajectory)

# Define a function to help us visualize different quantities from the NVE MD run.
def plot_nve(hist: dict, traj: dict):
    """Plot the kinetic, potential, and total energy; the energies per atom; and the
    temperature from an NVE simulation as functions of time.

    Args:
        hist (dict): Simulation history dictionary.
        traj (dict): Simulation trajectory dictionary.

    Returns:
        fig: matplotlib.pyplot.Figure
        axes: List[matplotlib.pyplot.Axis]
    """
    n_atoms = traj['pos'][0].shape[0]
    fig, axes = plt.subplots(2, 2)
    # Plot the energies
    axes[0,0].plot(hist['time'], hist['e_kin'], label=r'$E_{kin}$')
    axes[0,0].plot(hist['time'], hist['e_pot'], label=r'$E_{pot}$')
    axes[0,0].plot(hist['time'], hist['e_kin'] + hist['e_pot'], label=r'$E_{tot}$')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Energy')
    # Plot the energies per atom
    axes[0,1].plot(hist['time'], hist['e_kin'] / n_atoms, label=r'$E_{kin}$')
    axes[0,1].plot(hist['time'], hist['e_pot'] / n_atoms, label=r'$E_{pot}$')
    axes[0,1].plot(hist['time'], (hist['e_kin'] + hist['e_pot']) / n_atoms, label=r'$E_{tot}$')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Energy / Atom')
    axes[0,1].legend()
    # Plot the temperature over time
    axes[1,0].plot(hist['time'], hist['temp'], label=r'$T$')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Temperature')
    # Disable the last, unused axis
    axes[1,1].set_visible(False)
    # Make things prettier
    fig.tight_layout()
    # Return the figure
    return fig, axes
