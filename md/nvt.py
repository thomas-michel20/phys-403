from numba import jit, float64, int64
import numpy as np

from distance import compute_squared_distances
from lennard_jones import compute_forces, compute_e_pot
from integrators import nose_hoover
from measures import compute_e_kin, compute_e_nh, compute_temp, compute_msd, compute_vacf

@jit((float64[:,:], float64[:,:], float64, float64, float64, float64, float64, float64,
     float64, int64, int64), nopython=True)
def run_nvt(
    pos: np.ndarray,
    vel: np.ndarray,
    box_len: float,
    r_cut: float,
    dt: float,
    temp: float,
    eff_mass: float,
    frict: float,
    ln_s: float,
    n_steps: int,
    log_every: int = 1
):
    """Run an MD simulation in the canonical (NVT) ensemble using a Nose-Hoover thermostat.

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
        temp (float): Target temperature for the thermostat.
        eff_mass (float): Nose-Hoover effective mass Q.
        frict (float): Nose-Hoover friction ξ (xi).
        ln_s (float): Natural logarithm of the Nose-Hoover parameter s.
        n_steps (int): Number of time steps to simulate.
        log_every (int, optional): Log every this many steps. Defaults to 1.

    Returns:
        history (dict):
            'iter': (n_logs,) vector of iteration indices.
            'time': (n_logs,) vector of times.
            'e_pot': (n_logs,) vector of potential energies.
            'e_kin': (n_logs,) vector of total energies.
            'e_nh': (n_logs,) vector of Nose-Hoover energies.
            'frict': (n_logs,) vector of Nose-Hover friction values.
            'ln_s': (n_logs,) vector of Nose-Hoover ln(s) values.
            'temp': (n_logs,) vector of temperatures.
            'vacf': (n_logs,) vector of velocity autocorrelation values.
            'msd': (n_logs,) vector of mean squared displacement values.
        trajectory (dict):
            'pos': (n_logs x n_atoms x 3) position history array.
            'vel': (n_logs x n_atoms x 3) velocity history array.

    """
    pos_init = pos.copy()
    vel_init = vel.copy()

    dof = pos.shape[0] * pos.shape[1]

    dist2 = compute_squared_distances(pos, box_len)
    forces = compute_forces(pos, dist2, box_len, r_cut)
    shift = np.zeros_like(pos)

    i_log = 0
    n_log = n_steps // log_every + 1
    _iter = np.empty(n_log)
    time = np.empty(n_log)
    e_pot = np.empty(n_log)
    e_kin = np.empty(n_log)
    e_nh = np.empty(n_log)
    frict_hist = np.empty(n_log)
    ln_s_hist = np.empty(n_log)
    temp_hist = np.empty(n_log)
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
            e_nh[i_log] = compute_e_nh(dof, temp, frict, ln_s, eff_mass)
            frict_hist[i_log] = frict
            ln_s_hist[i_log] = ln_s
            temp_hist[i_log] = compute_temp(vel)
            vacf[i_log] = compute_vacf(vel, vel_init)
            msd[i_log] = compute_msd(pos, pos_init, shift, box_len)
            pos_traj[i_log, :, :] = pos
            vel_traj[i_log, :, :] = vel
            i_log += 1

        pos, vel, forces, shift, dist2, frict, ln_s = nose_hoover(pos, vel, forces, shift,
                                                                  box_len, r_cut, dt, temp,
                                                                  eff_mass, frict, ln_s)

    history = {
        'iter': _iter,
        'time': time,
        'e_pot': e_pot,
        'e_kin': e_kin,
        'e_nh': e_nh,
        'frict': frict_hist,
        'ln_s': ln_s_hist,
        'temp': temp_hist,
        'vacf': vacf,
        'msd': msd,
    }

    trajectory = {
        'pos': pos_traj,
        'vel': vel_traj,
    }

    return (history, trajectory)
