import typing as ty
from numba import jit, float64
import numpy as np

from distance import compute_squared_distances
from lennard_jones import compute_forces
from measures import compute_e_kin

@jit((float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64),
     nopython=True, parallel=True)
def verlet(pos: np.ndarray, vel: np.ndarray, forces: np.ndarray,
           shift: np.ndarray, box_len: float, r_cut: float, dt: float) -> ty.Tuple[
               np.ndarray, np.ndarray]:
    """Integrate a system in the microcanonical (NVE) ensemble using Verlet integration.

    v(t + 1/2 dt) = v(t) + 1/2 a(t) dt
    x(t + dt) = x(t) + v(t + 1/2 dt) dt
    a(t + dt) = f(x(t + dt))
    v(t + dt) = v(t + 1/2 dt) + 1/2 a(t + dt) dt

    Args:
        pos (np.ndarray): (n_atoms x 3) array of positions.
        vel (np.ndarray): (n_atoms x 3) array of velocities.
        forces (np.ndarray): (n_atoms x 3) array of forces.
        shift (np.ndarray): (n_atoms x 3) array of PBC shifts.
        box_len (float): Cubic box length.
        r_cut (float): Radial cutoff.
        dt (float): Time step.

    Returns:
        ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            pos (np.ndarray): (n_atoms x 3) array of positions after the integration step.
            vel (np.ndarray): (n_atoms x 3) array of velocities after the integration step.
            forces (np.ndarray): (n_atoms x 3) array of forces after the integration step.
            shift (np.ndarray): (n_atoms x 3) array of PBC shifts after the integration step.
            dist2 (np.ndarray): (n_atoms x n_atoms) array of squared euclidean distances
                between atoms (upper triangle only) after the integration step.

    """
    # f = ma; given m = 1, a = f
    acc = forces
    # v(t + 1/2 dt) = v(t) + 1/2 a(t) dt
    vel = vel + 0.5 * acc * dt
    # x(t + dt) = x(t) + v(t + 1/2 dt) dt
    pos = pos + vel * dt
    # Track and enforce PBCs
    shift = shift + pos // box_len
    pos = pos % box_len
    # f(t + dt)
    dist2 = compute_squared_distances(pos, box_len)
    forces = compute_forces(pos, dist2, box_len, r_cut)
    acc = forces
    # v(t + dt) = v(t + 1/2 dt) + 1/2(a(t) + a(t + dt)) dt
    vel += 0.5 * acc * dt

    return pos, vel, forces, shift, dist2

@jit((float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64,
      float64, float64, float64, float64), nopython=True, parallel=False)
def nose_hoover(pos: np.ndarray, vel: np.ndarray, forces: np.ndarray,
                shift: np.ndarray, box_len: float, r_cut: float, dt: float,
                temp: float, eff_mass: float, frict: float, ln_s: float) -> ty.Tuple[
                    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
                ]:
    """Integrate a system in the canonical (NVT) ensemble using a Nosé-Hoover thermostat.

    Args:
        pos (np.ndarray): (n_atoms x 3) array of positions.
        vel (np.ndarray): (n_atoms x 3) array of velocities.
        forces (np.ndarray): (n_atoms x 3) array of forces.
        shift (np.ndarray): (n_atoms x 3) array of PBC shifts.
        box_len (float): Cubic box length.
        r_cut (float): Radial cutoff.
        dt (float): Time step.
        temp (float): Target temperature in energy units.
        eff_mass (float): Nosé-Hoover effective mass Q.
        frict (float): Nosé-Hoover friction ξ.
        ln_s (float): Natural logarithm of the Nosé-Hoover scaling coefficient ln(s).

    Returns:
        ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
            pos (np.ndarray): (n_atoms x 3) array of positions after the integration step.
            vel (np.ndarray): (n_atoms x 3) array of velocities after the integration step.
            forces (np.ndarray): (n_atoms x 3) array of forces after the integration step.
            shift (np.ndarray): (n_atoms x 3) array of PBC shifts after the integration step.
            dist2 (np.ndarray): (n_atoms x n_atoms) array of squared euclidean distances
                between atoms (upper triangle only) after the integration step.
            frict (float): Nosé-Hoover friction ξ after the integration step.
            ln_s (float): Natural logarithm of the Nosé-Hoover scaling coefficient ln(s)
                after the integration step.

    """
    # ξ = xi = frict                   | friction strength
    # β = 1 / kT = inv_temp            | inverse temperature in energy units
    # Q = eff_mass                     | effective mass
    # v = p / m                        | velocity = momentum / mass
    # a = -f - xi * p                  | acceleration = -force - friction * momentum
    # dxi / dt = (2 E_kin - g kT) / Q  | rate of change of friction strength
    # ds / dt / s = d ln(s) / dt = ξ  |

    n_atoms = pos.shape[0]
    # Number of degrees of freedom (g) is 3N because we use the real-variable formulation
    # of Nosé-Hoover (pp. 139 in md-thermostats.pdf)
    dof = 3 * n_atoms
    # a(t) = f_lj(t) + f_nh(t)
    acc = forces - frict * vel
    # x(t + dt) = x(t) + v(t) dt + 1/2 a(t) dt**2
    pos = pos + vel * dt + 0.5 * acc * dt**2
    # Track and enforce PBCs
    shift = shift + pos // box_len
    pos = pos % box_len
    # v(t + 1/2 dt) = v(t) + 1/2 a(t) dt
    vel = vel + 0.5 * acc * dt
    # f_lj(t + dt)
    dist2 = compute_squared_distances(pos, box_len)
    forces = compute_forces(pos, dist2, box_len, r_cut)
    # E_kin(t + dt)
    e_kin = compute_e_kin(vel)
    # dxi/dt(t + dt/2) = (2 E_kin(t + dt/2) - g kT) / Q
    dfrict_dt = (2 * e_kin - dof * temp) / eff_mass
    # Recall that dln(s)/dt = xi. We update ln(s) using the friction xi and its derivative:
    # ln(s)(t + dt/2) = ln(s)(t) + xi(t + dt/2) dt + 1/2 dxi_dt(t + dt/2) dt^2
    ln_s += frict * dt + 0.5 * dfrict_dt * dt**2
    # xi(t + dt/2) = xi(t) + 1/2 dxi/dt(t + dt/2) dt
    frict += 0.5 * dfrict_dt * dt
    # a(t + dt/2) = f_lj(t + dt) + f_nh(t + dt/2)
    acc = forces - frict * vel
    # v(t + dt) = v(t + dt / 2) + 1/2 acc(t + dt/2) dt
    vel = vel + 0.5 * acc * dt
    # E_kin(t + dt) = 1/2 v(t + dt)^2
    e_kin = compute_e_kin(vel)
    # dxi/dt(t + dt) = (2 E_kin(t + dt) - g kT) / Q
    dfrict_dt = (2 * e_kin - dof * temp) / eff_mass
    # xi(t + dt) = xi(t + dt/2) + 1/2
    frict += 0.5 * dfrict_dt * dt

    return pos, vel, forces, shift, dist2, frict, ln_s
