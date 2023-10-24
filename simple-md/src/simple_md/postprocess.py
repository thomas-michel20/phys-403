# -*- coding: utf-8 -*-
"""Postprocessing utilities."""
import typing as ty

from numba import float64, int64, jit, prange
import numpy as np

from .system import System


@jit(nopython=True)
def compute_msd(system: System, init_system: System) -> float:
    """Compute the mean squared displacement between two systems.

    MSD(t) = <(r_i(t) - r_i(t=0)^2>

    Args:
        system (System): System at some time t.
        init_system (System): The initial system at some time before t.

    Returns:
        float: Mean squared displacement.
    """
    r = system.absolute_positions()
    r_0 = init_system.absolute_positions()
    return np.mean(np.sum((r - r_0)**2, axis=1))


@jit(nopython=True)
def compute_vacf(system: System, init_system: System) -> float:
    """Compute the unnormalized velocity autocorrelation function between two systems.

    VACF(t) = 1/3 <v(t) . v(t=0)>

    Args:
        system (System): System at some time t.
        init_system (System): The initial system at some time before t.

    Returns:
        float: Unnormalized velocity autocorrelation function.
    """
    return np.mean(np.sum(system.velocities * init_system.velocities, axis=1)) / 3


def compute_rdf(traj: ty.Sequence[System], n_bins: int = 300):
    """Compute the radial distribution function for a MD trajectory.

    Args:
        traj (ty.Sequence[System]): List of frames in the trajectory.
        n_bins (int): Number of bins for the radial distribution function.

    Returns:
        ty.Tuple[np.ndarray, np.ndarray]: (n_bins,) vectors r (bin-centers) and g(r)
            (radial distribution function).
    """
    box_parameter = traj[0].box_parameter
    n_atoms = traj[0].n_atoms
    volume = traj[0].volume

    n_dists = n_atoms * (n_atoms - 1) / 2
    box_radius = box_parameter / 2
    bin_width = box_radius / n_bins

    rdf_sum = np.zeros(n_bins)
    for sys in traj:
        rdf_sum += sys.radial_distribution(n_bins)
    rdf_mean = rdf_sum / len(traj)

    r = bin_width * (np.arange(n_bins) + 0.5)
    g = volume / (n_dists * 4 * np.pi * bin_width) * rdf_mean / r**2

    return r, g


def compute_sf(r: np.ndarray, g: np.ndarray, density: float, n_q: int = 300, q_max: float = 32.0):
    """Compute the structure factor for an MD trajectory by Fourier transforming its
    radial distribution function.

    Args:
        r (np.ndarray): Domain of the radial distribution function.
        g (np.ndarray): Value of the radial distribution function.
        density (float): Density of the material.
        n_q (int, optional): Number of q-points. Defaults to 300.
        q_max (float, optional): Maximum value of q. Defaults to 32.0.

    Returns:
        ty.Tuple[np.ndarray, np.ndarray]: (n_q,) vectors q (bin-centers) and S(q)
        (structure factor).
    """
    return _compute_sf(r, g, density, n_q, q_max)


@jit((float64[:], float64[:], float64, int64, float64), nopython=True, parallel=True)
def _compute_sf(r: np.ndarray, g: np.ndarray, density: float, n_q: int, q_max: float):
    # Compute dr for use in the trapezoidal integration `np.trapz`.
    dr = r[1] - r[0]
    # r^2 (g - 1) is constant in the loop below, so we calculate it only once.
    r2_gm1 = r * r * (g - 1)
    # Define the domain of the structure factor.
    q = np.linspace(start=0.0, stop=q_max, num=n_q)
    # Create an array of zeros to fill with the structure factor.
    structure_factor = np.zeros(n_q)
    # Compute the integral expression for each value of `q`.
    for i_q in prange(n_q):  # pylint: disable=not-an-iterable
        # Integrand = r^2 (g - 1) sin(q r) / (q r)
        #           = r^2 (g - 1) sinc((q r) / pi), where
        # sinc(x) = sin(pi x) / (pi x)
        integrand = r2_gm1 * np.sinc(q[i_q] / np.pi * r)
        structure_factor[i_q] = 1 + 4 * np.pi * density * np.trapz(y=integrand, dx=dr)
    return q, structure_factor


def compute_velocity_component_density(
    system: System, v_min: float = -10.0, v_max: float = +10.0, n_bins: int = 300
) -> np.ndarray:
    """Compute a histogram of velocity components from `v_min` to `v_max` over `n_bins`
    bins.

    Args:
        system (System): System state.
        v_min (float, optional): Minimum of the histogram domain. Defaults to -10.0.
        v_max (float, optional): Maximum of the histogram domain. Defaults to +10.0.
        n_bins (int, optional): Number of histogram bins. Defaults to 300.

    Returns:
        np.ndarray: Density-normalized histogram of velocity components.
    """
    density, _ = np.histogram(system.velocities.flatten.flatten(), bins=n_bins, range=(v_min, v_max), density=True)
    return density


def compute_velocity_magnitude_density(system: System, v_magnitude_max: float = 10.0, n_bins: int = 300) -> np.ndarray:
    """Compute a histogram of velocity magnitudes up to `v_magnitude_max` over `n_bins`
    bins.

    Args:
        system (System): System state.
        v_magnitude_max (float, optional): Maximum of the histogram domain. Defaults to 10.0.
        n_bins (int, optional): Number of histogram bins. Defaults to 300.

    Returns:
        np.ndarray: Density-normalized histogram of velocity magnitudes.
    """
    velocity_magnitudes = np.sqrt(np.sum(system.velocities**2, axis=1))
    density, _ = np.histogram(velocity_magnitudes, bins=n_bins, range=(0.0, v_magnitude_max), density=True)
    return density
