# -*- coding: utf-8 -*-
"""Molecular dynamics system."""
from math import sqrt
import typing as ty

from numba import float64, prange
from numba.experimental import jitclass
import numpy as np

from .potential import LennardJones


@jitclass([
    ('box_parameter', float64),
    ('atomic_mass', float64),
    ('positions', float64[:, :]),
    ('velocities', float64[:, :]),
    ('shifts', float64[:, :]),
])
class System:
    """Atomic system inside a cubic box with periodic boundary conditions."""
    box_parameter: float
    atomic_mass: float
    positions: np.ndarray
    velocities: np.ndarray
    shifts: np.ndarray
    potential: LennardJones

    def __init__(
        self, box_parameter: float, atomic_mass: float, positions: np.ndarray, velocities: np.ndarray,
        shifts: np.ndarray, potential: LennardJones
    ):  # pylint: disable=too-many-arguments
        """Initialize a system of identical atoms with in a periodic cubic box.

        Args:
            box_parameter (float): Side length of the cubic box.
            atomic_mass (float): Atomic mass.
            positions (np.ndarray): (n_atoms, 3) array of atomic positions.
            velocities (np.ndarray): (n_atoms, 3) array of velocities.
            shifts (np.ndarray): (n_atoms, 3) array of periodic shifts for tracking true
            atomic displacements.
            potential (LennardJones): Lennard-Jones pairwise interatomic potential.
        """
        self.box_parameter = box_parameter
        self.atomic_mass = atomic_mass
        self.positions = positions
        self.velocities = velocities
        self.shifts = shifts
        self.potential = potential

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return self.positions.shape[0]

    @property
    def volume(self) -> float:
        """Box volume.

            V = a^3
        """
        return self.box_parameter**3

    @property
    def density(self) -> float:
        """Density in units of atoms / volume.

            ρ = N / V
        """
        return self.n_atoms / self.volume

    @property
    def degrees_of_freedom(self) -> int:
        """Number of degrees of freedom.

            g = 3 N
        """
        return 3 * self.n_atoms

    def copy(self) -> 'System':
        """Return a deep copy of the system.

        Returns:
            System: Copy of the system.
        """
        return System(
            self.box_parameter, self.atomic_mass, self.positions.copy(), self.velocities.copy(), self.shifts.copy(),
            self.potential.copy()
        )

    def absolute_positions(self) -> np.ndarray:
        """Return absolute positions (accounting for the recorded shift).

            r_{i, abs} = r_i + a * shift_i

        Returns:
            np.ndarray: (n_atoms, 3) array of absolute atomic positions (possibly outside
            the box).
        """
        return self.positions + self.box_parameter * self.shifts

    def minimum_image_diff(self, i: int, j: int) -> ty.Tuple[float, float, float]:
        """Compute the vector from position `i` to position `j` in the minimum image convention.

            dr = (r_i - r_j) - a round((r_i - r_j) / a)

        Args:
            i (int): First position index.
            j (int): Second position index.
            dr (np.ndarray): Difference vector.
        """
        # N.B. This code is in-lined and uses a tuple return value for performance.
        #      It should not allocate any memory, and it avoids bounds-checking overhead
        #      from numpy.
        # dr_{ij} = r_{i} - r_{j}
        dx = self.positions[i, 0] - self.positions[j, 0]
        dy = self.positions[i, 1] - self.positions[j, 1]
        dz = self.positions[i, 2] - self.positions[j, 2]
        # Apply minimum image convention
        dx -= self.box_parameter * np.rint(dx / self.box_parameter)
        dy -= self.box_parameter * np.rint(dy / self.box_parameter)
        dz -= self.box_parameter * np.rint(dz / self.box_parameter)

        return (dx, dy, dz)

    def distance_squared_pbc(self, i: int, j: int) -> float:
        """Compute the squared distance between positions `i` and `j` in the minimum
        image convention (avoiding sqrt for performance).

        Args:
            i (int): First position index.
            j (int): Second position index.

        Returns:
            float: Distance.
        """
        # N.B. This code is in-lined for performance. It should not allocate any memory,
        #      and it avoids bounds-checking overhead from numpy.
        dr = self.minimum_image_diff(i, j)
        return dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]

    def distance_pbc(self, i: int, j: int) -> float:
        """Compute the distance between positions `i` and `j` in the minimum image convention.

            r = ||dr_{min. img.}||_2

        Args:
            i (int): First position index.
            j (int): Second position index.

        Returns:
            float: Distance.
        """
        r_squared = self.distance_squared_pbc(i, j)
        return sqrt(r_squared)

    def distances_pbc(self) -> np.ndarray:
        """Compute the upper triangle of the distance matrix for all pairs of positions
        in the minimum image convention.

        Returns:
            np.ndarray: (n_atoms, n_atoms) distance matrix.
        """
        distances = np.zeros((self.n_atoms, self.n_atoms))
        for i in prange(self.n_atoms):  # pylint: disable=not-an-iterable
            for j in range(i + 1, self.n_atoms):
                distances[i, j] = self.distance_pbc(i, j)
        return distances

    def forces_and_potential_energy(self):
        """Compute the forces on all atoms of the system. Only force contributions for pairs
        of atoms whose minimum-image distance is less than the radial cutoff of the potential
        are computed. Potential energy is computed simultaneously at little overhead.

        Returns:
            np.ndarray: (n_atoms x 3) forces array.
            float: total potential energy.
        """
        forces = np.zeros_like(self.positions)
        potential_energy = self.potential.potential_energy_tail
        for i in prange(self.n_atoms):  # pylint: disable=not-an-iterable
            for j in range(i + 1, self.n_atoms):
                dr = self.minimum_image_diff(i, j)
                r = sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])
                if r < self.potential.r_cut:
                    # N.B. This code is in-lined for performance. It should not allocate
                    #      any memory,and it avoids bounds-checking overhead from numpy
                    #      except for possibly indexing into `forces`.
                    # Force magnitude
                    f = self.potential.force(r)
                    # force = force_magnitude * force_direction
                    force = (f * dr[0] / r, f * dr[1] / r, f * dr[2] / r)
                    # F[i] += force
                    forces[i, 0] += force[0]
                    forces[i, 1] += force[1]
                    forces[i, 2] += force[2]
                    # F[j] -= force
                    forces[j, 0] -= force[0]
                    forces[j, 1] -= force[1]
                    forces[j, 2] -= force[2]
                    # Potential energy
                    potential_energy += self.potential.potential_energy(r)
        return forces, potential_energy

    def potential_energy(self) -> float:
        """Compute the total potential energy of the system, summing over pairs of atoms whose
        minimum-image distance is less than the radial cutoff of the potentisl.

        Returns:
            float: Potential energy.
        """
        total_potential_energy = self.potential.potential_energy_tail
        for i in prange(self.n_atoms):  # pylint: disable=not-an-iterable
            for j in range(i + 1, self.n_atoms):
                r = self.distance_pbc(i, j)
                if r < self.potential.r_cut:
                    total_potential_energy += self.potential.potential_energy(r)
        return total_potential_energy

    def kinetic_energy(self) -> float:
        """Compute the total kinetic energy of all particles.

            K = 1/2 m v^2

        Returns:
            float: Total kinetic energy.
        """
        return 0.5 * self.atomic_mass * np.sum(self.velocities**2)

    def instantaneous_temperature(self) -> float:
        """Compute the instantaneous temperature of the system.

            T = 2 / g K

        Returns:
            float: Instantaneous temperature.
        """
        return 2 / self.degrees_of_freedom * self.kinetic_energy()

    def remove_translation(self):
        """Remove center of mass translation."""
        total_mass = self.atomic_mass * self.n_atoms
        center_of_mass_velocity = self.atomic_mass * np.sum(self.velocities, axis=0) / total_mass
        self.velocities -= center_of_mass_velocity

    def initialize_velocities(self, temperature: float):
        """Initialize velocities according to a Maxwell-Boltzmann distribution of ||v||_2
        with the given temperature.

        Args:
            temperature (float): Target temperature.
        """
        self.velocities = np.random.normal(loc=0.0, scale=sqrt(temperature), size=(self.n_atoms, 3))
        self.remove_translation()
        self.scale_temperature(temperature)

    def scale_temperature(self, temperature: float):
        """Scale all velocities to fulfil the given temperature.

        Args:
            temperature (float): Target temperature.
        """
        self.velocities *= sqrt(temperature / self.instantaneous_temperature())

    def radial_distribution(self, n_bins: int) -> np.ndarray:
        """Calculate the unnormalized radial distribution histogram.

        Args:
            n_bins (int): Number of bins for this distribution histogram.

        Returns:
            np.ndarray: Unnormalized radial distribution histogram.
        """
        box_radius = self.box_parameter / 2
        box_radius_squared = box_radius**2
        bin_width = box_radius / n_bins
        rdf = np.zeros(n_bins)
        for i in prange(self.n_atoms):  # pylint: disable=not-an-iterable
            for j in range(i + 1, self.n_atoms):
                r_squared = self.distance_squared_pbc(i, j)
                if r_squared < box_radius_squared:
                    bin_index = int(sqrt(r_squared) / bin_width)
                    rdf[bin_index] += 1.0
        return rdf
