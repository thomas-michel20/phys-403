# -*- coding: utf-8 -*-
"""Interatomic potentials."""
from numba import float64
from numba.experimental import jitclass
import numpy as np


@jitclass([('sigma', float64), ('epsilon', float64), ('r_cut', float64)])
class LennardJones:
    """Lennard-Jones potential."""
    sigma: float
    epsilon: float
    r_cut: float

    def __init__(self, sigma: float, epsilon: float, r_cut: float):
        """Initialize Lennard-Jones potential.

        Args:
            sigma (float): Length scale parameter σ.
            epsilon (float): Energy scale parameter ε.
            r_cut (float): Cutoff radius.
        """
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_cut = r_cut

    def copy(self) -> 'LennardJones':
        """Make a deep copy of this potential.

        Returns:
            LennardJones: Copy of this potential.
        """
        return LennardJones(self.sigma, self.epsilon, self.r_cut)

    @property
    def r_cut_squared(self) -> float:
        """Square of the radial cutoff."""
        return self.r_cut * self.r_cut

    @property
    def potential_energy_tail(self) -> float:
        """Tail correction of the potential due to the radial cutoff.

            V_{tail}(r_cut) = 2π 4ε σ^3 (1/9 (σ / r_cut)^9 - 1/3 (σ / r_cut)^3)

        Returns:
            float: Potential energy tail correction.
        """
        x = (self.sigma / self.r_cut)**3
        return 2 * np.pi * 4 / 3 * self.epsilon * self.sigma**3 * (1 / 3 * x**3 - x)

    def potential_energy(self, r: float) -> float:
        """Potential energy evaluated at radial coordinate `r`.

            V(r) = 4ε ((σ / r)^12 - (σ / r)^6)

        Args:
            r (float): Radial coordinate.

        Returns:
            float: Potential energy (without tail correction).
        """
        x = (self.sigma / r)**6
        return 4 * self.epsilon * (x**2 - x)

    def force(self, r: float) -> float:
        """Force magnitude evaluated at radial coordinate `r`.

            F(r) = dV/dr(r) = 4ε (12 (σ / r)^12 - 6 (σ / r)^6) / r

        Args:
            r (float): Radial coordinate.

        Returns:
            float: Force magnitude.
        """
        x = (self.sigma / r)**6
        return 24 * self.epsilon * (2 * x**2 - x) / r
