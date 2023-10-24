# -*- coding: utf-8 -*-
"""Time integrators."""
from numba import float64
from numba.experimental import jitclass
import numpy as np

from .system import System


@jitclass([('dt', float64), ('accelerations', float64[:, :])])
class VelocityVerlet:  # pylint: disable=too-few-public-methods
    """Velocity-Verlet integrator."""
    dt: float
    accelerations: np.ndarray

    def __init__(self, dt: float, accelerations: np.ndarray):
        """Initialize a Velocity-Verlet integrator.

        Args:
            dt (float): Timestep.
            accelerations (np.ndarray): (n_atoms, 3) array of accelerations.
        """
        self.dt = dt
        self.accelerations = accelerations

    def step(self, system: System):
        """Make an integration step.

            [a(t) = F(r(t)) / m]  # This is saved in the integrator.

            v(t + 1/2 dt) = 1/2 a(t) dt

            r(t + dt) = r(t) + v(t) + v(t + 1/2 dt) dt

            a(t + dt) = f(r(t + dt)) / m

            v(t + dt) = v(t + 1/2 dt) + 1/2 a(t + dt) dt

        Args:
            system (System): System to integrate.

        Returns:
            float: Total potential energy
        """
        # Update velocities to t + 1/2 dt
        system.velocities += self.accelerations * 0.5 * self.dt
        # Update positions to t + dt
        system.positions += system.velocities * self.dt
        # Update shifts (atoms passing through boundaries)
        system.shifts += system.positions // system.box_parameter
        # Wrap positions back into the box
        system.positions %= system.box_parameter
        # Compute accelerations at t + dt
        forces, potential_energy = system.forces_and_potential_energy()
        self.accelerations = forces / system.atomic_mass
        # Update velocities to t + dt
        system.velocities += self.accelerations * 0.5 * self.dt
        return potential_energy


@jitclass([('dt', float64), ('accelerations', float64[:, :]), ('temperature', float64), ('effective_mass', float64),
           ('friction_coefficient', float64), ('ln_s', float64)])
class NoseHoover:  # pylint: disable=too-few-public-methods
    """Nosé-Hoover thermostat integrator."""
    dt: float
    accelerations: np.ndarray
    temperature: float
    effective_mass: float
    friction_coefficient: float
    ln_s: float

    def __init__(
        self, dt: float, accelerations: np.ndarray, temperature: float, effective_mass: float,
        friction_coefficient: float, ln_s: float
    ):  # pylint: disable=too-many-arguments
        """Initialize a Nose-Hoover thermostat.

        Args:
            dt (float): Timestep.
            accelerations (np.ndarray): (n_atoms, 3) array of accelerations.
            temperature (float): Temperature.
            effective_mass (float): Nose-Hoover effective mass Q.
            friction_coefficient (float): Nose-Hoover friction coefficient ξ.
            ln_s (float): Natural logarithm of the Nose-Hoover parameter s.
        """
        self.dt = dt
        self.accelerations = accelerations
        self.temperature = temperature
        self.effective_mass = effective_mass
        self.friction_coefficient = friction_coefficient
        self.ln_s = ln_s

    def step(self, system: System):
        """Make an integration step.

            [a(t) = (F(t) - ξ(t) v(t)) / m]  # This is saved in the integrator.

            [ξ(t)]  # This is saved in the integrator.

            [ln(s(t))]  # This is saved in the integrator.

            r(t + dt) = r(t) + v(t) dt + 1/2 a(t) dt^2

            v(t + 1/2 dt) = v(t) + 1/2 a(t) dt

            dξ/dt(t + 1/2 dt) = (2 K(v(t + 1/2 dt)) - g T) / Q

            ln(s(t + 1/2 dt)) = ξ(t) dt + 1/2 dξ/dt(t + 1/2 dt) dt^2

            ξ(t + 1/2 dt) = ξ(t) + 1/2 dξ/dt(t + 1/2 dt) dt

            a(t + dt) = (f(r(t + dt)) - ξ(t + 1/2 dt) v(t + 1/2 dt)) / m

            v(t + dt) = v(t + 1/2 dt) + 1/2 a(t + dt) dt

            dξ/dt(t + dt) = (2 K(v(t + dt)) - g T) / Q

            ξ(t + dt) = 1/2 ξ(t + 1/2 dt) dt

        Args:
            system (System): System to integrate.

        Returns:
            float: Total potential energy.
        """
        # Update positions to t + dt
        system.positions += system.velocities * self.dt + 0.5 * self.accelerations * self.dt**2
        # Update shifts (atoms passing through boundaries)
        system.shifts += system.positions // system.box_parameter
        # Wrap positions back into the box
        system.positions %= system.box_parameter
        # Update velocities to t + 1/2 dt
        system.velocities += self.accelerations * 0.5 * self.dt
        # Compute the time derivative of the friction coefficient at t + 1/2 dt
        dfriction_dt = (
            2 * system.kinetic_energy() - system.degrees_of_freedom * self.temperature
        ) / self.effective_mass
        # Update the natural logarithm of the Nose-Hoover parameter to t + dt
        self.ln_s += self.friction_coefficient * self.dt + 0.5 * dfriction_dt * self.dt**2
        # Update the friction coefficient to t + 1/2 dt
        self.friction_coefficient += 0.5 * dfriction_dt * self.dt
        # Update the accelerations to t + dt
        forces, potential_energy = system.forces_and_potential_energy()
        self.accelerations = (forces - self.friction_coefficient * system.velocities) / system.atomic_mass
        # Update the velocities to t + dt
        system.velocities += 0.5 * self.accelerations * self.dt
        # Compute the time derivative of the friction coefficient at t + dt
        dfriction_dt = (
            2 * system.kinetic_energy() - system.degrees_of_freedom * self.temperature
        ) / self.effective_mass
        # Update the friction coefficient to t + dt
        self.friction_coefficient += 0.5 * dfriction_dt * self.dt
        return potential_energy
