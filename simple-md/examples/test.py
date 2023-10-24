# -*- coding: utf-8 -*-
# %%
import matplotlib.pyplot as plt
import numpy as np

from simple_md import integrator, postprocess, potential, structure, system

np.random.seed(40323)

n_cells = 6
lattice_parameter = 1.7048
atomic_mass = 1.0
sigma = 1.0
epsilon = 1.0
r_cut = 2.5
temperature = 0.7867
dt = 0.0046
effective_mass = 10.0
friction_coefficient = 0.0
ln_s = 0.0

positions, box_parameter = structure.build_fcc(n_cells=n_cells, lattice_parameter=lattice_parameter)
velocities = np.zeros_like(positions)
shifts = np.zeros_like(positions)
potential = potential.LennardJones(sigma, epsilon, r_cut)

sys = system.System(box_parameter, atomic_mass, positions, velocities, shifts, potential)
sys.initialize_velocities(temperature)

accelerations = (sys.forces_and_potential_energy()[0] - friction_coefficient * sys.velocities) / sys.atomic_mass
integrator = integrator.NoseHoover(dt, accelerations, temperature, effective_mass, friction_coefficient, ln_s)

# accelerations = sys.forces_and_potential_energy()[0] / sys.atomic_mass
# integrator = integrator.VelocityVerlet(dt, accelerations)
# %%
n_steps = 200
for i in range(n_steps):
    integrator.step(sys)

n_steps = 900
traj = []
time = np.zeros(n_steps)
V = np.zeros(n_steps)
for i in range(n_steps):
    traj.append(sys.copy())
    time[i] = i * integrator.dt
    V[i] = integrator.step(sys)
# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

K = np.array([sys.kinetic_energy() for sys in traj])
E = K + V
axes[0].plot(time, K / sys.n_atoms)
axes[0].plot(time, V / sys.n_atoms)
axes[0].plot(time, (K + V) / sys.n_atoms)

T = np.array([sys.instantaneous_temperature() for sys in traj])
axes[1].plot(time, T)
# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

r, g = postprocess.compute_rdf(traj[::10])
axes[0].plot(r, g)

q, S = postprocess.compute_sf(r, g, sys.density)
axes[1].plot(q, S)
# %%
# %%
