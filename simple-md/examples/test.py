# %%
import numpy as np
from simple_md import structure, potential, system, integrator, postprocess

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

positions, box_parameter = structure.build_fcc(
    n_cells=n_cells,
    lattice_parameter=lattice_parameter
)
velocities = np.zeros_like(positions)
shifts = np.zeros_like(positions)
potential = potential.LennardJones(sigma, epsilon, r_cut)

system = system.System(box_parameter, atomic_mass, positions, velocities, shifts, potential)
system.initialize_velocities(temperature)

# accelerations = (system.forces() - friction_coefficient * system.velocities) / system.atomic_mass
# integrator = integrator.NoseHoover(dt, accelerations, temperature, effective_mass, friction_coefficient, ln_s)

accelerations = system.forces() / system.atomic_mass
integrator = integrator.VelocityVerlet(dt, accelerations)
# %%
%%time
time = []
traj = []
for i in range(2000):
    time.append(i * integrator.dt)
    traj.append(system.copy())
    integrator.step(system)
# %%
%%time
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9,4))

K = np.array([sys.kinetic_energy() for sys in traj])
V = np.array([sys.potential_energy() for sys in traj])
E = K + V
axes[0].plot(time, K)
axes[0].plot(time, V)
axes[0].plot(time, K + V)

T = np.array([sys.instantaneous_temperature() for sys in traj])
axes[1].plot(time, T)

# %%
%%time
fig, axes = plt.subplots(1, 2, figsize=(9,4))

r, g = postprocess.compute_rdf(traj)
axes[0].plot(r, g)

q, S = postprocess.compute_sf(r, g, system.volume)
axes[1].plot(q, S)
# %%
