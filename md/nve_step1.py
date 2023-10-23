# %%
import numpy as np
import matplotlib.pyplot as plt

from structure import build_fcc, init_velocities
from nve import run_nve
from measures import compute_rdf, compute_structure_factor_FT, compute_structure_factor_direct

import MD_old
# %%
n_cells = 6
lat_par = 1.7048
temp = 0.7867
r_cut = 2.5
dt = 0.0046
n_steps = 200
log_every = 1
target_temp = 0.0
n_bins = 300

box_len, pos = build_fcc(n_cells=n_cells, lat_par=lat_par)
vel = init_velocities(n_atoms=pos.shape[0], temp=temp)
# %%
%%time
n_steps = 100
hist_1, traj_1 = run_nve(pos, vel, box_len, r_cut, dt, n_steps,
                         log_every=log_every, target_temp=target_temp)

n_steps = 2000
hist_2, traj_2 = run_nve(traj_1['pos'][-1], traj_1['vel'][-1], box_len, r_cut, dt, n_steps,
                         log_every=log_every, target_temp=target_temp)

# %%
%%time
n_steps = 100
old_hist_1 = MD_old.run_NVE(pos, vel, box_len, n_steps, pos.shape[0], dt, Nbins=n_bins,
                            r_cutoff=r_cut)
n_steps = 2000
old_hist_2 = MD_old.run_NVE(pos, vel, box_len, n_steps, pos.shape[0], dt, Nbins=n_bins,
                            r_cutoff=r_cut)
# %%
fig, ax = plt.subplots()
ax.plot(hist_2['iter'], hist_2['e_pot'])
ax.plot(hist_2['iter'], hist_2['e_kin'])
ax.plot(hist_2['iter'], hist_2['e_pot'] + hist_2['e_kin'])
# %%
r, g = compute_rdf(traj_2['pos'], box_len, n_bins)
fig, ax = plt.subplots()
ax.plot(r, g)
# %%
q, S = compute_structure_factor_FT(traj_2['pos'], box_len, n_bins=n_bins, n_q=n_bins)
fig, ax = plt.subplots()
ax.plot(q, S)
# %%
%%time
q, S = compute_structure_factor_direct(traj_2['pos'], box_len, k_mesh=30, n_q_max=300,
                                       q_max=30.0)
fig, ax = plt.subplots()
ax.plot(q, S)

# %%
%%time
old_output = MD_old.run_NVE(pos, vel, box_len, n_steps, pos.shape[0], dt, Nbins=n_bins, r_cutoff=r_cut)

fig, ax = plt.subplots()
ax.plot(old_output['gofr']['r'], old_output['gofr']['g'])

fig, ax = plt.subplots()
ax.plot(old_output['nsteps'], old_output['EnPot'])
ax.plot(old_output['nsteps'], old_output['EnKin'])
ax.plot(old_output['nsteps'], old_output['EnPot'] + old_output['EnKin'])
