# %%
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

from structure import build_fcc, init_velocities
from nvt import run_nvt
from nve import run_nve
# %%
# Structure parameters
n_cells = 6          # Number of FCC unit cells along each lattice direction
lat_par = 1.7048     # Lattice parameter in Lennard-Jones length units

# Potential parameters
r_cut = 2.5          # Radial cutoff in Lennard-Jones length units

# NVE parameters
target_temp = 0.7807 # NVE target temperature
n_steps_nve = 200    # Number of NVE MD steps

# NVT parameters
eff_mass = 10.0      # Effective mass / thermal inertia Q
temp = 0.7807        # Temperature in Lennard-Jones energy units
frict = 0.0          # Initial friction parameter ξ
ln_s = 0.0           # Initial value of ln(s)
n_steps_nvt = 500    # Number of NVT MD steps

# MD parameters
dt = 0.0046      # Time step in Lennard-Jones units
log_every = 1    # Log positions, velocities, etc. every this many steps

# %%
box_len, pos = build_fcc(n_cells, lat_par)
n_atoms = pos.shape[0]
vel = init_velocities(n_atoms, temp)
# %% 1.0 Initialize a system using NVE
nve_hist, nve_traj = run_nve(
    pos, vel, box_len, r_cut, dt, n_steps_nve, log_every, target_temp
)

n_atoms = pos.shape[0]
fig, ax = plt.subplots()
ax.plot(nve_hist['time'], nve_hist['e_kin'] / n_atoms, label=r'$E_{kin}$')
ax.plot(nve_hist['time'], nve_hist['e_pot'] / n_atoms, label=r'$E_{pot}$')
ax.plot(nve_hist['time'], (nve_hist['e_kin'] + nve_hist['e_pot']) / n_atoms, label=r'$E_{pot+kin}$')

ax2 = ax.twinx()
ax2.plot(nve_hist['time'], nve_hist['temp'], label='Temperature', c='grey')
ax2.set_ylabel('Temperature')
ax2.set_ylim(target_temp + 0.1 * target_temp, target_temp - 0.1 * target_temp)

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Energy / Atom')
# %% 1.1 Run an NVT simulation
%%time
nvt_hist, nvt_traj = run_nvt(
    nve_traj['pos'][-1], nve_traj['vel'][-1], box_len, r_cut, dt, temp, eff_mass,
    frict, ln_s, n_steps_nvt, log_every
)

n_atoms = pos.shape[0]
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].plot(nvt_hist['time'], nvt_hist['e_kin'] / n_atoms, label=r'$E_{kin}$')
axes[0].plot(nvt_hist['time'], nvt_hist['e_pot'] / n_atoms, label=r'$E_{pot}$')
axes[0].plot(nvt_hist['time'], nvt_hist['e_nh'] / n_atoms, label=r'$E_{NH}$')
axes[0].plot(nvt_hist['time'], (nvt_hist['e_kin'] + nvt_hist['e_pot']) / n_atoms, label=r'$E_{pot+kin}$')
axes[0].plot(nvt_hist['time'], (nvt_hist['e_kin'] + nvt_hist['e_pot'] + nvt_hist['e_nh']) / n_atoms, label=r'$E_{tot}$')
axes[0].legend()
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Energy')

axes[1].plot(nvt_hist['time'], nvt_hist['temp'], label='Temperature', c='grey')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Temperature')

fig.tight_layout()
# %% 1.2 Study the fluctuations

fig, ax = plt.subplots()
ax.hist(nvt_hist['temp'], fc='none', histtype='step', density=True)
ax.set_xlabel('Temperature')
ax.set_ylabel(r'$PDF(T)$')

fig, axes = plt.subplots(1, 2, figsize=(9,4))

v_min = -5.0
v_max = +5.0
n_bins = 300
v = np.linspace(v_min, v_max, n_bins)
v_pdf, _ = np.histogram(
    nvt_traj['vel'].flatten(),
    bins=n_bins, range=(v_min, v_max), density=True
)
axes[0].scatter(v, v_pdf, marker='+', c='tab:red', s=20,
                label='NVT Simulation')
axes[0].plot(v, sps.norm.pdf(v, 0, np.sqrt(temp)), c='k',
             label=r'Normal$\left(\mu=0,\sigma=\sqrt{T}\right)$')
axes[0].set_xlabel(r'Velocity component $v_i$')
axes[0].set_ylabel(r'$PDF(v_i)$')
axes[0].legend()
axes[0].set_ylim(-0.1, 1.0)

v_norm_max = +5.0
n_bins = 300
v_norm = np.linspace(0, v_norm_max, n_bins)
v_norm_pdf, _ = np.histogram(
    np.linalg.norm(nvt_traj['vel'], axis=2).flatten(),
    bins=n_bins, range=(0, v_norm_max), density=True
)
axes[1].scatter(v_norm, v_norm_pdf, marker='+', c='tab:red', s=20,
                label='NVT Simulation')
axes[1].plot(v_norm, sps.maxwell.pdf(v_norm, 0, np.sqrt(temp)), c='k',
             label=r'Maxwell-Boltzmann$\left(\mu=0, \sigma=\sqrt{T}\right)$')
axes[1].set_xlabel(r'Velocity norm $||v||_2$')
axes[1].set_ylabel(r'$PDF(||v||_2)$')
axes[1].legend()
axes[1].set_ylim(-0.1, 1.0)

fig.tight_layout()
# %% 2.1 Test Nose-Hoover effective mass
fig, ax = plt.subplots()

v_min = -5.0
v_max = +5.0
n_bins = 300
v = np.linspace(v_min, v_max, n_bins)

ax.plot(v, sps.norm.pdf(v, 0, np.sqrt(temp)), c='k',
        label=r'Normal$\left(\mu=0,\sigma=\sqrt{T}\right)$')

for Q in [0.01, 0.1, 1.0, 10.0, 100.0]:
    nvt_hist, nvt_traj = run_nvt(
        nve_traj['pos'][-1], nve_traj['vel'][-1], box_len, r_cut, dt, temp, Q,
        frict, ln_s, n_steps_nvt, log_every
    )

    v_pdf, _ = np.histogram(nvt_traj['vel'].flatten(), bins=n_bins, range=(v_min, v_max),
                            density=True)
    ax.plot(v, v_pdf, label=f'Q={Q:0.2f}', alpha=0.5)

ax.legend()

# %% 2.2 Study a change of temperature
temp_1 = 0.7807
temp_2 = 1.2500

fig, ax = plt.subplots()
for Q in [1.0, 10.0, 100.0]:
    nvt_hist_1, nvt_traj_1 = run_nvt(
        pos=nve_traj['pos'][-1],
        vel=nve_traj['vel'][-1],
        box_len=box_len,
        r_cut=r_cut,
        dt=dt,
        temp=temp_1,
        eff_mass=Q,
        frict=0.0,
        ln_s=0.0,
        n_steps=n_steps_nvt,
        log_every=log_every
    )

    nvt_hist_2, nvt_traj_2 = run_nvt(
        pos=nvt_traj_1['pos'][-1],
        vel=nvt_traj_1['vel'][-1],
        box_len=box_len,
        r_cut=r_cut,
        dt=dt,
        temp=temp_2,
        eff_mass=Q,
        frict=nvt_hist_1['frict'][-1],
        ln_s=nvt_hist_1['ln_s'][-1],
        n_steps=n_steps_nvt,
        log_every=log_every
    )

    temp = np.concatenate((nvt_hist_1['temp'], nvt_hist_2['temp']))
    time = np.concatenate((nvt_hist_1['time'], nvt_hist_2['time'] + nvt_hist_1['time'][-1]))

    ax.plot(time, temp, label=f'Q={Q:0.2f}')

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Temperature')
# %% 3.1 Test Nose-Hoover at several temperatures

equil_n_steps = 200
equil_log_every = 100
prod_n_steps = 2_000
prod_log_every = 5

base_temp = 0.7807
temps = base_temp * np.arange(0.4, 1.6, 0.4)
v_min = -10.0
v_max = +10.0
n_bins = 500

fig, ax = plt.subplots()
for temp in temps:
    equil_hist, equil_traj = run_nvt(
        pos=nve_traj['pos'][-1],
        vel=nve_traj['vel'][-1],
        box_len=box_len,
        r_cut=r_cut,
        dt=dt,
        temp=temp,
        eff_mass=eff_mass,
        frict=0.0,
        ln_s=0.0,
        n_steps=equil_n_steps,
        log_every=equil_log_every
    )

    prod_hist, prod_traj = run_nvt(
        pos=equil_traj['pos'][-1],
        vel=equil_traj['vel'][-1],
        box_len=box_len,
        r_cut=r_cut,
        dt=dt,
        temp=temp,
        eff_mass=eff_mass,
        frict=equil_hist['frict'][-1],
        ln_s=equil_hist['ln_s'][-1],
        n_steps=2_000,
        log_every=5
    )

    v_pdf, bin_edges = np.histogram(prod_traj['vel'].flatten(), bins=n_bins,
                                    range=(v_min, v_max), density=True)
    v = bin_edges[:-1] + np.diff(bin_edges) / 2
    ax.plot(v, v_pdf, label=f'T={temp:0.3f}')

ax.set_xlabel(r'Velocity component $v_i$')
ax.set_ylabel(r'$PDF(v_i)$')
ax.legend()

# %%
# %% 3.2 Temperature fluctuations in NVE vs Nose-Hoover NVT
equil_n_steps = 200
equil_log_every = 100
prod_n_steps = 2_000
prod_log_every = 5

dt = 0.0046
eff_mass = 5.0

base_temp = 0.7807
temps = base_temp * np.arange(0.8, 1.8, 0.2)

temp_mean_sqs = []
temp_variances = []
for temp in temps:
    equil_hist, equil_traj = run_nvt(
        pos=nve_traj['pos'][-1],
        vel=nve_traj['vel'][-1],
        box_len=box_len,
        r_cut=r_cut,
        dt=dt,
        temp=temp,
        eff_mass=eff_mass,
        frict=0.0,
        ln_s=0.0,
        n_steps=equil_n_steps,
        log_every=equil_log_every
    )

    prod_hist, prod_traj = run_nvt(
        pos=equil_traj['pos'][-1],
        vel=equil_traj['vel'][-1],
        box_len=box_len,
        r_cut=r_cut,
        dt=dt,
        temp=temp,
        eff_mass=eff_mass,
        frict=equil_hist['frict'][-1],
        ln_s=equil_hist['ln_s'][-1],
        n_steps=prod_n_steps,
        log_every=prod_log_every
    )

    temp_mean_sqs.append(np.mean(prod_hist['temp'])**2)
    temp_variances.append(np.var(prod_hist['temp']))

fig, ax = plt.subplots()
ax.plot(temp_mean_sqs, temp_variances, marker='o')
ax.set_xlabel(r'$\langle T \rangle^2$')
ax.set_ylabel(r'$\langle T^2 \rangle$')

# %% 4.1 Compute the diffusion coefficient
%%time
temp = 0.7807
dt = 0.0046
eff_mass = 10.0

n_runs = 40
equil_n_steps = 200
equil_log_every = 100
prod_n_steps = 500
prod_log_every = 1

equil_hist, equil_traj = run_nvt(
    pos=nve_traj['pos'][-1],
    vel=nve_traj['vel'][-1],
    box_len=box_len,
    r_cut=r_cut,
    dt=dt,
    temp=temp,
    eff_mass=eff_mass,
    frict=0.0,
    ln_s=0.0,
    n_steps=equil_n_steps,
    log_every=equil_log_every
)

hist = equil_hist
traj = equil_traj

diff_coeff_hist = []
for _ in range(n_runs):
    hist, traj = run_nvt(
        pos=traj['pos'][-1],
        vel=traj['vel'][-1],
        box_len=box_len,
        r_cut=r_cut,
        dt=dt,
        temp=temp,
        eff_mass=eff_mass,
        frict=hist['frict'][-1],
        ln_s=hist['ln_s'][-1],
        n_steps=equil_n_steps,
        log_every=equil_log_every
    )
    slope, *_ = sps.linregress(hist['time'], hist['msd'])
    # Compute the diffusion coefficient for this sub-trajectory
    D = slope / 6 * 3.4e-10**2 * 4.6286e11 * 1e4
    diff_coeff_hist.append(D)

D_avg = np.mean(diff_coeff_hist)
D_std = np.std(diff_coeff_hist) / np.sqrt(n_runs)

print(f'D = {D_avg:0.5e}  STDEV(D) = {D_std:0.5e}')

# %% 4.2 Compute the radial distribution function g(r) and the structure factor S(k)
%%time
from measures import compute_rdf, compute_structure_factor_FT

temp = 0.01
dt = 0.0046
eff_mass = 10.0

n_runs = 40
equil_n_steps = 200
equil_log_every = 100
prod_n_steps = 2_000
prod_log_every = 5

equil_hist, equil_traj = run_nvt(
    pos=nve_traj['pos'][-1],
    vel=nve_traj['vel'][-1],
    box_len=box_len,
    r_cut=r_cut,
    dt=dt,
    temp=temp,
    eff_mass=eff_mass,
    frict=0.0,
    ln_s=0.0,
    n_steps=equil_n_steps,
    log_every=equil_log_every
)

prod_hist, prod_traj = run_nvt(
    pos=equil_traj['pos'][-1],
    vel=equil_traj['vel'][-1],
    box_len=box_len,
    r_cut=r_cut,
    dt=dt,
    temp=temp,
    eff_mass=eff_mass,
    frict=equil_hist['frict'][-1],
    ln_s=equil_hist['ln_s'][-1],
    n_steps=prod_n_steps,
    log_every=prod_log_every
)

r, g = compute_rdf(prod_traj['pos'], box_len, n_bins=300)
q, S = compute_structure_factor_FT(prod_traj['pos'], box_len, n_bins=300, n_q=300, q_max=32.0)

fig, axes = plt.subplots(1, 2, figsize=(9,4))

axes[0].plot(r, g, c='tab:blue')
axes[0].set_xlabel('r')
axes[0].set_ylabel('g(r)')
axes[0].set_title('Radial distribution')

axes[1].plot(q, S, c='tab:red')
axes[1].set_xlabel('q')
axes[1].set_ylabel('S(q)')
axes[1].set_title('Structure factor')
# %%
