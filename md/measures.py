import typing as ty

from numba import jit, prange, float64, int64
import numpy as np

from distance import compute_squared_distances

@jit(float64(float64[:,:]), nopython=True)
def compute_e_kin(vel: np.ndarray) -> float:
    """Compute the kinetic energy of a set of velocities, assuming mass = 1.

    E_kin = p^2 / 2m
          = v^2       given that m = 1

    Args:
        vel (np.ndarray): (n_atoms x 3) array of velocities.

    Returns:
        float: Total kinetic energy.
    """
    return 0.5 * np.sum(vel**2)

@jit(float64(float64[:,:]), nopython=True)
def compute_temp(vel: np.ndarray) -> float:
    """Compute the temperature of a set of velocities.

    kT = 2 E_kin / (g N)
    where g is the number of degrees of freedom per particle, and N is the number of
    particles.

    Args:
        vel (np.ndarray): (n_atoms x 3) array of velocities.

    Returns:
        float: Temperature in units of energy.
    """
    # 3N degrees of freedom in total
    dof = vel.shape[1] * vel.shape[0]
    return 2 * compute_e_kin(vel) / dof

@jit(float64(int64, float64, float64, float64, float64), nopython=True)
def compute_e_nh(dof: int, temp: float, frict: float, ln_s: float, eff_mass: float) -> float:
    """Compute the Nosé-Hoover energy contribution.

    E_nh = ξ^2 Q / 2 + g s kT
    where
        ξ is the friction strength
        Q is the effective mass
        g is the number of degrees of freedom
        s is the Nosé-Hoover scaling term such that
            p' = p / s
            Δt' = Δt / s
        kT is the temperature in units of energy

    Args:
        dof (int): Total number of degrees of freedom.
        temp (float): Target temperature in units of energy.
        frict (float): Friction strength ξ.
        ln_s (float): Natural logarithm of the Nosé-Hoover conversion factor s.
        eff_mass (float): Effective mass Q.

    Returns:
        float: Total Nosé-Hoover energy contribution.
    """
    # ???: shouldn't the second term read (dof * exp(ln_s) * temp)?
    return (0.5 * frict**2 * eff_mass) + (dof * ln_s * temp)

@jit(float64(float64[:,:], float64[:,:]), nopython=True)
def compute_vacf(vel: np.ndarray, vel_init: np.ndarray) -> float:
    """Compute the autocorrelation between two sets of velocities.

    Args:
        vel (np.ndarray): (n_atoms x 3) array of velocities.
        vel_init (np.ndarray): (n_atoms x 3) array of velocities.

    Returns:
        float: Velocity autocorrelation.
    """
    # i = [1, N] (particle index), a = [1, 3] (x, y, z)
    # (sum_{i,a}[v(t)[i][a] * v(0)[i][a]] / 3N) / (sum_{i,a}[v(0)[i][a] * v(0)[i][a]] / 3N)
    return np.mean(vel * vel_init) / np.mean(vel_init * vel_init)

@jit(float64(float64[:,:], float64[:,:], float64[:,:], float64), nopython=True)
def compute_msd(pos: np.ndarray, pos_init: np.ndarray, shift: np.ndarray, box_len: float) -> float:
    """Compute the mean squared displacement between two sets of positions in cubic periodic
    boundary conditions.

    Args:
        pos (np.ndarray): (n_atoms x 3) array of positions.
        pos_init (np.ndarray): (n_atoms x 3) array of positions.
        shift (np.ndarray): (n_atoms x 3) array of PBC shifts.
        box_len (float): Cubic box length.

    Returns:
        float: Mean squared displacement
    """
    pos_shifted = pos + box_len * shift
    return np.mean(np.sum((pos_shifted - pos_init)**2, axis=1))

@jit((float64[:,:,:], float64, int64), nopython=True)
def compute_rdf(pos_traj: np.ndarray, box_len: float,
                n_bins: int = 300) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function for a MD trajectory.

    Args:
        pos_traj (np.ndarray): (n_iterations x n_atoms x 3) array of positions.
        box_len (float): Cubic box length.
        n_bins (int): Number of bins for the radial distribution function.

    Returns:
        ty.Tuple[np.ndarray, np.ndarray]: (n_bins,) vectors r (bin-centers) and g(r)
            (radial distribution function).
    """
    n_iters = pos_traj.shape[0]
    n_atoms = pos_traj.shape[1]
    n_dists = n_atoms * (n_atoms - 1)  # Upper triangle only

    volume = box_len**3

    half_bin_width = 0.5 * box_len / n_bins
    r_max2 = (box_len / 2)**2  # r_max = box_len / 2

    rdfs = np.zeros((n_iters, n_bins))
    for i in range(n_iters):
        dist2 = compute_squared_distances(pos_traj[i], box_len)
        for j in range(n_atoms):
            for k in range(j + 1, n_atoms):
                if dist2[j, k] < r_max2:
                    dist_jk = np.sqrt(dist2[j, k])
                    bin_index = int(dist_jk / half_bin_width)
                    rdfs[i, bin_index] += 1

    g = np.zeros(n_bins)
    r = np.zeros(n_bins)
    pre_factor = 2 * volume / n_dists / (4 * np.pi * half_bin_width)
    for i in range(n_bins):
        r[i] = half_bin_width * (i + 0.5)
        g[i] = pre_factor * np.mean(rdfs[:,i]) / r[i]**2

    return r, g

@jit(nopython=True)
def compute_structure_factor_FT(pos_traj: np.ndarray, box_len: float, n_bins: int = 300,
                                n_q: int = 300, q_max: float = 30.0) -> ty.Tuple[
                                    np.ndarray, np.ndarray
                                ]:
    n_atoms = pos_traj.shape[1]
    volume = box_len**3
    density = n_atoms / volume

    r, g = compute_rdf(pos_traj, box_len, n_bins)
    n_r = r.shape[0]
    dr = r[1] - r[0]
    integrand = np.zeros_like(r)

    q = np.linspace(start=0.0, stop=q_max, num=n_q)
    S = np.zeros(n_q)

    for i_q in range(n_q):
        for i_r in range(n_r):
            qr = q[i_q] * r[i_r]

            if qr < 1e-8:  # Small-sine
                integrand[i_r] = r[i_r]**2 * (g[i_r] - 1) * (
                    1 - qr**2 / 6 * (
                        1 - qr**2 / 20 * (
                            1 - qr**2 / 42
                        )
                    )
                )
            else:
                integrand[i_r] = r[i_r] * (g[i_r] - 1) * np.sin(qr) / q[i_q]

            S[i_q] = 1 + 4 * np.pi * density * np.trapz(y=integrand, dx=dr)

    return q, S

## * This should give practically the same results as the old generator, but currently
## * the q-values don't agree and the k-points don't _exactly_ match.
## * The number of k-vectors in each q-shell is identical, however.
# def _build_k_grid(box_len: float, k_mesh: int = 31,
#                  n_q_max: int = 300, q_max: float = 32.0):

#     recip_box_len = 2 * np.pi / box_len
#     k_mesh = 2 * (k_mesh // 2) + 1  # Round k_mesh up to the next odd integer
#     n_q = min(n_q_max, int(q_max / recip_box_len))

#     # Make a vector of integers from -k//2 to +k//2 inclusive.
#     # e.g. for k_mesh == 5, ijk_lin = [-2, -1, 0, +1, +1].
#     ijk_lin = np.arange(-(k_mesh // 2), k_mesh // 2 + 1, dtype=np.int64)

#     # Make the 3D grid and reduce it to a list of [i, j, k] vectors.
#     ijk_vectors = np.meshgrid(ijk_lin, ijk_lin, ijk_lin)
#     ijk_vectors = np.hstack([np.reshape(ijk_vectors[i], (k_mesh**3, 1)) for i in range(3)])

#     # Remove k-vectors related by time-reversal symmetry ([i, j, k] = [-i, -j, -k]
#     ijk_vectors = ijk_vectors[:(ijk_vectors.shape[0] // 2)]

#     # Compute q in (box_len sigma)^-1 units to agree with positions units (box_len sigma).
#     qs_cart = recip_box_len * np.linalg.norm(ijk_vectors, axis=1)

#     # Take only k-vectors where b|k| < q_max and remove |k| = 0
#     indices_in_cutoff = (qs_cart < q_max) & (qs_cart != 0)
#     ijk_vectors = ijk_vectors[indices_in_cutoff]
#     qs_cart = qs_cart[indices_in_cutoff]

#     # Count the number of qs in each 'shell' or histogram bin.
#     n_k_vectors_in_q_shell, q_shell_edges = np.histogram(qs_cart, bins=n_q, range=(0.0, q_max))
#     q_shell_width = q_shell_edges[1] - q_shell_edges[0]
#     q_shell_centers = (q_shell_edges + q_shell_width / 2)[:-1]

#     # Find the mapping from k-vectors to q-bins
#     i_k_vector_to_i_q_shell = (qs_cart / q_max * n_q).astype(np.int64)

#     return ijk_vectors, q_shell_centers, i_k_vector_to_i_q_shell, n_k_vectors_in_q_shell

@jit((float64, int64, int64, float64), nopython=True)
def build_k_grid(box_len: float, k_mesh: int = 30, n_q_max: int = 300,
                  q_max: float = 32.0):
    permutations = np.array([
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
    ], dtype=np.int64)
    n_k_max = 300_000

    recip_box_len = 2 * np.pi / box_len
    n_q = min(n_q_max, int(q_max / recip_box_len))

    dq = q_max / (n_q - 1)
    q = dq * (np.arange(n_q) + 0.5)
    q[0] = 0.0

    k_vectors = np.full((n_k_max, 3), np.nan)
    i_k_vector_to_i_q_shell = np.full((n_k_max, ), -1, dtype=np.int64)
    n_k_vectors_in_q_shell = np.zeros((n_q, ), dtype=np.int64)

    i_k = 0
    k_norm = 0.0
    dk = 1

    while k_norm < 4 * q_max:
        for i in range((dk - 1) * k_mesh, dk * k_mesh, dk):
            for j in range((dk - 1) * k_mesh, dk * k_mesh, dk):
                for k in range((dk - 1) * k_mesh, dk * k_mesh, dk):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    ijk = np.array([i, j, k], dtype=np.float64)
                    k_norm = recip_box_len * np.linalg.norm(ijk)
                    if k_norm < q_max:
                        i_q = int(k_norm / q_max * n_q)
                        k_vectors[i_k] = np.array([i, j, k], dtype=np.float64)
                        i_k_vector_to_i_q_shell[i_k] = i_q
                        n_k_vectors_in_q_shell[i_q] += 1
                        i_k += 1

                        ijk_old = ijk.copy()
                        for perm in permutations:
                            ijk_new = ijk * perm
                            if (
                                np.sum(np.abs(ijk_new - ijk)) != 0 and
                                np.sum(np.abs(ijk_new + ijk_old)) != 0
                            ):
                                k_vectors[i_k] = ijk_new
                                i_k_vector_to_i_q_shell[i_k] = i_q
                                n_k_vectors_in_q_shell[i_q] += 1
                                ijk_old = ijk_new
                                i_k += 1
        dk += 1

    # Take only the k-vectors that we generated
    k_vectors = k_vectors[:i_k]
    i_k_vector_to_i_q_shell = i_k_vector_to_i_q_shell[:i_k]

    return q, k_vectors, i_k_vector_to_i_q_shell, n_k_vectors_in_q_shell


@jit((float64[:,:,:], float64, int64, int64, float64), nopython=True, parallel=True)
def compute_structure_factor_direct(pos_traj: np.ndarray, box_len: float,
                                    k_mesh: int = 31, n_q_max: int = 300,
                                    q_max: float = 32.0) -> ty.Tuple[np.ndarray, np.ndarray]:
    q, k_vectors, i_k_vector_to_i_q_shell, n_k_vectors_in_q_shell = build_k_grid(
        box_len, k_mesh, n_q_max, q_max
    )
    n_k_vectors = k_vectors.shape[0]
    n_q = q.shape[0]

    n_iters = pos_traj.shape[0]
    n_atoms = pos_traj.shape[1]
    recip_box_len = 2 * np.pi / box_len

    S = np.full((n_q, ), 1 / n_atoms)

    for i_iter in prange(n_iters):
        for i_k in range(n_k_vectors):
            k_dot_r = recip_box_len * np.sum(k_vectors[i_k] * pos_traj[i_iter], axis=1)
            sum_sin = np.sum(np.sin(k_dot_r))
            sum_cos = np.sum(np.cos(k_dot_r))
            i_q_shell = i_k_vector_to_i_q_shell[i_k]
            S[i_q_shell] += (sum_cos**2 + sum_sin**2) / n_k_vectors_in_q_shell[i_q_shell]

    S /= (n_iters * n_atoms)

    return q, S

def compute_velocity_density(vel: np.ndarray, v_min: float = -10.0, v_max: float=10.0, n_bins: int=300):
    vel_dens, _ = np.histogram(
        vel.flatten(), bins=n_bins, range=(v_min, v_max), density=True
    )
    return vel_dens

def compute_velocity_norm_density(vel: np.ndarray, v_norm_max: float = 10.0, n_bins: int=300):
    vel_norm = np.sqrt(np.sum(vel**2, axis=1))
    vel_norm_dens, _ = np.histogram(
        vel_norm, bins=n_bins, range=(0.0, v_norm_max), density=True
    )
    return vel_norm_dens
