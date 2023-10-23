from numba import jit, prange, float64
import numpy as np

from distance import compute_pbc_diff

@jit(float64[:,:](float64[:,:], float64[:,:], float64, float64), nopython=True, parallel=True)
def compute_forces(pos: np.ndarray, dist2: np.ndarray, box_len: float, r_cut: float) -> np.ndarray:
    n_atoms = pos.shape[0]
    forces = np.zeros_like(pos)
    r_cut2 = r_cut**2

    for i in prange(n_atoms):
        for j in range(i + 1, n_atoms):
            if dist2[i, j] < r_cut2:
                dr = compute_pbc_diff(pos[i], pos[j], box_len)
                inv_dist2 = 1 / dist2[i,j]
                force = 4 * (12 * inv_dist2**7 - 6 * inv_dist2**4) * dr
                forces[i] += force
                forces[j] -= force

    return forces

@jit(float64(float64[:,:], float64[:,:], float64), nopython=True, parallel=False)
def compute_e_pot(pos: np.ndarray, dist2: np.ndarray, r_cut: float) -> float:
    n_atoms = pos.shape[0]
    e_pot = 0.0
    r_cut2 = r_cut**2

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if dist2[i,j] < r_cut2:
                inv_dist2 = 1 / dist2[i, j]
                e_pot += 4 * (inv_dist2**6 - inv_dist2**3)

    inv_rcut = 1 / r_cut
    e_pot += 8 * np.pi * (inv_rcut**9 / 9 - inv_rcut**3 / 3)

    return e_pot
