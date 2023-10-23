from numba import jit, prange, float64
import numpy as np

@jit(float64[:,:](float64[:,:], float64), nopython=True, parallel=True)
def compute_squared_distances(pos: np.ndarray, box_len: float) -> np.ndarray:
    n_atoms = pos.shape[0]
    dist2 = np.zeros((n_atoms, n_atoms))

    for i in prange(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dx -= box_len * np.rint(dx / box_len)
            dy -= box_len * np.rint(dy / box_len)
            dz -= box_len * np.rint(dz / box_len)
            dist2[i, j] = (dx * dx + dy * dy + dz * dz)
            # dist2[i, j] = np.sum(compute_pbc_diff(pos[i], pos[j])**2)

    return dist2

@jit(float64[:](float64[:], float64[:], float64), nopython=True, parallel=False)
def compute_pbc_diff(r1: np.ndarray, r2: np.ndarray, box_len: float) -> np.ndarray:
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    dx -= box_len * np.rint(dx / box_len)
    dy -= box_len * np.rint(dy / box_len)
    dz -= box_len * np.rint(dz / box_len)
    return np.array([dx, dy, dz])
