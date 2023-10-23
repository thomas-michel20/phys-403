# -*- coding: utf-8 -*-
"""Structure builders."""
import typing as ty

import numpy as np


def build_fcc(n_cells: int, lattice_parameter: float) -> ty.Tuple[np.ndarray, float]:
    """Build an FCC supercell.

    Args:
        n_cells (int): Number of cells along each axis.
        lattice_parameter (float): Lattice parameter.

    Returns:
        np.ndarray: (n_atoms x 3) array of atomic positions.
        float: Box length.
    """
    frac_coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]])
    # N_atoms = 4 * N_cells**3
    n_atoms = frac_coords.shape[0] * n_cells**3
    # L = a * N_cells
    box_parameter = lattice_parameter * n_cells
    # Make an empty array for the positions of all the atoms
    pos = np.empty((n_atoms, 3))
    # Compute the x (also y and z, because the system is cubic) position of the origin
    # of each unit cell within the supercell.
    x_lin = np.linspace(0, lattice_parameter * (n_cells - 1), n_cells)
    # Get a mesh that has all the appropriate combinations to form the (x, y, z) coordinates
    # of the unit cell origins within the supercell.
    cell_origins = np.meshgrid(x_lin, x_lin, x_lin)
    # Reshape the data into a list of (x, y, z) coordinates.
    cell_origins = np.hstack([np.reshape(cell_origins[i], (n_cells**3, 1)) for i in range(3)])
    # Generate the atomic positions.
    for (i, frac_coord) in enumerate(frac_coords):
        pos[i * n_cells**3:(i + 1) * n_cells**3] = cell_origins + lattice_parameter * frac_coord
    # Return the size of the supercell box as well as all the positions.
    return pos, box_parameter
