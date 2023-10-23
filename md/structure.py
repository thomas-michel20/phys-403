import json
import numba
import typing as ty
import numpy as np

def build_fcc(n_cells: int, lat_par: float) -> ty.Tuple[float, np.ndarray]:
    """Build an FCC supercell.

    Args:
        n_cells (int): Number of cells along each axis.
        lat_par (float): Lattice parameter.

    Returns:
        float: Box length.
        np.ndarray: (n_atoms x 3) array of atomic positions.
    """
    frac_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.5]
    ])
    # N_atoms = 4 * N_cells**3
    n_atoms = frac_coords.shape[0] * n_cells**3
    # L = a * N_cells
    box_len = lat_par * n_cells
    # Make an empty array for the positions of all the atoms
    pos = np.empty((n_atoms, 3))
    # Compute the x (also y and z, because the system is cubic) position of the origin
    # of each unit cell within the supercell.
    x_lin = np.linspace(0, lat_par * (n_cells - 1), n_cells)
    # Get a mesh that has all the appropriate combinations to form the (x, y, z) coordinates
    # of the unit cell origins within the supercell.
    cell_origins = np.meshgrid(x_lin, x_lin, x_lin)
    # Reshape the data into a list of (x, y, z) coordinates.
    cell_origins = np.hstack([np.reshape(cell_origins[i], (n_cells**3, 1)) for i in range(3)])
    # Generate the atomic positions.
    for (i, frac_coord) in enumerate(frac_coords):
        pos[i * n_cells**3: (i + 1) * n_cells**3] = cell_origins + lat_par * frac_coord
    # Return the size of the supercell box as well as all the positions.
    return box_len, pos

def init_velocities(n_atoms: int, temp: float) -> np.ndarray:
    """Initialize velocities using a Maxwell-Boltzmann distribution.

    Args:
        n_atoms (int): Number of atoms.
        temp (float): Temperature in units of energy (kT).

    Returns:
        np.ndarray: (n_atoms x 3) array of velocities.
    """
    # Velocity components are Normally distributed (velocity norms will follow a
    # Maxwell-Boltzmann distribution).
    vel = np.random.normal(loc=0.0, scale=np.sqrt(temp), size=(n_atoms, 3))
    # Remove any center-of-mass velocity.
    vel -= np.mean(vel, axis=0)
    return vel

def save_frame(filename: str, hist: dict, traj: dict, index: int=-1, **kwargs):
    """Save a frame from an MD trajectory using JSON.

    Args:
        filename (str): Filename (should end in JSON)
        hist (dict): History (contains energies, temperature, etc.).
        traj (dict): Trajectory (contains, positions, velocities).
        index (int, optional): Index of the frame to save to disk. Defaults to -1.
    """
    if not filename.endswith('.json'):
        print(f'Adding JSON extension to filename; writing {filename}.json')
        filename += '.json'
    # Initialize the data dictionary and convert the positions and velocities at the
    # requested indices to Python lists.
    data = {
        'pos': traj['pos'][index].tolist(),
        'vel': traj['vel'][index].tolist()
    }
    # Add in all the data from traj at the requested index.
    for (key, value) in hist.items():
        data[key] = value[index]
    data = {**data, **kwargs}
    # Dump the json file.
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)

def load_frame(filename: str) -> dict:
    """Load a frame from an MD trajectory using JSON.

    Args:
        filename (str): Filename (file must contain JSON data).

    Returns:
        dict: Frame data (positions, velocities, energies, etc.).
    """
    # Load the file, assuming it contains JSON data.
    with open(filename, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    # Convert the positions and velocities back into NumPy arrays.
    for key in ['pos', 'vel']:
        data[key] = np.array(data[key])
    # Return the frame dictionary.
    return data
