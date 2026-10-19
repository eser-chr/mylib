import numpy as np

kB = 8.617333e-5 # Boltzman constant in ev/K
ev = 1.602176634e-19 # 1 eV in Joules
nm = 1.0e-9 # 
AA = 1.0e-10 
pm = 1.0e-12


def Ek_to_T(Ek:float, N:int):
    
    """ 
    Computes the Temperature of N particles moving in 3D given their kinetic energy.
    No rotational dof are assumed.
    
    ARGS:
    - Ek: kinetic energy
    - N: Number of particles 
    """
    return (2*Ek)/(3*N*kB)

def theo_kpoints(lattice: np.ndarray, KSPACING: float) -> np.ndarray:
    """
    Computes the number of k-points for a given lattice matrix and KSPACING.

    Parameters:
    - lattice (np.array): 3x3 real-space lattice matrix (rows are basis vectors).
    - KSPACING (float): Desired k-point spacing in reciprocal space.

    Returns:
    - np.array: Integer array with the number of k-points along each reciprocal direction.
    """
    
    
    if lattice.shape != (3, 3):
        raise ValueError("Lattice must be a 3x3 matrix.")

    if KSPACING <= 0:
        raise ValueError("KSPACING must be positive.")
    
    reciprocal_lattice = 2 * np.pi * np.linalg.inv(lattice).T
    b_norms = np.linalg.norm(reciprocal_lattice, axis=1)

    if np.any(b_norms <= 0):
        raise ValueError("Invalid reciprocal lattice: Contains near-zero vectors.")


    num_kpoints = np.ceil(b_norms / KSPACING).astype(int)
    return np.maximum(num_kpoints, 1)


# def theo_kpoints(cell: np.ndarray, kspacing: float = 0.1) -> np.ndarray:
#     """
#     Calculate k-point mesh based on reciprocal lattice norms and spacing.
#     """
#     rec_lat = 2 * np.pi * np.linalg.inv(cell).T
#     norms = np.linalg.norm(rec_lat, axis=1)
#     points = np.ceil(norms / kspacing).astype(int)
#     return np.maximum(points, 1)