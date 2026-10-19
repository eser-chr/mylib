from ase import Atoms
from ase.constraints import FixAtoms
from ase.build import surface
import numpy as np
from typing import Tuple


def invert_selective_dynamics(atoms: Atoms) -> Atoms:
    """
    Inverts the constraint of fixed atoms in an atoms object.
    Args:
    - atoms: The ase.Atoms object
    """
    if not atoms.constraints:
        raise RuntimeError("The system contains no constraints !!!")

    N_atoms = len(atoms.get_chemical_symbols())

    A = {}
    for constraint in atoms.constraints:
        if type(constraint) == FixAtoms:
            A = set(constraint.index)

    atoms.set_constraint()
    a = set(range(N_atoms))
    new = a.difference(A)
    new = list(new)
    new.sort()
    new = np.array(new)
    atoms.set_constraint(FixAtoms(new))
    return atoms


def fix_positions_bottom_half(slab: Atoms) -> Atoms:
    """It fixes the atoms of an atomic configuration that are below the middle z-point.

    Args:
        slab (Atoms): The atomic configuration.

    Returns:
        Atoms: The new atomic configuration.
    """

    z_positions = slab.positions[:, 2]
    sorted_indices = np.argsort(z_positions)  # Sort by z-position
    num_atoms = len(z_positions)
    num_fixed_atoms = num_atoms // 2  # Fix bottom half
    fixed_indices = sorted_indices[:num_fixed_atoms]
    constraint = FixAtoms(indices=fixed_indices)
    slab.set_constraint(constraint)
    return slab


def create_surface(
    bulk: Atoms,
    miller_index: Tuple,
    layers: int,
    vacuum: float,
    supercell_size: Tuple = (2, 2, 1),
) -> Atoms:
    """API for ase.surface

    Args:
        bulk (_type_): _description_
        miller_index (_type_): _description_
        layers (_type_): _description_
        vacuum (_type_): _description_
        supercell_size (tuple, optional): _description_. Defaults to (2, 2, 1).

    Returns:
        _type_: _description_
    """

    slab = surface(bulk, miller_index, layers=layers, vacuum=vacuum, periodic=True)
    slab = slab.repeat(supercell_size)  # Expand to desired supercell

    # Get atom symbols and positions
    symbols = slab.get_chemical_symbols()

    positions = slab.get_positions()

    # Sort atom indices based on chemical symbols (to ensure VASP-compatible ordering)
    unique_elements = sorted(set(symbols), key=symbols.index)
    element_indices = {el: [] for el in unique_elements}

    for i, symbol in enumerate(symbols):
        element_indices[symbol].append(i)

    # Reorder atoms by element type
    reordered_indices = sum(element_indices.values(), [])  # Flatten list
    reordered_positions = positions[reordered_indices]
    reordered_symbols = [symbols[i] for i in reordered_indices]

    # Create reordered Atoms object
    reordered_slab = slab[reordered_indices]
    reordered_slab.set_positions(reordered_positions)
    reordered_slab.set_chemical_symbols(reordered_symbols)
    return reordered_slab


def level_correction(slab: Atoms)->Atoms:
    """It removes the atoms on the upper z layer and appends them to the bottom layer.
    Warning! It does not create a new layer. The new position might be too close to
    other particles and lead to explosive behavior.

    The itention of this function is to correct for ase.build.surface, which sometimes wrongly
    appends some atoms on top of the slab assuming that the void does not exist and we have pbc.

    Ideally this function should not exist. But sadly we do not live in an ideal world.

    Args:
    -slab: an ase.Atoms object, representing the slab

    """
    P = slab.positions
    M = np.max(P[:, 2])
    m = np.min(P[:, 2])
    mask = P[:, 2] == M
    P[:, 2] = P[:, 2] * (1 - mask) + m * mask
    slab.positions = P
    return slab
