import numpy as np
from ase.constraints import FixAtoms
from ase import Atoms
import ase
from typing import List, Dict



def stoichiometry(atoms:Atoms)->Dict[str, int]:

    ''' Returns a dictionary with how many atoms each element has'''
    symbols = atoms.get_chemical_symbols()
    unique = list(set(symbols))
    symbols = np.array(symbols)

    return {symbol:int(np.sum(symbols==symbol)) for symbol in unique}


def get_composition(atoms: ase.Atoms) -> dict:
    """Returns the composition of a structure.
    Args:
    """
    N: int = len(atoms)
    elements: List[str] = atoms.get_chemical_symbols()
    unique_elements: List[str] = list(set(elements))
    unique_elements.sort()

    res = {"N": N}
    for element in unique_elements:
        res[element] = elements.count(element)
    return res


def count_number_fixed_atoms(atoms: ase.Atoms) -> int:
    """Counts the number of fixed atoms in an ase.Atoms structure"""
    fixed_atoms = 0
    for constraint in atoms.constraints:
        if (type(constraint)) == FixAtoms:
            fixed_atoms = len(constraint.index)
    return fixed_atoms


class Surface:
    @staticmethod
    def calc_gamma_corrected(Ef, Ei, A, N, e_bulk, SI=True):
        factor = 16 if SI else 1
        return factor * (2 * Ef - Ei - N * e_bulk) / (2 * A)

    @staticmethod
    def calc_gamma_simple(Ef, A, N, e_bulk, SI=True):
        factor = 16 if SI else 1
        return factor * (Ef - N * e_bulk) / (2 * A)

    @staticmethod
    def calc_surface_area(atoms: ase.Atoms) -> float:
        """
        Calculates the area of the surface structure, based on a POSCAR or CONTCAR file.
        Args:
        -path: The path to a POSCAR or CONTCAR file or a folder that contains it.
        -file type: The type of the file. Either POSCAR or CONTCAR. If None(default) then
        it is assumed that the path contains the path to the proper file.
        """

        lattice = atoms.get_cell().array
        return np.cross(lattice[0], lattice[1])[2]


# class Symmetry:
#     @staticmethod
#     def get_spacegroup(atoms: ase.Atoms) -> str:
#         """
#         Returns the symmetry group.
#         Interface for spglib.

#         Args:
#         -----
#         atoms: The atomic configuration to be evaluated.
#         """
#         import spglib
#         cell = (
#             atoms.get_cell().array,
#             atoms.get_positions(),
#             atoms.get_atomic_numbers(),
#         )
#         sym_dataset = spglib.get_symmetry_dataset(cell) # type: ignore
#         return sym_dataset.international # type: ignore

#     @staticmethod
#     def get_standard_cell(atoms: ase.Atoms, tol: float = 1e-5):
#         """
#         Returns the conventional cell.
#         Interface for spglib.

#         Args:
#         -----
#         atoms: The atomic configuration to be evaluated.
#         tol: a tolerance value. Indicates a small range of possible rattling under which
#             an atomic configuration is considered symmetric. The higher, the more
#             symmetric the system is to be seen.
#         """
#         import spglib
#         cell = (
#             atoms.get_cell().array,
#             atoms.get_positions(),
#             atoms.get_atomic_numbers(),
#         )

#         return spglib.standardize_cell(
#             cell, to_primitive=True, no_idealize=False, symprec=tol # type: ignore
#         )
