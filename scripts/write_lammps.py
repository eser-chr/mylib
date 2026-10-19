from ase.io import read, write
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np

def get_unique_symbols(atoms: Atoms) -> List[str]:
    """Return sorted unique chemical symbols present in atoms."""
    return sorted(set(atoms.get_chemical_symbols()))



def write_lammps_with_masses(atoms: Atoms, filename: str) -> None:
    symbols = atoms.get_chemical_symbols()
    uniq = get_unique_symbols(atoms)

    types = {s: i + 1 for i, s in enumerate(uniq)}
    masses = {s: atomic_masses[atomic_numbers[s]] for s in uniq}

    natoms = len(atoms)
    ntypes = len(uniq)

    cell = atoms.get_cell()
    xhi, yhi, zhi = cell[0, 0], cell[1, 1], cell[2, 2]

    with open(filename, "w") as f:
        # Title line (ASE always writes one)
        f.write("LAMMPS data file via custom writer\n\n")

        f.write(f"{natoms} atoms\n")
        f.write(f"{ntypes} atom types\n\n")

        f.write(f"0.0 {xhi:.15f} xlo xhi\n")
        f.write(f"0.0 {yhi:.15f} ylo yhi\n")
        f.write(f"0.0 {zhi:.15f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for s in uniq:
            f.write(f"{types[s]} {masses[s]:.6f} # {s}\n")
        f.write("\n")

        f.write("Atoms # atomic\n\n")
        for i, atom in enumerate(atoms, start=1):
            t = types[atom.symbol]
            x, y, z = atom.position
            f.write(f"{i} {t} {x:.8f} {y:.8f} {z:.8f}\n")
        f.write("\n")