import hashlib
import os
import yaml

from typing import List
import warnings
import numpy as np
import re
from pathlib import Path

import ase
from ase.io import read, write
from ase import units
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from mace.calculators import MACECalculator


def folder_structure_to_yaml(root_dir, output_file):

    def build_structure(path):
        if os.path.isfile(path):
            return None
        else:
            contents = {}
            for file in sorted(os.listdir(path)):
                full_path = os.path.join(path, file)
                contents[file] = build_structure(full_path)
            return contents

    tree = {os.path.basename(root_dir): build_structure(root_dir)}

    with open(output_file, "w") as f:
        yaml.dump(tree, f, sort_keys=False)


def compute_file_hash(file_path):
    """Compute SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def compute_simulation_hash(sim_path):
    """Generate a hash of key input files for verification."""
    files_to_hash = ["POSCAR", "INCAR", "KPOINTS", "POTCAR"]
    hashes = {f: compute_file_hash(os.path.join(sim_path, f)) for f in files_to_hash}
    return hashlib.sha256("".join(filter(None, hashes.values())).encode()).hexdigest()


def find_vasp_sims(folder_path: Path) -> List[Path]:
    """Return paths to VASP simulations. A simulation is defined by presence of vasprun.xml in a folder."""
    res = [p.parent for p in folder_path.rglob("vasprun.xml")]
    if not res:
        warnings.warn("No simulations found.")
    return res


def concat_extxyz(list_of_files: List[Path], output_path: Path) -> None:
    """
    Concatenates the content of xyz files into one extxyz file.
    """

    for file in list_of_files:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        if file.suffix != ".extxyz":
            raise FileExistsError("The type of the file should be extxyz!")

    ATOMS = []
    for file in list_of_files:
        ATOMS.extend(read(file, index=":"))
    write(output_path, ATOMS, format="extxyz")


def classify_atomic_configuration(atoms: ase.Atoms, vac_threshold: float = 3.0) -> str:
    """
    Read an ase.Atoms object and decide if it’s bulk / surface / cluster.

    We assume that a bulk object is periodic in all directions a surface only in 2
    and a cluster in None.

    Parameters
    ----------
    atoms : the ase.Atom structure
    vac_threshold: Minimum “vacuum” thickness [Å] along an axis to count as
        non-periodic in that direction. The higher the easier to consider a structure
        periodic.

    Returns
    -------
    one of { 'bulk', 'surface', 'cluster' }
    """
    lengths = atoms.get_cell_lengths_and_angles()[:3]

    pos = atoms.get_positions()  # N×3 array
    bbox = pos.max(axis=0) - pos.min(axis=0)  # atom extents along x,y,z
    vacuum: float = lengths - bbox  # how much empty space in each dir
    nonperiodic: int = np.count_nonzero(vacuum > vac_threshold)

    if nonperiodic == 0:
        return "bulk"
    elif nonperiodic == 1:
        return "surface"
    else:
        return "cluster"


def get_element_of_unary_system(atom: ase.Atoms):
    """
    Returns the unique element of a unary system (consists of one atom only)
    """
    tmp = list(set(atom.get_chemical_symbols()))
    return tmp[0]


def fix_contcar_element_line(
    input_file: str | Path, output_file: str | Path | None = None
):

    if output_file == None:
        output_file = input_file
    with open(input_file, "r") as f:
        lines = f.readlines()

    potcar_line_index = 5  # Line index starts from 0
    potcar_elements = lines[potcar_line_index].strip().split()

    corrected_elements = [re.sub(r"[_/].*", "", elem) for elem in potcar_elements]

    lines[5] = " ".join(corrected_elements) + "\n"

    with open(output_file, "w") as f:
        f.writelines(lines)


def my_md(
    init_pos: ase.Atoms,
    time_fs: float,
    timestep: float,
    temp_K: float,
    writing_time: float,
    calculator: MACECalculator,
    output_file: str | Path,
) -> None:
    steps = int(time_fs / timestep)
    writing_steps = int(writing_time / timestep)
    init_pos.calc = calculator
    dyn = NoseHooverChainNVT(
        init_pos, timestep * units.fs, temperature_K=temp_K, tdamp=50 * units.fs
    )

    def write_frame():
        dyn.atoms.write(output_file, append=True)

    dyn.attach(write_frame, interval=writing_steps)
    dyn.run(steps)
    print("MD finished !!")


def get_matplotlib_iter():
    import matplotlib.pyplot as plt

    return iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
