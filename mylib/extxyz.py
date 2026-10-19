from typing import List
import numpy as np
import numpy.typing as npt
import ase
from ase.io import read, write
from sklearn.model_selection import train_test_split
import random
from pathlib import Path

def extract_array_tag_from_atoms_list(atoms: List[ase.Atoms], tag:str)->npt.NDArray:
    to_rtn = [atom_cfg.arrays[tag] for atom_cfg in atoms]
    return np.array(to_rtn)


def train_test_split_xyz(file_path: Path, test_size: float = 0.2) -> None:
    folder = file_path.parent
    train_path = folder / "train.extxyz"
    test_path = folder / "test.extxyz"

    atoms = read(file_path, index=":", format="extxyz")
    
    assert isinstance(atoms, List)
    random.shuffle(atoms)
    train_atoms, test_atoms = train_test_split(atoms, test_size=test_size)

    write(train_path, train_atoms, format="extxyz")
    write(test_path, test_atoms, format="extxyz")


def clean_unimportant_tags(atoms, keep_keys=['numbers', 'positions', 'tags']):
    new = {}
    for key in atoms.arrays.keys():
        if key in keep_keys:
            new[key] = atoms.arrays[key]
    atoms.arrays = new
    
    
