from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
from ase import Atoms
from ase.io import read, write
from sklearn.model_selection import train_test_split


class Filter(ABC):
    @abstractmethod
    def __call__(self, structure: Atoms) -> bool:
        """Return True to keep the structure."""


class ForceFilter(Filter):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, structure: Atoms) -> bool:
        forces = structure.arrays["REF_forces"]
        return np.linalg.norm(forces, axis=1).max() < self.threshold


@dataclass(frozen=True)
class Split:
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        if self.val <= 0 or self.test <= 0 or self.val + self.test >= 1:
            raise ValueError(f"invalid split: val={self.val}, test={self.test}")

    @property
    def train(self) -> float:
        return 1 - self.val - self.test


def create_dirs(
    dataset_paths: list[Path],
    filters: list[Filter],
    output_dir: Path,
    split: Split = Split(),
    split_seed: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_atoms, valid_atoms, test_atoms = [], [], []

    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.suffix != ".extxyz":
            raise ValueError(f"{path} must be an .extxyz file")

        atoms_all = read(path, index=":", format="extxyz")
        kept = [at for at in atoms_all if all(f(at) for f in filters)]
        if len(kept) < 3:
            raise ValueError(f"{path}: only {len(kept)} of {len(atoms_all)} structures left after filtering")

        # test first, then validation (as a fraction of the remainder, so it is split.val of the total)
        rest, test = train_test_split(kept, test_size=split.test, random_state=split_seed)
        train, valid = train_test_split(
            rest, test_size=split.val / (1 - split.test), random_state=split_seed
        )

        train_atoms += train
        valid_atoms += valid
        test_atoms += test

    write(output_dir / "train.extxyz", train_atoms)
    write(output_dir / "valid.extxyz", valid_atoms)
    write(output_dir / "test.extxyz", test_atoms)

    logging.info(
        f"Created splits: train={len(train_atoms)}, valid={len(valid_atoms)}, test={len(test_atoms)}"
    )