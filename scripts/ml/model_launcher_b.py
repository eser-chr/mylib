#!/usr/bin/env python3
"""
Generate folders for MACE models with per-file dataset splits.
Each extxyz file is split independently to ensure equal representation.
"""

import logging
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from ase.io import read, write
from mace.cli import run_train
from typing import List
import json
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)

def create_dirs(
    base: Path,
    dataset_folder: Path,
    file_list: List[str],
    N_of_model: int,
    split_seed: int = 42,
    max_force_threshold: float = 30.0,  # filter threshold
    valid_fraction: float = 0.1,
    test_fraction: float = 0.2,
):

    train_atoms = []
    valid_atoms = []
    test_atoms = []

    for i, file in enumerate(file_list):
        path = dataset_folder / file

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.suffix != ".extxyz":
            raise ValueError(f"{path} must be an .extxyz file")

        try:
            atoms_all = list(read(path, index=":", format="extxyz"))
        except Exception as e:
            raise RuntimeError(f"Could not read {path}: {e}")

        if len(atoms_all) < 2:
            raise ValueError(f"{path} contains too few structures ({len(atoms_all)})")

        # --- 1. Filter high-force structures ---
        filtered_atoms = []
        for at in atoms_all:
            forces = at.arrays["REF_forces"]
            # forces = at.get_forces()
            f_mag = np.linalg.norm(forces, axis =1)
            if np.all(f_mag <= max_force_threshold):
                filtered_atoms.append(at)

        if len(filtered_atoms) < 2:
            raise ValueError(
                f"{path} has too few structures after filtering high forces"
            )

        # --- 2. Split test set first ---
        test_size = max(1, int(test_fraction * len(filtered_atoms)))
        train_plus_valid, test = train_test_split(
            filtered_atoms,
            test_size=test_size,
            random_state=split_seed + N_of_model + i,
        )

        # --- 3. Split validation from train_plus_valid ---
        valid_size = max(1, int(valid_fraction * len(train_plus_valid)))
        train, valid = train_test_split(
            train_plus_valid,
            test_size=valid_size,
            random_state=split_seed + N_of_model + i,
        )

        train_atoms.extend(train)
        valid_atoms.extend(valid)
        test_atoms.extend(test)

    # --- 4. Write files ---
    model_dir = base / str(N_of_model)
    model_dir.mkdir(exist_ok=True)

    write(model_dir / "train.extxyz", train_atoms)
    write(model_dir / "valid.extxyz", valid_atoms)
    write(model_dir / "test.extxyz", test_atoms)

    logging.info(
        f"Created splits: train={len(train_atoms)}, valid={len(valid_atoms)}, test={len(test_atoms)}"
    )
    
    


def train_model(model_path: Path, hyperparams: dict, flags: List[str]):
    orig = Path.cwd()

    os.chdir(model_path)

    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ["MKL_THREADING_LAYER"] = "INTEL"

    try:

        sys.argv = (
            ["mace_run_train"]
            + [f"--{k}={v}" for k, v in hyperparams.items()]
            + [f"--{flag}" for flag in flags]
        )

        run_train.main()

    finally:
        os.chdir(orig)


def main(args):

    base = args.base_dir
    dataset_folder = args.dataset_dir
    N_of_model = args.task_id

    settings_file = base / "description.json"

    logging.info(f"base: {base}")
    logging.info(f"dataset_folder: {dataset_folder}")
    logging.info(f"settings_file: {settings_file}")

    try:
        with open(settings_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not read description file: {e}")
        sys.exit(1)

    if "train" not in data:
        raise KeyError("description.json must contain a 'train' list of dataset files")

    create_dirs(
        base=base,
        dataset_folder=dataset_folder,
        file_list=data["train"],
        N_of_model=N_of_model,
        valid_fraction=args.valid_fraction,
        max_force_threshold=args.force_filter,
    )

    hp = data["model_params"].copy()
    hp["seed"] = N_of_model * 100 + 100

    flags = data.get("flags", [])

    train_model(base / str(N_of_model), hp, flags)


def main_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "base_dir",
        type=Path,
        help="Path to the base directory for this model generation",
    )

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing extxyz datasets",
    )

    parser.add_argument("job_array_id", type=int, help="Job array id (unused)")
    parser.add_argument("task_id", type=int, help="Model index")

    parser.add_argument(
        "--valid_fraction",
        type=float,
        default=0.1,
        help="Fraction of structures used for validation (default: 0.1)",
    )

    parser.add_argument(
        "--force_filter",
        type=float,
        default=30.0,
        help="Maximum allowed force magnitude in eV/A before filtering (default: 30.0)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = main_parser()
    main(args)(base) 