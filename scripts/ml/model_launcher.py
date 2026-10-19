#!/usr/bin/env python3
""" 
It generates folders for mace models.
"""


import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from ase.io import read, write
from mace.cli import run_train
from typing import List
import json


def concat_extxyz(list_of_files: List[Path], output_path: Path) -> None:
    for file in list_of_files:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        if file.suffix != ".extxyz":
            raise FileExistsError("The type of the file should be extxyz!")

    ATOMS = []
    for file in list_of_files:
        try:
            ATOMS.extend(read(file, index=":"))
        except Exception as e:
            raise RuntimeError(f"Failed to read {file}: {e}")
    write(output_path, ATOMS, format="extxyz")


def create_db_from_desc_file(
    description_file_json: Path, dataset_folder: Path, output_path: Path
) -> None:
    try:
        with open(description_file_json, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could Not Read the description file. Exit with error {e}")
        sys.exit(1)

    lines = data["train"]
    concat_extxyz([dataset_folder / line for line in lines], output_path)


def create_dirs(base: Path, n_models: int, split_seed: int = 42):
    try:
        all_atoms = list(read(base / "all.extxyz", index=":", format="extxyz"))
    except Exception as e:
        print(f"Could not read all.extxyz: {e}")
        sys.exit(1)

    for i in range(n_models):
        model_dir = base / str(i)
        model_dir.mkdir(exist_ok=True)
        train, test = train_test_split(
            all_atoms, test_size=0.2, random_state=split_seed + i
        )
        write(model_dir / "train.extxyz", train)
        write(model_dir / "test.extxyz", test)


def train_model(model_path: Path, hyperparams: dict, base_path: Path):
    orig = Path.cwd()
    os.chdir(model_path)
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ["MKL_THREADING_LAYER"] = "INTEL"

    try:
        sys.argv = ["mace_run_train"] + [f"--{k}={v}" for k, v in hyperparams.items()]
        run_train.main()
    finally:
        os.chdir(orig)


def main(description_file: Path, dataset_folder: Path):
    N = 5
    base = Path(".").resolve()
    output_file = description_file.parent / "all.extxyz"
    create_db_from_desc_file(description_file, dataset_folder, output_file)
    create_dirs(base, N)

    common = {
        "name": "model",
        "train_file": "train.extxyz",
        "test_file": "test.extxyz",
        "valid_fraction": 0.1,
        "E0s": "average",
        "model": "ScaleShiftMACE",
        "num_interactions": 3,
        "correlation": 3,
        "r_max": 6.0,
        "max_num_epochs": 200,
        "hidden_irreps": "64x0e + 64x1o",
        "default_dtype": "float32",
        "device": "cuda",
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        
    }

    for i in range(N):
        hp = common.copy()
        hp["seed"] = i* 100 + 100
        train_model(base / str(i), hp, base)


if __name__ == "__main__":
    DESCRIPTION_FILE = Path("description.json")
    DATASET_FOLDER = Path("~/Desktop/mlip_db/train").expanduser()

    main(DESCRIPTION_FILE, DATASET_FOLDER)
