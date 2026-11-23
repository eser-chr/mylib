#!/usr/bin/env python3
"""
Generate folders for MACE models with per-file dataset splits.
Each extxyz file is split independently to ensure equal representation.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mace.cli import run_train

from mylib.ml.datasets import create_dirs, ElementFilter, ForceFilter, Split

logging.basicConfig(level=logging.INFO)

SPLIT_SEED = 42
DEFAULT_VALID_FRACTION = 0.1


def train_model(model_path: Path, hyperparams: dict, flags: list[str]):
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

    filters = []
    if "max_force" in data:
        filters.append(ForceFilter(data["max_force"]))
    if "elements" in data:
        filters.append(ElementFilter(data["elements"]))

    model_dir = base / str(N_of_model)

    create_dirs(
        dataset_paths=[dataset_folder / file for file in data["train"]],
        filters=filters,
        output_dir=model_dir,
        split=Split(val=data.get("valid_fraction", DEFAULT_VALID_FRACTION)),
        split_seed=SPLIT_SEED,
    )

    hp = data["model_params"].copy()
    hp["seed"] = N_of_model * 100 + 100

    flags = data.get("flags", [])

    train_model(model_dir, hp, flags)


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

    return parser.parse_args()


if __name__ == "__main__":
    args = main_parser()
    main(args)