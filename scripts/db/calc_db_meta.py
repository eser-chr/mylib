"""
A script to calculate some metadata for the database of single points.
So far the only metric is the number of structure in each extxyz file.
"""

import json
import argparse
import pandas as pd
from ase import Atoms
from ase.io import read
from pathlib import Path
from typing import List, Tuple, Dict, Any, cast
from time import time
import numpy as np
import numpy.typing as npt
import logging
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logging.basicConfig(level=logging.INFO)
from joblib import Parallel, delayed

HISTOGRAM = Dict[str, Dict[str, npt.NDArray]]


def main_parser() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Creates metadata for a DB.")
    parser.add_argument("ifolder", type=Path, help="Path to the DB folder")
    parser.add_argument(
        "--metadata-folder",
        type=Path,
        help="Path to the folder where metadata will be written (default: <ifolder>/meta)",
    )
    parser.add_argument(
        "--to-plot",
        action="store_true",
        help="Create plots (default: off)",
    )
    parser.add_argument(
        "--no-plot",
        dest="to_plot",
        action="store_false",
        help="Disable plots",
    )
    parser.add_argument(
        "--to-save",
        action="store_true",
        help="Save histograms as pickle files (default: off)",
    )
    parser.add_argument(
        "--no-save",
        dest="to_save",
        action="store_false",
        help="Disable saving histograms",
    )

    parser.set_defaults(to_plot=True, to_save=True)

    args = parser.parse_args()

    if args.metadata_folder is None:
        args.metadata_folder = args.ifolder / "meta"

    return vars(args)


def get_size_of_file(file: Path) -> int:
    return int(file.stat().st_size / 1000)


def get_num_of_atomic_configs(atoms: List[Atoms]) -> int:
    return len(atoms)


def is_neccessary_info(atoms) -> bool:
    necessary_keys = {"numbers", "positions", "REF_forces"}
    necessary_info = {"REF_energy"}
    for atom in atoms:
        if not necessary_keys.issubset(set(atom.arrays.keys())):
            return False
        if not necessary_info.issubset(set(atom.info.keys())):
            return False

    return True


def get_redundant_arrays(atoms):
    necessary_keys = {"numbers", "positions", "REF_forces"}
    unnecessary_keys = set()
    for atom in atoms:
        unnecessary_keys |= set(atom.arrays.keys())
    unnecessary_keys = unnecessary_keys - necessary_keys
    return [f"array:{k}" for k in unnecessary_keys]


def get_histogram_of_forces(atoms, is_file_ok: bool) -> HISTOGRAM:
    if not is_file_ok:
        return {}
    bins = np.arange(-40, 40 + 0.4, 0.4)
    # pdf, edges = np.histogram(all_forces[:, i], bins=bins, density=True)

    all_forces = np.vstack([a.arrays["REF_forces"] for a in atoms])

    hist_data = {}
    for i, comp in enumerate(["Fx", "Fy", "Fz"]):
        pdf, edges = np.histogram(
            all_forces[:, i], bins=bins, density=True
        )  # dens is the PDF

        N = len(all_forces)
        hist_data[comp] = {"N": N, "pdf": pdf, "edges": edges}

    return hist_data


def get_histogram_of_force_magnitudes(atoms, is_file_ok: bool) -> HISTOGRAM:

    if not is_file_ok or len(atoms) == 0:
        return {}

    all_forces = np.vstack([a.arrays["REF_forces"] for a in atoms])
    force_magnitudes = np.linalg.norm(all_forces, axis=1)
    bins = np.arange(0, 40 + 0.4, 0.4)  # from 0 to 40 eV/Å

    pdf, edges = np.histogram(force_magnitudes, bins=bins, density=True)
    N = len(force_magnitudes)

    return {"mag": {"N": N, "pdf": pdf, "edges": edges}}


def save_histogram(histogram: HISTOGRAM, output: Path) -> None:
    with open(output, "wb") as f:
        pickle.dump(histogram, f)


def plot_histogram(histogram: HISTOGRAM) -> Figure:
    fig, ax = plt.subplots(1, 1)

    # for i, comp in enumerate(["Fx", "Fy", "Fz", "mag"]):
    for i, comp in enumerate(histogram.keys()):
        counts = histogram[comp]["pdf"]
        edges = histogram[comp]["edges"]
        N = histogram[comp]["N"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, label=comp)

    if "mag" in histogram.keys():
        ax.set(
            title="Force Histogram",
            xlabel="edges",
            ylabel="counts",
            yscale="log",
            xlim=[0, 40],
            ylim=[-0.01, 1.01],
        )
    else:

        ax.set(
            title="Force Histogram",
            xlabel="edges",
            ylabel="counts",
            yscale="log",
            xlim=[-40, 40],
            ylim=[-0.01, 1.01],
        )

    ax.legend()
    return fig


def read_extxyz_file(file: Path) -> List[Atoms]:
    atoms = read(file, index=":")
    if isinstance(atoms, Atoms):
        return [atoms]
    if not isinstance(atoms, list) or any(not isinstance(a, Atoms) for a in atoms):
        logging.error(f"Reading of file {file} resulted in wrong data structure")
    return atoms


def single_file_job(
    file: Path, meta_folder: Path, to_save: bool = True, to_plot: bool = True
) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info(f"Work in file {file}")
    try:
        size_KB = get_size_of_file(file)
        atoms = read_extxyz_file(file)
        natoms = get_num_of_atomic_configs(atoms)
        reduantant_arrays = get_redundant_arrays(atoms)
        is_file_ok = is_neccessary_info(atoms)

        histogram = get_histogram_of_forces(atoms, is_file_ok)

        mag_histogram = get_histogram_of_force_magnitudes(atoms, is_file_ok)
        if to_save:
            save_histogram(histogram, meta_folder / f"{file.stem}.pickle")
        if to_plot:
            fig = plot_histogram(histogram)
            fig.savefig(meta_folder / f"{file.stem}.png", dpi=300)
            fig = plot_histogram(mag_histogram)
            fig.savefig(meta_folder / f"{file.stem}_mag.png", dpi=300)

        if not is_file_ok:
            logging.warning(f"I found info missing from file {file}")
        if len(reduantant_arrays):
            logging.warning(f"I found redundant arrays in file {file}")
        return {"path": file, "num_configs": natoms, "size_kB": size_KB}

    except Exception as e:
        logging.error(f" Found exception during main run at file {file} \n {e}")
        return {"path": file, "num_configs": -1, "size_kB": -1}


def create_json_file(files: List[Path], meta_folder: Path):
    for path in files:
        print(path, path.parent)
    train_paths = [str(path.name) for path in files if path.parent.name == "train"]
    validation_paths = [
        str(path.name) for path in files if path.parent.name == "validation"
    ]

    with open(meta_folder / "description.json", "w") as f:
        json.dump({"train": train_paths, "validation": validation_paths}, f)


def main():

    args = main_parser()
    metdata: Path = args["metadata_folder"]
    metdata.mkdir(exist_ok=True)
    to_save: bool = args["to_save"]
    to_plot: bool = args["to_plot"]

    logging.info(f"Calc meta for {args['ifolder']}")

    start = time()
    files = list(args["ifolder"].rglob("*.extxyz"))
    RES = Parallel(n_jobs=-1)(
        delayed(single_file_job)(file, args["metadata_folder"], to_save, to_plot)
        for file in files
    )

    # Collect results to a csv file
    df = pd.DataFrame(RES)  # type: ignore
    df.to_csv(args["metadata_folder"] / "db_composition.csv", index=False)
    create_json_file(files, metdata)
    end = time()

    logging.info(f"DONE in {end-start:.2f} seconds !")


if __name__ == "__main__":
    main()
