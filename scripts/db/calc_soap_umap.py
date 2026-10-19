from ase.io import read
import numpy as np
from dscribe.descriptors import SOAP
import pickle
import cupy as cp
import umap

import ase
from ase.io import read, iread
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Tuple, List, Dict, Any
from time import perf_counter
import umap
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def main_parser() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Creates metadata for a DB.")
    parser.add_argument("--folder", type=Path, help="Path to the DB folder")
    parser.add_argument("--file", type=Path, help="Path to the DB file")

    parser.add_argument("--soap_file", type=Path, help="File to the soap output")
    parser.add_argument("--umap_file", type=Path, help="File to save the umap data")
    parser.add_argument(
        "--umap_obj_file", type=Path, help="pickle file for umap object"
    )
    parser.add_argument("--device", type=str, help="cpu|gpu. Supports only NVIDIA")
    parser.add_argument(
        "--reducer_file", type=Path, help="if specified loads the umap from that file"
    )

    parser.set_defaults(to_plot=True, to_save=True)

    args = parser.parse_args()

    if args.soap_file is None:
        args.soap_file = Path("./soap.np")

    if args.umap_file is None:
        args.umap_file = Path("./umap.np")

    if args.umap_obj_file is None:
        args.umap_obj_file = Path("./umap_model.pkl")

    if args.device is None:
        args.device = "cpu"

    # return vars(args)
    return args


def calc_total_soap(files: List[Path]) -> npt.NDArray:

    soap = SOAP(
        species=["Cu", "Ga", "O"],
        r_cut=5.0,
        n_max=8,
        l_max=6,
        sigma=0.5,
        periodic=True,
        sparse=False,
    )

    total_descriptors = []
    for file in files:
        atoms = read(file, index=":")
        for atom in atoms:
            tmp = soap.create(atom)
            total_descriptors.append(tmp)

    return np.vstack(total_descriptors)


def calc_umap(descriptors: npt.NDArray) -> Tuple[npt.NDArray, umap.UMAP]:

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean"
    )

    return reducer.fit_transform(descriptors), reducer


def main():
    logging.info("# calc_soap_umap.py")
    logging.info("Enter script")
    start = perf_counter()
    args = main_parser()

    if args.file is None:
        if args.folder is None:
            logging.error(f"File and folder args are none !!!")

        files = list(args.folder.rglob("*.extxyz"))

    files = [args.file]

    logging.info("Extracted list of file")
    logging.info("======================")
    for file in files:
        logging.info(f" >    {file}")

    descriptors = calc_total_soap(files)
    logging.info("calculated descriptors done")

    if args.reducer_file:
        with open(args.reducer_file, "rb") as f:
            reducer = pickle.load(f)
        assert (type(reducer) == umap.UMAP)
        umap_data = reducer.transform(descriptors)
    else:
        if args.device == "cpu":
            logging.info("calc umap via cpu")
            umap_data, umap_obj = calc_umap(descriptors)
        elif args.device == "gpu":
            logging.warning("GPU UMAP not implemented yet, falling back to CPU")
        with open("umap_model.pkl", "wb") as f:
            pickle.dump(umap_obj, f)

    logging.info("calculated umap done")

    if args.soap_file:
        logging.info("Save soap")
        np.save(args.soap_file, descriptors)

    if args.umap_file:
        logging.info("Save umap")
        np.save(args.umap_file, umap_data)
    end = perf_counter()

    logging.info(f"Done in {end-start} seconds!")


if __name__ == "__main__":
    main()
