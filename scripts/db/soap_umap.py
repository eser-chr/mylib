from ase.io import read
import numpy as np
from dscribe.descriptors import SOAP
import pickle
import cupy as cp
import umap
from cuml.manifold import UMAP as cuUMAP

import ase
from ase.io import read, iread
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Tuple, List, Dict, Any
from time import perf_counter
import argparse
import logging

logging.basicConfig(level=logging.INFO)


def main_parser() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Creates metadata for a DB.")
    parser.add_argument("--folder", type=Path, help="Path to the DB folder")
    parser.add_argument("--file", type=Path, help="Path to the DB file")
    parser.add_argument("--soap_file", type=Path, help="File to the soap output")
    parser.add_argument("--umap_file", type=Path, help="File to save the umap data")
    parser.add_argument("--umap_obj_file", type=Path, help="pickle file for umap object")
    parser.add_argument("--device", type=str, help="cpu|gpu. Supports only NVIDIA")
    parser.add_argument("--reducer_file", type=Path, help="if specified loads the umap from that file")
    parser.set_defaults(to_plot=True, to_save=True)
    args = parser.parse_args()

    if args.soap_file is None:
        args.soap_file = Path("./soap.npy")
    if args.umap_file is None:
        args.umap_file = Path("./umap.npy")
    if args.umap_obj_file is None:
        args.umap_obj_file = Path("./umap_model.pkl")
    if args.device is None:
        args.device = "cpu"

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

    return np.vstack(total_descriptors).astype(np.float32)


def calc_umap_cpu(descriptors: npt.NDArray):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean"
    )
    embedding = reducer.fit_transform(descriptors)
    return embedding, reducer


def calc_umap_gpu(descriptors: npt.NDArray):
    descriptors_gpu = cp.asarray(descriptors, dtype=cp.float32)
    reducer = cuUMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        output_type="cupy",
    )
    embedding_gpu = reducer.fit_transform(descriptors_gpu)
    embedding = cp.asnumpy(embedding_gpu)
    return embedding, reducer


def main():
    logging.info("# calc_soap_umap.py")
    start = perf_counter()
    args = main_parser()

    if args.file is None:
        if args.folder is None:
            logging.error("File and folder args are none")
            return
        files = list(args.folder.rglob("*.extxyz"))
    else:
        files = [args.file]

    descriptors = calc_total_soap(files)

    if args.reducer_file:
        with open(args.reducer_file, "rb") as f:
            reducer = pickle.load(f)

        if args.device == "gpu":
            descriptors_gpu = cp.asarray(descriptors, dtype=cp.float32)
            umap_gpu = reducer.transform(descriptors_gpu)
            umap_data = cp.asnumpy(umap_gpu)
        else:
            umap_data = reducer.transform(descriptors)
    else:
        if args.device == "gpu":
            logging.info("GPU acceleration")

            umap_data, umap_obj = calc_umap_gpu(descriptors)
        else:
            umap_data, umap_obj = calc_umap_cpu(descriptors)

        with open(args.umap_obj_file, "wb") as f:
            pickle.dump(umap_obj, f)

    if args.soap_file:
        np.save(args.soap_file, descriptors)

    if args.umap_file:
        np.save(args.umap_file, umap_data)

    end = perf_counter()
    logging.info(f"Done in {end-start} seconds!")


if __name__ == "__main__":
    main()
