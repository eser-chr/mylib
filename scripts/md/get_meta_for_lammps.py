"""
Christoforos Eseroglou
24/9/2025

I call this script to calculate some metadata after running some lammps md.

Metadata:
-> extratcion of traj to extxyz
-> uncertainty same file for extxyz and a plot with the maximum and minimum uncertainty

-> energies
-> com trajectory
-> msd for each species
-> coordination number
-> prdf

"""

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle

from ase.io import read, write
from ase.geometry import get_angles
from ase.geometry.analysis import Analysis
from ase.neighborlist import NeighborList
from mace.calculators import MACECalculator

import freud
from typing import List, Tuple, Dict, Optional
from ase import Atoms
import logging
import argparse


logging.basicConfig(level=logging.INFO)


def append_uncertainty_to_traj(
    traj: List[Atoms], uncertainties: np.ndarray, uncertainty_tag: str = "uncertainty"
) -> None:
    """Attach per-atom uncertainties in place; errors if shapes disagree."""
    arr = np.asarray(uncertainties)
    expected = (len(traj), len(traj[0]))
    if arr.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {arr.shape}")
    for snap, u in zip(traj, arr):
        snap.arrays[uncertainty_tag] = u


def get_trajectory(path: Path) -> List[Atoms]:
    try:
        atoms = read(path, index=":")
    except Exception as e:
        logging.error(f"Could not read trajectory {path}: {e}")
        raise
    if isinstance(atoms, Atoms):
        logging.warning("trajectory consists of only one snapshot")
        return [atoms]
    if not atoms:
        raise ValueError(f"No frames read from {path}")
    return atoms


def get_rdf(frames: List[Atoms], bins: int, r_max: float):
    RES = np.zeros((len(frames), bins), dtype=np.float64)
    rdf0 = freud.density.RDF(bins=bins, r_max=r_max)
    box0 = freud.box.Box.from_matrix(
        frames[0].cell.array, tuple(frames[0].pbc))
    rdf0.compute((box0, frames[0].positions.astype(np.float32)))
    r = rdf0.bin_centers.copy()

    for i, at in enumerate(frames):
        rdf = freud.density.RDF(bins=bins, r_max=r_max)  # fresh instance
        box = freud.box.Box.from_matrix(at.cell.array, tuple(at.pbc))
        rdf.compute((box, at.positions.astype(np.float32)))
        RES[i] = rdf.rdf
    return r, RES


def get_partial_rdf(
    frames: List[Atoms], bins: int, r_max: float
) -> Tuple[np.ndarray, Dict[Tuple[str, str], npt.NDArray[np.float64]]]:
    elems = sorted(set(frames[0].get_chemical_symbols()))
    pairs = [(a, b) for a in elems for b in elems]

    edges = np.linspace(0.0, r_max, bins + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    shell = 4.0 * np.pi * r**2 * dr

    R: Dict[Tuple[str, str], npt.NDArray[np.float64]] = {
        p: np.zeros((len(frames), bins), float) for p in pairs
    }

    for t, at in enumerate(frames):
        box = freud.box.Box.from_matrix(at.cell.array, tuple(at.pbc))
        pos = at.positions.astype(np.float32)
        types = np.array(at.get_chemical_symbols())
        V = box.volume
        if V <= 0:
            continue

        for A, B in pairs:
            A_pos = pos[types == A]
            B_pos = pos[types == B]
            NA, NB = len(A_pos), len(B_pos)
            if NA == 0 or NB == 0:
                continue

            aq = freud.locality.AABBQuery.from_system((box, B_pos))
            nlist = aq.query(
                A_pos, {"r_max": r_max, "exclude_ii": (A == B)}
            ).toNeighborList()
            d = nlist.distances
            counts, _ = np.histogram(d, bins=edges)
            rho_B = NB / V
            R[(A, B)][t] = counts / (NA * shell * rho_B)
    return r, R


def get_pair_cn(
    frames: List[Atoms], cutoff: float
) -> Tuple[List[str], npt.NDArray[np.float64]]:
    elems = sorted(set(frames[0].get_chemical_symbols()))
    nF, nE = len(frames), len(elems)
    CN = np.zeros((nF, nE, nE), dtype=float)

    for t, at in enumerate(frames):
        box = freud.box.Box.from_matrix(at.cell.array, tuple(at.pbc))
        pos = at.positions.astype(np.float32)
        types = np.array(at.get_chemical_symbols())

        for ai, A in enumerate(elems):
            A_pos = pos[types == A]
            NA = len(A_pos)
            if NA == 0:
                continue
            for bi, B in enumerate(elems):
                B_pos = pos[types == B]
                if len(B_pos) == 0:
                    continue
                aq = freud.locality.AABBQuery.from_system((box, B_pos))
                nlist = aq.query(
                    A_pos, {"r_max": cutoff, "exclude_ii": (A == B)}
                ).toNeighborList()
                counts = np.bincount(nlist.query_point_indices, minlength=NA)
                CN[t, ai, bi] = counts.mean()
    return elems, CN


def compute_dcom(
    frames: List[Atoms],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ref = frames[0]
    P = np.stack([a.get_positions() for a in frames], axis=0)
    masses = np.asarray(ref.get_masses())
    msum = masses.sum()
    COM = (P * masses[None, :, None]).sum(axis=1) / msum
    DCOM = np.linalg.norm(COM - COM[0], axis=1)
    return COM, DCOM


def compute_partial_msd(frames: List[Atoms]) -> Dict[str, npt.NDArray[np.float64]]:
    ref = frames[0]
    sym = np.array(ref.get_chemical_symbols())
    unique_el, inv = np.unique(sym, return_inverse=True)
    el_indices = {el: np.where(inv == i)[0] for i, el in enumerate(unique_el)}
    P = np.stack([a.get_positions() for a in frames], axis=0)
    dr = P - P[0:1]
    sq = np.einsum("tnc,tnc->tn", dr, dr)
    MSD_RES = {el: sq[:, idx].mean(axis=1) for el, idx in el_indices.items()}
    return MSD_RES


def compute_partial_avg_distance_com(
    frames: List[Atoms], COM: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    ref = frames[0]
    sym = np.array(ref.get_chemical_symbols())
    unique_el, inv = np.unique(sym, return_inverse=True)
    el_indices = {el: np.where(inv == i)[0] for i, el in enumerate(unique_el)}
    P = np.stack([a.get_positions() for a in frames], axis=0)
    diff_to_com = P - COM[:, None, :]
    dist = np.linalg.norm(diff_to_com, axis=2)
    AVG_DISTANCE_RES = {el: dist[:, idx].mean(
        axis=1) for el, idx in el_indices.items()}
    return AVG_DISTANCE_RES


def compute_raw_tensors_per_sim(
    frames: List[Atoms], potentials: List[MACECalculator]
) -> Tuple[np.ndarray, np.ndarray]:
    Nframes = len(frames)
    Natoms = len(frames[0])
    Npot = len(potentials)

    E = np.empty((Nframes, Npot), dtype=np.float32)
    F = np.empty((Nframes, Npot, Natoms, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        for j, potential in enumerate(potentials):
            frame.calc = potential
            E[i, j] = frame.get_potential_energy()
            F[i, j] = frame.get_forces().astype(np.float32)
    F = F.mean(axis=3)  # reduce xyz
    return E, F


def process_raw_traj_tensors(E_fm, F_fma):
    E_mean = np.mean(E_fm, axis=1)  # Reduction of models
    E_std = np.std(E_fm, axis=1)

    F_fa = np.std(
        F_fma, axis=1
    )  # Reduction of models. Usually no interest in mean force
    F_mean = np.mean(
        F_fa, axis=1
        # Reduction in atoms. Get mean uncertainty per frame (as a whole system/image)
    )
    # Reduction in atoms. Get atom with max uncertainty.
    F_max = np.max(F_fa, axis=1)

    return (E_mean, E_std, F_fa, F_mean, F_max)


@dataclass
class Arguments:
    main_folder: Path
    mace_potential_ensemble: Path
    uncertainty_tag: str

    i_trajectory: Optional[Path] = None
    o_trajectory: Path = field(init=False)
    paths_of_valid_models: List[Path] = field(init=False)
    o_pickle: Path = field(init=False)

    def __post_init__(self):
        if not self.main_folder.exists():
            raise FileNotFoundError(
                f"Main folder not found: {self.main_folder}")
        if not self.mace_potential_ensemble.exists():
            raise FileNotFoundError(
                f"MACE ensemble folder not found: {self.mace_potential_ensemble}"
            )
        if not self.i_trajectory:
            self.i_trajectory = self.main_folder / "dump.vel"
            if not self.i_trajectory.exists():
                raise FileNotFoundError(
                    "Main trajectory with the expected name dump.vel not found"
                )

        self.o_trajectory = self.main_folder / "traj.extxyz"
        self.o_pickle = self.main_folder / "meta.pickle"

        self.paths_of_valid_models = list(
            self.mace_potential_ensemble.rglob("model.model")
        )
        if not self.paths_of_valid_models:
            raise FileNotFoundError(
                "No valid models found under the ensemble folder")


def main_parser() -> Arguments:
    parser = argparse.ArgumentParser(
        description="Creates metadata of a mlip generation."
    )
    parser.add_argument("main_folder", type=Path,
                        help="The path to the main folder.")
    parser.add_argument(
        "mace_potential_ensemble", type=Path, help="The path to the mace potential"
    )
    parser.add_argument(
        "uncertainty_tag",
        type=str,
        help="The tag stored in the extxyz file for the uncertainty.",
    )
    parser.add_argument(
        "--i_trajectory",
        type=Path,
        default = None,
        help="Optional path to the initial trajectory. If not specified it assumes a dump.vel exists. Otherwise it returns an error"
    )
    args = parser.parse_args()
    return Arguments(**vars(args))


def main(arguments: Arguments) -> None:
    # assert(type(arguments.i_trajectory)!=None)
    frames = get_trajectory(arguments.i_trajectory) #type:ignore
    com, dcom = compute_dcom(frames)
    p_msd = compute_partial_msd(frames)
    avg_distance = compute_partial_avg_distance_com(frames, com)
    p_rdf = get_partial_rdf(frames, 200, 4.0)
    cn = get_pair_cn(frames, 3.5)

    models = [
        MACECalculator(model_paths=str(model_path),
                       device="cuda", enable_cueq=True)
        for model_path in arguments.paths_of_valid_models
    ]

    E, F = compute_raw_tensors_per_sim(frames, models)
    E_mean, E_std, F_fa, F_mean, F_max = process_raw_traj_tensors(E, F)
    append_uncertainty_to_traj(
        frames, F_fa, uncertainty_tag=arguments.uncertainty_tag)

    write(arguments.o_trajectory, frames)

    with open(arguments.o_pickle, "wb") as fout:
        pickle.dump(
            {
                "E_mean": E_mean,
                "E_std": E_std,
                "F_mean": F_mean,
                "F_max": F_max,
                "dcom": dcom,
                "msd": p_msd,
                "avg": avg_distance,
                "p_rdf": p_rdf,
                "cn": cn,
            },
            fout,
        )


if __name__ == "__main__":
    logging.info("Start generation of meta for mace ensemble")
    args = main_parser()
    logging.info(type(args.i_trajectory))
    main(args)
    logging.info("Done with generation of meta")
