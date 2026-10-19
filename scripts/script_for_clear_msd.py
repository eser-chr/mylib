import ase
from ase import Atoms, Atom
from ase.io import read, iread, write
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix
import networkx as nx
from typing import List

import numpy as np
from typing import Dict, List, Tuple
from ase.io import read

from time import perf_counter

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s -%(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def classify_clusters(positions, cutoff):
    """
    positions: (n_atoms, 3)
    cutoff: distance threshold (Å, same units as positions)
    Returns: list of cluster indices (component ID per atom)
    """
    # Compute pairwise distances
    dists = distance_matrix(positions, positions)

    # Build adjacency matrix (True if atoms are bonded)
    adjacency = (dists < cutoff) & (dists > 0)

    # Build graph
    G = nx.Graph()
    n_atoms = len(positions)
    G.add_nodes_from(range(n_atoms))
    for i in range(n_atoms):
        neighbors = np.where(adjacency[i])[0]
        for j in neighbors:
            G.add_edge(i, int(j))

    # Find connected components
    components = list(nx.connected_components(G))

    # Assign component IDs
    cluster_id = np.zeros(n_atoms, dtype=int)
    for cid, comp in enumerate(components):
        for atom_idx in comp:
            cluster_id[atom_idx] = int(cid)

    return cluster_id, components


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


def load_traj(traj_path: str | Path) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    traj: List[Atoms] = read(traj_path, index=":")  # type: ignore
    masses = traj[0].get_masses().astype(float)
    symbols = np.array(traj[0].get_chemical_symbols())
    pos = np.asarray([f.positions for f in traj], dtype=float)
    return traj, masses, symbols, pos


def classify_nnpt_all(pos: np.ndarray, cluster_cut: float) -> np.ndarray:
    n_frames = pos.shape[0]
    CIDs = np.asarray(
        [classify_clusters(pos[i], cluster_cut)[0] for i in range(n_frames)], dtype=int
    )
    return CIDs == 0


def compute_COM(pos: np.ndarray, masses: np.ndarray, is_nnpt: np.ndarray) -> np.ndarray:
    n_frames = pos.shape[0]
    COM = np.full((n_frames, 3), np.nan, dtype=float)
    for t in range(n_frames):
        mask = is_nnpt[t]
        if mask.any():
            m = masses[mask][:, None]
            p = pos[t, mask]
            COM[t] = (p * m).sum(0) / m.sum()
    return COM


def compute_omega(
    pos: np.ndarray, COM: np.ndarray, masses: np.ndarray, is_nnpt: np.ndarray, dt: float
) -> np.ndarray:
    n_frames = pos.shape[0]
    OMEGA = np.full((n_frames - 1, 3), np.nan, dtype=float)
    for t in range(n_frames - 1):
        mask = is_nnpt[t] & is_nnpt[t + 1]
        if not mask.any():
            continue
        m = masses[mask][:, None]
        r1 = pos[t, mask] - COM[t]
        r2 = pos[t + 1, mask] - COM[t + 1]
        v = (r2 - r1) / dt
        L_tot = (np.cross(r1, v) * m).sum(0)
        I = np.zeros((3, 3), dtype=float)
        for ri, mi in zip(r1, m.flatten()):
            I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
        OMEGA[t] = np.linalg.solve(I, L_tot)
    return OMEGA


def integrate_clean_positions(
    pos: np.ndarray, COM: np.ndarray, OMEGA: np.ndarray, is_nnpt: np.ndarray, dt: float
) -> np.ndarray:
    n_frames = pos.shape[0]
    v = np.diff(pos, axis=0) / dt
    v_clean = v.copy()
    for t in range(n_frames - 1):
        mask = is_nnpt[t] & is_nnpt[t + 1]
        if not mask.any():
            continue
        r1 = pos[t, mask] - COM[t]
        w = OMEGA[t]
        v_rot = np.cross(np.broadcast_to(w, r1.shape), r1)
        v_clean[t, mask] = v[t, mask] - v_rot
    pos_clean = np.empty_like(pos)
    pos_clean[0] = pos[0]
    for t in range(1, n_frames):
        pos_clean[t] = pos_clean[t - 1] + v_clean[t - 1] * dt
    return pos_clean


def msd_by_elements(
    pos_clean: np.ndarray, symbols: np.ndarray, persistent_mask: np.ndarray, dt: float
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    times = np.arange(pos_clean.shape[0], dtype=float) * dt
    ref = pos_clean[0, persistent_mask]
    dr = pos_clean[:, persistent_mask] - ref[None, :, :]
    sq = np.einsum("tnc,tnc->tn", dr, dr)
    els = symbols[persistent_mask]
    uniq = np.unique(els)
    out: Dict[str, np.ndarray] = {}
    for el in uniq:
        idx = np.where(els == el)[0]
        out[el] = sq[:, idx].mean(axis=1)
    return times, out


def rotation_corrected_msd_per_element(
    traj_path: str | Path, cluster_cut: float = 3.0, dt: float = 1.0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    traj, masses, symbols, pos = load_traj(traj_path)
    is_nnpt = classify_nnpt_all(pos, cluster_cut)
    persistent_mask = np.all(is_nnpt, axis=0)
    if not persistent_mask.any():
        raise RuntimeError("No atoms persist in nnpt across all frames.")
    COM = compute_COM(pos, masses, is_nnpt)
    OMEGA = compute_omega(pos, COM, masses, is_nnpt, dt)
    pos_clean = integrate_clean_positions(pos, COM, OMEGA, is_nnpt, dt)
    return msd_by_elements(pos_clean, symbols, persistent_mask, dt)


def main():
    base_dir = Path("/home/christoforos.eseroglou/Desktop/GRAND_SCAN/GRAND_SCAN")
    output_dir = Path("/home/christoforos.eseroglou/Desktop/GRAND_SCAN/CLEAN_MSD")

    for dir in base_dir.iterdir():
        assert dir.is_dir()
        name = dir.name
        file = dir / "traj.extxyz"

        output_subdir = output_dir / name
        output_subdir.mkdir(exist_ok=True)
        output_file = output_subdir / "msd.pickle"
        times, msd = rotation_corrected_msd_per_element(str(file))
        with open(output_file, "wb") as f:
            pickle.dump({"times": times, "msd": msd}, f)

    # # nOs = [100, 150, 200]
    # # Ts = [300, 500, 700, 900]
    # # pGas = [8, 16, 22]

    # # combos = zip(nOs, Ts, pGas)

    # for combo in combos:
    #     nO, pGa, T = combo
    #     subdir = base_dir/f"{nO}_{pGa}_{T}"
    #     file = subdir/"traj.extxyz"
    #     output = subdir/"clear_msd.pickle"
    #     with open(output, "wb") as f:
    #         pickle.dump({"times": times, "msd": msd}, f)
