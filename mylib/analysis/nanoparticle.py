from ase import Atom, Atoms
import numpy as np
import numpy.typing as npt
from ase.neighborlist import NeighborList
from scipy.spatial import ConvexHull
from scipy.stats import entropy
from scipy.spatial import distance_matrix
import networkx as nx



def warren_cowley_binary(atoms: Atoms, cutoff, species_A="Cu", species_B="Ga"):
    symbols = np.array(atoms.get_chemical_symbols())
    N = len(atoms)

    # Global concentration of B
    c_B = np.sum(symbols == species_B) / N

    # Build neighbor list (no PBC assumed)
    cutoffs = [cutoff / 2.0] * N
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    AB_neighbors = 0
    total_neighbors_A = 0

    for i in range(N):
        if symbols[i] != species_A:
            continue

        indices, offsets = nl.get_neighbors(i)

        total_neighbors_A += len(indices)
        AB_neighbors += np.sum(symbols[indices] == species_B)

    if total_neighbors_A == 0:
        return np.nan

    P_AB = AB_neighbors / total_neighbors_A
    alpha = 1.0 - P_AB / c_B

    return alpha


def calc_asphericity(atoms: Atoms):
    points = atoms.get_positions()
    hull = ConvexHull(points)
    normals = hull.equations[:, :3]

    def ref_pdf(x):
        return 1 / (2 * np.sqrt(1 - x**2))

    kl_total = 0
    for i in range(3):
        hist, bins = np.histogram(normals[:, i], bins=30, range=(-1, 1), density=True)
        centers = 0.5 * (bins[1:] + bins[:-1])
        p = hist / hist.sum()
        q = ref_pdf(centers)
        q /= q.sum()
        kl_total += entropy(p, q)
    return kl_total


def calc_depth(frame: Atoms) -> npt.NDArray:
    pos = frame.get_positions()
    symbols = np.array(frame.get_chemical_symbols())

    ga_pos = pos[symbols == "Ga"]

    hull = ConvexHull(pos)

    # Extract plane equations
    normals = hull.equations[:, :3]  # shape (F, 3)
    offsets = hull.equations[:, 3]  # shape (F,)

    # Precompute normal norms (shape: F,)
    norm_normals = np.linalg.norm(normals, axis=1)

    # Compute all signed distances at once
    # Result shape: (n_Ga, F)
    distances = np.abs(ga_pos @ normals.T + offsets) / norm_normals

    # Minimum distance per Ga atom
    depths = np.min(distances, axis=1)

    return depths


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


def calc_ga_chains(frame: Atoms):
    _, components = classify_clusters(frame.get_positions(), 2.0)
    return len(components)
