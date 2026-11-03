
import numpy as np
from typing import List, Optional, Tuple, Union, Any, Dict
from ase import Atoms
import freud


def get_pair_cn_with_com_cutoff(
    frames: List[Atoms], 
    cutoff: float, 
    r_min: float, 
    r_max: float
) -> Tuple[List[str], npt.NDArray[np.float64]]:
    elems = sorted(set(frames[0].get_chemical_symbols()))
    nF, nE = len(frames), len(elems)
    CN = np.zeros((nF, nE, nE), dtype=float)

    for t, at in enumerate(frames):
        box = freud.box.Box.from_matrix(at.cell.array, tuple(at.pbc))
        pos = at.positions.astype(np.float32)
        types = np.array(at.get_chemical_symbols())
        masses = at.get_masses()  # Get atomic masses

        # Calculate the center of mass (COM)
        total_mass = np.sum(masses)
        com = np.sum(pos.T * masses, axis=1) / total_mass  # COM calculation

        # Filter particles based on r_min and r_max from the COM
        distances = np.linalg.norm(pos - com, axis=1)
        within_range_indices = np.where((distances >= r_min) & (distances <= r_max))[0]

        # Filtered positions and types
        filtered_pos = pos[within_range_indices]
        filtered_types = types[within_range_indices]

        for ai, A in enumerate(elems):
            A_pos = filtered_pos[filtered_types == A]
            NA = len(A_pos)
            if NA == 0:
                continue
            for bi, B in enumerate(elems):
                B_pos = filtered_pos[filtered_types == B]
                if len(B_pos) == 0:
                    continue
                
                # Create an AABB query for B positions
                aq = freud.locality.AABBQuery.from_system((box, B_pos))
                
                # Query neighbors of A_pos within the cutoff
                nlist = aq.query(
                    A_pos, {"r_max": cutoff, "exclude_ii": (A == B)}
                ).toNeighborList()
                
                # Count the number of neighbors for each A atom
                counts = np.bincount(nlist.query_point_indices, minlength=NA)
                
                # Calculate the coordination number
                CN[t, ai, bi] = counts.mean() if len(nlist.query_point_indices) > 0 else 0.0

    return elems, CN

def get_partial_cn_distributions(frames: List[Atoms], cutoff: float = 3.0) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculates CN for specific pairs (e.g., Cu-Cu, Cu-O).
    Returns: { 'Cu': {'Cu': [2, 3, ...], 'O': [1, 0, ...]}, 'O': {...} }
    """
    # Initialize nested structure
    results = {}

    for frame in frames:
        symbols = np.array(frame.get_chemical_symbols())
        unique_elements = np.unique(symbols)
        
        # Get neighbor indices
        i_idx, j_idx = neighbor_list('ij', frame, cutoff)
        
        # Map indices to symbols
        center_symbols = symbols[i_idx]
        neighbor_symbols = symbols[j_idx]

        for spec_i in unique_elements:
            if spec_i not in results:
                results[spec_i] = {}
            
            # Mask for all pairs where the center atom is spec_i
            mask_i = (center_symbols == spec_i)
            
            for spec_j in unique_elements:
                if spec_j not in results[spec_i]:
                    results[spec_i][spec_j] = []

                # Mask for pairs where neighbor is spec_j AND center is spec_i
                mask_ij = mask_i & (neighbor_symbols == spec_j)
                
                # Filter the i_idx to only those specific pairs
                relevant_centers = i_idx[mask_ij]
                
                # Count neighbors for EVERY atom of spec_i
                # We use minlength=len(frame) then slice by symbols==spec_i 
                # to ensure we count atoms that have ZERO neighbors of type spec_j
                counts = np.bincount(relevant_centers, minlength=len(frame))
                spec_i_counts = counts[symbols == spec_i]
                
                results[spec_i][spec_j].extend(spec_i_counts)

    return {idx: {jdx: np.array(vals) for jdx, vals in j_dict.items()} 
            for idx, j_dict in results.items()}


def clean_trajectory(
    trajectory: List[Atoms], cutoff: float
) -> tuple[List[Atoms], List[Atoms]]:
    F = len(trajectory)
    def _largest_cluster_mask(positions: np.ndarray, cutoff: float) -> np.ndarray:
        N = len(positions)
        tree = KDTree(positions)
        pairs = tree.query_pairs(cutoff, output_type='ndarray')  # (M, 2)
        if len(pairs) == 0:
            mask = np.zeros(N, dtype=bool)
            mask[0] = True
            return mask
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(rows), dtype=bool)
        adj = csr_matrix((data, (rows, cols)), shape=(N, N))
        _, labels = connected_components(adj, directed=False)
        return labels == np.argmax(np.bincount(labels))

    
    N = len(trajectory[0])
    masks = np.empty((F, N), dtype=bool)

    for i, atoms in enumerate(trajectory):
        masks[i] = _largest_cluster_mask(atoms.positions, cutoff)
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1}/{F} frames")

    per_frame = [atoms[masks[i]] for i, atoms in enumerate(trajectory)]
    global_mask = masks.all(axis=0)
    global_cleaned = [atoms[global_mask] for atoms in trajectory]

    return per_frame, global_cleaned


def message_passing_analysis(frame: Atoms, cutoff: float, species_order: List[str]):
    num_atoms = len(frame)
    symbols = np.array(frame.get_chemical_symbols())
    
    # 1. Create One-Hot Encodings for each atom
    # e.g., if species_order is [Cu, Ga, O], Cu becomes [1, 0, 0]
    identity_map = {spec: i for i, spec in enumerate(species_order)}
    v = np.zeros((num_atoms, len(species_order)))
    for spec in species_order:
        v[symbols == spec, identity_map[spec]] = 1.0

    # 2. Get the Graph (Adjacency)
    i_idx, j_idx, dists = neighbor_list('ijd', frame, cutoff)
    
    # 3. Calculate Degree (Normalization factor) for each sender
    # Based on your rule: "Each atom prepares a message... depending on edges it normalizes"
    degrees = np.bincount(i_idx, minlength=num_atoms)
    # Avoid division by zero for isolated atoms
    norm_factor = np.where(degrees > 0, 1.0 / degrees, 0)
    
    # 4. Prepare the normalized messages being sent
    # Each atom j sends its identity divided by its own degree
    normalized_messages = v[j_idx] * norm_factor[j_idx][:, np.newaxis]

    # 5. Distance Weighting (Optional: 1/r^2 or exponential decay)
    # weights = np.exp(-dists) 
    # normalized_messages *= weights[:, np.newaxis]

    # 6. Aggregate at the receiver (atom i)
    # We sum the messages coming into atom i
    passed_features = np.zeros((num_atoms, len(species_order)))
    for d in range(len(species_order)):
        passed_features[:, d] = np.bincount(i_idx, weights=normalized_messages[:, d], minlength=num_atoms)

    return passed_features




def substitute_homogeneously(atoms, symbol_from='Cu', symbol_to='Ga', n_sub=1):
    positions = atoms.get_positions()
    indices = np.arange(len(atoms))

    # Track which atoms can be replaced
    copper_indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol_from]
    # print(copper_indices)

    # Pick the first substitution randomly
    chosen = [np.random.choice(copper_indices)]
    copper_indices.remove(chosen[0])

    for _ in range(n_sub - 1):
        # For each candidate, compute min distance to already chosen
        distances = []
        for i in copper_indices:
            dists = np.linalg.norm(positions[i] - positions[chosen], axis=1)
            distances.append(np.min(dists))
        
        # Choose atom that is farthest from all current Ga atoms
        best_index = copper_indices[np.argmax(distances)]
        chosen.append(best_index)
        copper_indices.remove(best_index)

    # Perform substitution
    for idx in chosen:
        atoms[idx].symbol = symbol_to
    
    return atoms

def add_random_oxygen_hull(atoms, N=5, bondlength=1.3):
    pos = atoms.get_positions()
    center = pos.mean(axis=0)
    hull = ConvexHull(pos)
    surface_idx = np.unique(hull.vertices)
    if len(surface_idx) < N:
        raise ValueError("Not enough surface atoms on convex hull.")

    chosen = np.random.choice(surface_idx, N, replace=False)
    new_positions = []

    for i in chosen:
        face_normals = []
        for simplex in hull.simplices:
            if i in simplex:
                pts = pos[simplex]
                n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                n /= np.linalg.norm(n)
                face_center = pts.mean(axis=0)
                if np.dot(n, face_center - center) < 0:
                    n = -n
                face_normals.append(n)

        normal = np.mean(face_normals, axis=0)
        normal /= np.linalg.norm(normal)
        new_positions.append(pos[i] + bondlength * normal)

    O_atoms = Atoms('O' * N, positions=new_positions)
    return atoms + O_atoms



def create_force_histogram(
    atoms: List[Atoms],
    key: str = "forces",
    bins: int = 120,
    force_range: Optional[Tuple[float, float]] = None,
    density: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute histograms of the Cartesian force components and force magnitudes.

    Returns a dict with the shared bin edges and the per-component counts, so
    that several sets (database, production MD, ...) binned on the same edges
    can be compared or plotted downstream.

    Keys:
        bin_edges      (bins + 1,)  shared edges for the x, y, z components
        counts         (3, bins)    histogram of Fx, Fy, Fz
        mag_bin_edges  (bins + 1,)  edges for the magnitude histogram
        mag_counts     (bins,)      histogram of |F|
        forces         (N, 3)       the raw stacked forces
    """
    def _extract_forces(atoms: List[Atoms], key: str = "forces") -> np.ndarray:
        """Stack per-atom force vectors from a list of Atoms into an (N_atoms, 3) array.

        Reads ``atoms.arrays[key]`` first, falling back to the attached calculator's
        stored results. Frames without force data are skipped.
        """
        collected = []
        for i, a in enumerate(atoms):
            f = a.arrays.get(key)
            if f is None and a.calc is not None:
                f = a.calc.results.get(key)
            if f is None:
                continue
            f = np.asarray(f, dtype=float)
            if f.shape != (len(a), 3):
                raise ValueError(f"frame {i}: expected forces of shape {(len(a), 3)}, got {f.shape}")
            collected.append(f)

        if not collected:
            raise ValueError(f"no frames contained a '{key}' array")

        return np.concatenate(collected, axis=0)
    
    
    forces = _extract_forces(atoms, key=key)
    magnitudes = np.linalg.norm(forces, axis=1)

    if force_range is None:
        vmax = np.abs(forces).max() * 1.05
        force_range = (-vmax, vmax)

    bin_edges = np.linspace(*force_range, bins + 1)
    counts = np.stack(
        [np.histogram(forces[:, i], bins=bin_edges, density=density)[0] for i in range(3)]
    )

    mag_counts, mag_bin_edges = np.histogram(
        magnitudes, bins=bins, range=(0.0, magnitudes.max() * 1.05), density=density
    )

    return {
        "bin_edges": bin_edges,
        "counts": counts,
        "mag_bin_edges": mag_bin_edges,
        "mag_counts": mag_counts,
        "forces": forces,
    }


def build_ga_graph(atoms: Atoms, cutoff: float):
    """
    Build a graph where:
      - Nodes: Ga atom indices
      - Edge between Ga_i and Ga_j if they share at least one O neighbor
      - Edge attribute 'shared_oxygen' contains O indices

    Parameters
    ----------
    atoms : ASE Atoms
    cutoff : float
        Ga-O cutoff distance (Å)

    Returns
    -------
    G : networkx.Graph
    """

    symbols = np.array(atoms.get_chemical_symbols())

    ga_indices = np.where(symbols == "Ga")[0]
    o_indices  = np.where(symbols == "O")[0]

    # Compute Ga-O neighbor list (PBC-aware)
    i_list, j_list = neighbor_list(
        'ij',
        atoms,
        {('Ga', 'O'): cutoff}
    )

    # Map: O_index -> list of Ga neighbors
    from collections import defaultdict
    oxygen_to_ga = defaultdict(list)

    for i, j in zip(i_list, j_list):
        # ensure i = Ga, j = O
        if symbols[i] == "Ga":
            ga_idx, o_idx = i, j
        else:
            ga_idx, o_idx = j, i

        oxygen_to_ga[o_idx].append(ga_idx)

    # Build graph
    G = nx.Graph()

    # Add Ga nodes
    for ga in ga_indices:
        G.add_node(ga)

    # For each oxygen, connect all Ga pairs attached to it
    for o_idx, ga_list in oxygen_to_ga.items():

        if len(ga_list) < 2:
            continue

        for a in range(len(ga_list)):
            for b in range(a + 1, len(ga_list)):

                ga_i = ga_list[a]
                ga_j = ga_list[b]

                if G.has_edge(ga_i, ga_j):
                    G[ga_i][ga_j]['shared_oxygen'].append(o_idx)
                else:
                    G.add_edge(
                        ga_i,
                        ga_j,
                        shared_oxygen=[o_idx]
                    )

    return G