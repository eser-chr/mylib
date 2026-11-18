import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def remove_escaped_atoms(atoms: Atoms, max_gap: float = 6.5) -> Atoms:
    """Keep only the largest connected fragment."""
    if len(atoms) < 2:
        return atoms

    dist = atoms.get_all_distances(mic=False)
    np.fill_diagonal(dist, np.inf)

    n_components, labels = connected_components(csr_matrix(dist <= max_gap), directed=False)
    if n_components == 1:
        logging.info("[remove_escaped_atoms] single fragment, nothing removed.")
        return atoms

    main_label = int(np.argmax(np.bincount(labels, minlength=n_components)))
    for label in range(n_components):
        if label == main_label:
            continue
        idx = np.flatnonzero(labels == label)
        logging.info(
            f"[remove_escaped_atoms] removing {atoms[idx].get_chemical_formula()} "
            f"at indices {idx.tolist()}."
        )

    cleaned = atoms[labels == main_label]
    logging.info(f"[remove_escaped_atoms] kept {len(cleaned)}/{len(atoms)} atoms.")
    return cleaned