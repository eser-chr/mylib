"""
Here are some structures that seem to be homogeneous enough.

In order to slect them I simply used brute force over all combinations and maximized the minimum pair distance over Gallia.
Then because of the degneracy (every concentration might have more than one structures) I added a 2 extra metrics:

maximize the average pair distance
and maximize the mean(std (of coordinates) in every direction) i.e (take the std of the coords σ_χ σ_υ, σ_ζ) calc the mean and try to maximize that.

For 5 and 6 I still got degeneracy which I solved by inspecting visually.

For 6 I chose the structure 24. For 5 the 2 metrics where not consistent, with each other, causing more problems.
For 5 I chose 58. But it seems that there was a symmetrical similarity. The same was for 3 but for 3 it was much more simple to distinguish.




Ideally I would like to check a Voronoi diagramm and minimize the std of the volume, but for now it is a bit too much.
""" 


from ase.io import write
from ase.build import bulk
import numpy as np
from itertools import combinations
from pathlib import Path


def min_pair_distance(atoms, symbol='Ga'):
    indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
    dummy_factor = np.max(atoms.get_cell())  # masks the diagonal of the pair-distances, while using the min.
    if len(indices) < 2:
        return None

    dmat = atoms.get_all_distances(mic=True)
    sub_dmat = dmat[np.ix_(indices, indices)]
    min_dist = np.min(sub_dmat + np.eye(len(indices)) * dummy_factor)

    return min_dist


def avg_pair_distance(atoms, symbol='Ga'):
    indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
    N = len(indices)
    if N < 2:
        return None

    dmat = atoms.get_all_distances(mic=True)
    sub_dmat = dmat[np.ix_(indices, indices)]
    
    return np.mean(sub_dmat) * (1 - N / N ** 2) / 2


def generate_boolean_combinations(n: int, k: int) -> list[list[bool]]:
    """
    Generates all boolean lists of length `n` with exactly `k` True values.

    Parameters:
        n (int): Length of each boolean list.
        k (int): Number of True values per list.

    Returns:
        List of boolean lists, each with exactly `k` True values.
    """
    if not (0 <= k <= n):
        raise ValueError("k must be between 0 and n")

    combos = []
    for indices in combinations(range(n), k):
        mask = [False] * n
        for idx in indices:
            mask[idx] = True
        combos.append(mask)
    return combos


def my_replace(atoms, bit_mask_to_replace, symbol_from, symbol_to):
    
    new_chemical_symbols = [symbol_to if bit else symbol_from for bit in bit_mask_to_replace]
    new = atoms.copy()
    new.set_chemical_symbols(new_chemical_symbols)
    return new


def create_all_substitutions(atoms, n_sub, symbol_from, symbol_to):
    if n_sub <= 1 or not isinstance(n_sub, int):
        raise ValueError("n_sub is belongs to intgeres larger than 1.")
    
    RES = []
    N = len(atoms)
    BIT_MASKS = generate_boolean_combinations(N - 1, n_sub - 1)
    
    for sub_bit_mask in BIT_MASKS:
        bit_mask = [True, *sub_bit_mask]
        RES.append(my_replace(atoms, bit_mask, symbol_from, symbol_to))
    return RES


def find_homogenised(atoms, k):
    
    combos = create_all_substitutions(atoms, k, "Cu", "Ga")
    metric = np.array([min_pair_distance(atoms) for atoms in combos])
    M = np.max(metric)
    S = np.argwhere(metric == M)
    return [combos[i] for i in list(S[:, 0])]


if __name__ == "__main__":
    atoms = bulk('Cu', crystalstructure='fcc', a=3.63, cubic=True) * (2, 2, 2)
    base = Path("./HOMOGENISED")

    for i in range(2, 5):
        file = base / f"{i}.extxyz"
        print(i)
        combos = find_homogenised(atoms, i)
        print(combos)
        write(file, combos, format = "extxyz")