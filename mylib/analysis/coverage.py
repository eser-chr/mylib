import numpy as np
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def gao_islands(atoms, cutoff=2.4, min_ga=2):
    """
    Label Ga-O-Ga connected islands.

    Returns an int array of length len(atoms).
    0 = not in any island, 1..K = island index, ordered by size (1 = largest).
    """
    n   = len(atoms)
    sym = np.array(atoms.get_chemical_symbols())

    # only Ga-O pairs within cutoff become edges (both directions returned)
    i, j = neighbor_list("ij", atoms, {("Ga", "O"): cutoff})
    adj  = coo_matrix((np.ones(len(i), dtype=bool), (i, j)), shape=(n, n))

    _, labels = connected_components(adj, directed=False)

    n_ga  = np.bincount(labels, weights=(sym == "Ga"))
    sizes = np.bincount(labels)

    kept = np.flatnonzero(n_ga >= min_ga)
    kept = kept[np.argsort(-sizes[kept], kind="stable")]

    remap = np.zeros(len(sizes), dtype=int)
    remap[kept] = np.arange(1, len(kept) + 1)
    return remap[labels]

RADII = {"Cu": 1.40, "Ga": 1.87, "O": 1.52}   # Bondi vdW

def voxel_atom_volumes(atoms, spacing=0.2):
    """Per-atom volumes (Å^3) from a union-of-spheres grid, overlaps split by power distance."""
    pos = atoms.positions
    r   = np.array([RADII[s] for s in atoms.get_chemical_symbols()])

    lo = pos.min(0) - r.max()
    hi = pos.max(0) + r.max()
    axes  = [np.arange(l, h + spacing, spacing) for l, h in zip(lo, hi)]
    shape = tuple(len(a) for a in axes)

    best  = np.full(shape, np.inf, dtype=np.float32)
    owner = np.full(shape, -1, dtype=np.int32)

    for a, (p, ri) in enumerate(zip(pos, r)):
        sl = tuple(slice(np.searchsorted(ax, c - ri), np.searchsorted(ax, c + ri, side="right"))
                   for ax, c in zip(axes, p))
        gx, gy, gz = np.meshgrid(*[ax[s] for ax, s in zip(axes, sl)], indexing="ij")
        score = (gx - p[0])**2 + (gy - p[1])**2 + (gz - p[2])**2 - ri**2

        b, o = best[sl], owner[sl]                 # views, writes go through
        upd  = (score <= 0) & (score < b)
        b[upd] = score[upd]
        o[upd] = a

    counts = np.bincount(owner[owner >= 0], minlength=len(atoms))
    return counts * spacing**3, owner

def interface_areas(owner, tags, spacing=0.2, iso_correction=True):
    """
    Face-count areas in Å^2, both arrays of length K+1, indexed by tag.
    A_iface[k] = island k vs tag-0 atoms (A_iface[0] is always 0).
    A_surf[k]  = tag k vs vacuum (A_surf[0] = free surface of non-island atoms).
    """
    L = np.where(owner >= 0, tags[np.clip(owner, 0, None)], -1)   # -1 = vacuum
    L = np.pad(L, 1, constant_values=-1)
    K = tags.max()

    iface = np.zeros(K + 1)
    surf  = np.zeros(K + 1)
    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax], s1[ax] = slice(None, -1), slice(1, None)
        a, b = L[tuple(s0)], L[tuple(s1)]
        for x, y in ((a, b), (b, a)):
            m = (x > 0) & (y == 0)
            iface += np.bincount(x[m], minlength=K + 1)
            m = (x >= 0) & (y == -1)
            surf  += np.bincount(x[m], minlength=K + 1)

    face = spacing**2 / (1.5 if iso_correction else 1.0)
    return iface * face, surf * face


from scipy.ndimage import distance_transform_edt


def shadow_coverage(owner, tags, spacing=0.2, delta=1.5):
    """
    Fraction of the tag-0 boundary within delta (Å) of an island.
    delta = 0 recovers A_iface.sum() / (A_iface.sum() + A_surf[0]).
    """
    L = np.where(owner >= 0, tags[np.clip(owner, 0, None)], -1).astype(np.int32)
    L = np.pad(L, 1, constant_values=-1)

    island = L > 0
    if not island.any():
        return 0.0
    dist = distance_transform_edt(~island, sampling=spacing)     # 0 inside islands

    covered = total = 0
    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax], s1[ax] = slice(None, -1), slice(1, None)
        s0, s1 = tuple(s0), tuple(s1)
        a, b   = L[s0], L[s1]
        da, db = dist[s0], dist[s1] #type:ignore
        for x, y, dy in ((a, b, db), (b, a, da)):
            m = (x == 0) & (y != 0)                  # tag-0 boundary face (to island or vacuum)
            total   += np.count_nonzero(m)
            covered += np.count_nonzero(m & (dy <= delta))
    return covered / total
