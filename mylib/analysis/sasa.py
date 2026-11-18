"""
sasa_surface.py
---------------
Surface atom detection for nanoparticles via Solvent Accessible Surface Area (SASA).

Two backends are provided, selected automatically:
  1. FreeSASA  – C library with Python bindings, multithreaded, fast.
                 Install: pip install freesasa
  2. Pure NumPy/SciPy Shrake–Rupley – zero extra dependencies beyond the
                 standard scientific stack, vectorised, ~5-20x slower than
                 FreeSASA but still fine for nanoparticles up to ~10k atoms.

Public API
----------
surface_indices_from_points(positions, radii, probe_radius, sasa_threshold, n_points, backend)
    Core calculation: Nx3 float array → list[int] of surface atom indices.

surface_indices_from_atoms(atoms, radii_dict, probe_radius, sasa_threshold, n_points, backend)
    ASE wrapper: ase.Atoms → list[int] of surface atom indices.

Typical radii for CuGaO nanoparticles (Angstrom, van der Waals):
    Cu  1.40   Ga  1.87   O   1.52
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike

# Default vdW radii (Å) for common elements in our nanoparticle systems.
_DEFAULT_VDW_RADII: dict[str, float] = {
    "Cu": 1.40,
    "Ga": 1.87,
    "O":  1.52,
    "H":  1.20,
    "C":  1.70,
    "N":  1.55,
    "S":  1.80,
    "Zn": 1.39,
    "Al": 1.84,
    "Si": 2.10,
    "Fe": 1.94,
    "Ni": 1.63,
    "Pt": 1.72,
    "Pd": 1.63,
    "Ag": 1.72,
    "Au": 1.66,
    "Co": 1.92,
    "Cr": 1.89,
    "Mn": 1.97,
    "Ti": 1.87,
    "Zr": 2.36,
}

# ---------------------------------------------------------------------------
# Backend 1 – FreeSASA (preferred)
# ---------------------------------------------------------------------------

def _sasa_freesasa(
    positions: np.ndarray,
    radii: np.ndarray,
    probe_radius: float,
    n_points: int,
) -> np.ndarray:
    """
    Per-atom SASA (Å²) using the FreeSASA C library.

    Uses the Shrake & Rupley algorithm with ``n_points`` test points.
    FreeSASA is multithreaded by default and runs at ~0.1 ms/atom.

    Parameters
    ----------
    positions : (N, 3) float array   – atomic positions in Å
    radii     : (N,)   float array   – atomic vdW radii in Å
    probe_radius : float             – probe sphere radius in Å
    n_points  : int                  – number of test points on each sphere

    Returns
    -------
    sasa : (N,) float array          – per-atom SASA in Å²
    """
    import freesasa  # type: ignore[import]

    N = len(positions)
    # FreeSASA expects a flat [x1, y1, z1, x2, y2, z2, ...] C-order array.
    coords_flat = positions.flatten().tolist()
    radii_list  = radii.tolist()

    params = freesasa.Parameters({
        "algorithm":   freesasa.ShrakeRupley,
        "probe-radius": probe_radius,
        "n-points":    n_points,
    })

    result = freesasa.calcCoord(coords_flat, radii_list, params)

    return np.array([result.atomArea(i) for i in range(N)], dtype=float)


# ---------------------------------------------------------------------------
# Backend 2 – Pure NumPy Shrake–Rupley
# ---------------------------------------------------------------------------

def _golden_spiral_points(n: int) -> np.ndarray:
    """
    Return ``n`` approximately uniformly distributed points on the unit sphere
    using the Fibonacci / golden-angle spiral method.

    Much faster to generate than random points and better distributed than
    latitude/longitude grids.

    Returns
    -------
    pts : (n, 3) float array
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i   = np.arange(n, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)          # polar angle
    theta = 2.0 * np.pi * i / golden                       # azimuthal angle
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)                    # (n, 3)


def _sasa_numpy(
    positions: np.ndarray,
    radii: np.ndarray,
    probe_radius: float,
    n_points: int,
) -> np.ndarray:
    """
    Per-atom SASA (Å²) via a vectorised Shrake–Rupley implementation.

    Algorithm
    ---------
    For each atom i:
      1. Place ``n_points`` test points uniformly on a sphere of radius
         (r_i + probe_radius) centred at atom i.
      2. A test point is "buried" if it lies within (r_j + probe_radius) of
         any neighbour j ≠ i.
      3. SASA_i = (fraction of exposed points) × 4π(r_i + probe_radius)²

    Vectorisation strategy
    ----------------------
    * A scipy cKDTree prunes the neighbour search: only atoms within
      2*(r_max + probe_radius) need to be checked per atom.
    * Test point coordinates are broadcast as (n_points, 3) arrays, and
      the burial check is a single batched distance comparison against the
      neighbour list → no Python loop over test points.

    Complexity: O(N · k · n_points) where k = average neighbour count.
    For a 3000-atom nanoparticle with n_points=100 this runs in ~1-2 s on CPU.

    Parameters
    ----------
    positions : (N, 3) float array
    radii     : (N,)   float array
    probe_radius : float
    n_points  : int

    Returns
    -------
    sasa : (N,) float array
    """
    from scipy.spatial import cKDTree  # type: ignore[import]

    N     = len(positions)
    pts   = _golden_spiral_points(n_points)           # (n_points, 3) unit sphere
    r_max = radii.max()
    cutoff = 2.0 * (r_max + probe_radius)

    tree    = cKDTree(positions)
    sasa    = np.zeros(N, dtype=float)

    for i in range(N):
        r_probe_i = radii[i] + probe_radius
        # Candidate neighbours (includes i itself; we'll mask it out below)
        neighbour_idx = np.array(tree.query_ball_point(positions[i], cutoff))
        neighbour_idx = neighbour_idx[neighbour_idx != i]

        # Absolute coordinates of test points on atom i's sphere
        sphere_pts = positions[i] + r_probe_i * pts   # (n_points, 3)

        if len(neighbour_idx) == 0:
            # Isolated atom – fully exposed
            sasa[i] = 4.0 * np.pi * r_probe_i**2
            continue

        # Neighbour positions and their (r_j + probe) radii
        nb_pos = positions[neighbour_idx]              # (k, 3)
        nb_r   = (radii[neighbour_idx] + probe_radius) # (k,)

        # Distance from each test point to each neighbour:
        # sphere_pts: (n_points, 3) → expand to (n_points, 1, 3)
        # nb_pos:     (k, 3)       → expand to (1, k, 3)
        diff    = sphere_pts[:, None, :] - nb_pos[None, :, :]  # (n_pts, k, 3)
        dist_sq = (diff ** 2).sum(axis=-1)                      # (n_pts, k)
        nb_r_sq = (nb_r ** 2)[None, :]                          # (1, k)

        # A point is buried if it is inside ANY neighbour sphere
        buried = (dist_sq < nb_r_sq).any(axis=1)               # (n_pts,)
        n_exposed = (~buried).sum()

        sasa[i] = (n_exposed / n_points) * 4.0 * np.pi * r_probe_i**2

    return sasa


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def surface_indices_from_points(
    positions: ArrayLike,
    radii: Optional[ArrayLike] = None,
    probe_radius: float = 1.4,
    sasa_threshold: float = 0.01,
    n_points: int = 100,
    backend: Literal["auto", "freesasa", "numpy"] = "auto",
) -> list[int]:
    """
    Identify surface atoms in a point cloud via SASA.

    An atom is considered a surface atom if its solvent accessible surface
    area exceeds ``sasa_threshold`` Å².  The default threshold (0.01 Å²)
    is deliberately small so that any non-zero SASA → surface.  Increase it
    (e.g. to 1.0 Å²) to require meaningful exposure before inclusion.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Atomic positions in Å.
    radii : array_like, shape (N,), optional
        Per-atom vdW radii in Å.  If ``None``, a uniform radius of 1.5 Å
        is used for all atoms.
    probe_radius : float, default 1.4
        Radius of the probe sphere in Å.  1.4 Å ≈ water molecule; use
        smaller values (0.5–1.0) for gas-phase surface detection.
    sasa_threshold : float, default 0.01
        Minimum SASA (Å²) for an atom to be classified as a surface atom.
    n_points : int, default 100
        Number of test points per atom sphere (Shrake–Rupley resolution).
        Higher → more accurate, slower.  200–500 for publication quality.
    backend : {"auto", "freesasa", "numpy"}
        Computation backend.  "auto" tries FreeSASA first, falls back to
        the pure NumPy implementation if FreeSASA is not installed.

    Returns
    -------
    surface_idx : list[int]
        Sorted list of 0-based atom indices whose SASA > sasa_threshold.

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.random.randn(200, 3) * 5          # random cluster
    >>> surf = surface_indices_from_points(pos)
    >>> print(len(surf), "surface atoms out of", len(pos))
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got shape {positions.shape}")
    N = len(positions)

    if radii is None:
        radii = np.full(N, 1.5, dtype=float)
    else:
        radii = np.asarray(radii, dtype=float)
        if radii.shape != (N,):
            raise ValueError(f"radii must have shape ({N},); got {radii.shape}")

    # ---- backend selection ----
    use_freesasa = False
    if backend in ("auto", "freesasa"):
        try:
            import freesasa  # noqa: F401
            use_freesasa = True
        except ImportError:
            if backend == "freesasa":
                raise RuntimeError(
                    "FreeSASA not installed. Run:  pip install freesasa"
                )
            warnings.warn(
                "FreeSASA not found; falling back to pure NumPy Shrake–Rupley. "
                "For faster calculations install:  pip install freesasa",
                stacklevel=2,
            )

    # ---- compute per-atom SASA ----
    if use_freesasa:
        sasa = _sasa_freesasa(positions, radii, probe_radius, n_points)
    else:
        sasa = _sasa_numpy(positions, radii, probe_radius, n_points)

    return sorted(int(i) for i in np.where(sasa > sasa_threshold)[0])


# ---------------------------------------------------------------------------
# ASE wrapper
# ---------------------------------------------------------------------------

# Element-symbol → vdW radius table used by the ASE wrapper when no
# custom dict is provided.  Same as _DEFAULT_VDW_RADII but exposed for
# user override.
DEFAULT_VDW_RADII = _DEFAULT_VDW_RADII.copy()


def get_surface_indices(
    atoms,
    radii_dict: Optional[dict[str, float]] = None,
    probe_radius: float = 1.4,
    sasa_threshold: float = 0.01,
    n_points: int = 100,
    backend: Literal["auto", "freesasa", "numpy"] = "auto",
    fallback_radius: float = 1.5,
) -> list[int]:
    """
    Identify surface atoms in an ASE Atoms object via SASA.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure.  Positions are read in Å (ASE default).
    radii_dict : dict[str, float], optional
        Map from element symbol to vdW radius (Å).  Missing elements
        are filled from ``DEFAULT_VDW_RADII``, and then from
        ``fallback_radius`` if still not found.
    probe_radius : float, default 1.4
        Probe sphere radius in Å.
    sasa_threshold : float, default 0.01
        Minimum SASA (Å²) to classify an atom as a surface atom.
    n_points : int, default 100
        Shrake–Rupley test point count.
    backend : {"auto", "freesasa", "numpy"}
        Computation backend.
    fallback_radius : float, default 1.5
        vdW radius assigned to elements not found in ``radii_dict`` or
        ``DEFAULT_VDW_RADII``.

    Returns
    -------
    surface_idx : list[int]
        Sorted list of 0-based atom indices on the surface.

    Examples
    --------
    >>> from ase.io import read
    >>> atoms = read("nanoparticle.xyz")
    >>> surf = surface_indices_from_atoms(atoms, probe_radius=1.4)
    >>> print(surf)
    """
    # Merge user-supplied radii on top of defaults
    merged = DEFAULT_VDW_RADII.copy()
    if radii_dict is not None:
        merged.update(radii_dict)

    positions = atoms.get_positions()   # (N, 3) Å
    symbols   = atoms.get_chemical_symbols()

    unknown = set()
    radii_list = []
    for sym in symbols:
        if sym not in merged:
            unknown.add(sym)
        radii_list.append(merged.get(sym, fallback_radius))
    radii_arr = np.array(radii_list, dtype=float)

    if unknown:
        warnings.warn(
            f"Unknown element(s) {unknown}; assigned fallback radius "
            f"{fallback_radius} Å.  Supply a custom radii_dict to override.",
            stacklevel=2,
        )

    return surface_indices_from_points(
        positions,
        radii=radii_arr,
        probe_radius=probe_radius,
        sasa_threshold=sasa_threshold,
        n_points=n_points,
        backend=backend,
    )



