"""trajectory_utils.py — Trajectory I/O utilities for the Helios MD pipeline.

Handles reading torchsim TrajectoryReporter HDF5 files and converting them to
ASE Atoms lists or extxyz files.

HDF5 layout written by torchsim (as of 0.5.x)
----------------------------------------------
data/
    positions       float32  (n_frames, n_atoms, 3)   Å
    cell            float32  (n_frames, 3, 3)          Å
    atomic_numbers  int32    (1, n_atoms)              written once
    velocities      float32  (n_frames, n_atoms, 3)   Å/ps   [if saved]
    forces          float32  (n_frames, n_atoms, 3)   eV/Å   [if saved]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from ase import Atoms
from ase.io import write as ase_write

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level HDF5 reader
# ---------------------------------------------------------------------------

def h5_to_arrays(h5_path: Union[str, Path]) -> Dict[str, Optional[np.ndarray]]:
    """Read raw arrays from a torchsim TrajectoryReporter HDF5 file.

    All datasets live under the ``data/`` subgroup. ``atomic_numbers`` is
    stored once with shape ``(1, n_atoms)`` and is squeezed to ``(n_atoms,)``
    before returning.

    Parameters
    ----------
    h5_path:
        Path to the ``.h5`` file.

    Returns
    -------
    dict with keys:
        ``positions``  – ``(n_frames, n_atoms, 3)`` float32
        ``cell``       – ``(n_frames, 3, 3)`` float32
        ``numbers``    – ``(n_atoms,)`` int32
        ``velocities`` – ``(n_frames, n_atoms, 3)`` float32, or ``None``
        ``forces``     – ``(n_frames, n_atoms, 3)`` float32, or ``None``
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        data = f["data"]
        arrays: Dict[str, Optional[np.ndarray]] = {
            "positions":  data["positions"][:],# type: ignore
            "cell":       data["cell"][:],# type: ignore
            "numbers":    data["atomic_numbers"][0],   # (1, n_atoms) -> (n_atoms,)# type: ignore
            "velocities": data["velocities"][:] if "velocities" in data else None,# type: ignore
            "forces":     data["forces"][:]     if "forces"     in data else None, # type: ignore
            "energy":     data["energy"][:] if "energy" in data else None # type:ignore
        } # type: ignore
    n_frames, n_atoms = arrays["positions"].shape[:2]# type: ignore
    logger.debug(
        "%s: %d frames, %d atoms%s%s",
        h5_path.name, n_frames, n_atoms,
        ", velocities" if arrays["velocities"] is not None else "",
        ", forces"     if arrays["forces"]     is not None else "",
    )
    return arrays


# ---------------------------------------------------------------------------
# Core conversion: HDF5 -> List[Atoms]
# ---------------------------------------------------------------------------

def h5_to_atoms(
    h5_path: Union[str, Path],
    temperature_K: Optional[int] = None,
    cell: Optional[np.ndarray] = None,
) -> List[Atoms]:
    """Load a torchsim HDF5 trajectory as a list of ASE ``Atoms`` objects.

    Parameters
    ----------
    h5_path:
        Path to the ``.h5`` file.
    temperature_K:
        If provided, stored in ``atoms.info["temperature_K"]`` for every
        frame. Useful when merging multi-stage cooldown trajectories.
    cell:
        Fallback cell used when the HDF5 file stores no cell data. Either a
        ``(3, 3)`` matrix applied to every frame, or a ``(n_frames, 3, 3)``
        array of per-frame cells. If the file *does* contain cells, this is
        ignored. When neither is available the frames are non-periodic.

    Returns
    -------
    List of ``Atoms``, one per saved frame, in chronological order.
    """
    h5_path = Path(h5_path)
    arrays = h5_to_arrays(h5_path)

    positions  = arrays["positions"]
    cells      = arrays["cell"]
    numbers    = arrays["numbers"]
    velocities = arrays["velocities"]
    forces     = arrays["forces"]
    energies = arrays["energy"]

    n_frames = positions.shape[0]

    if cells is None and cell is not None:
        cell = np.asarray(cell, dtype=float)
        if cell.shape == (3, 3):
            cells = np.broadcast_to(cell, (n_frames, 3, 3))
        elif cell.shape == (n_frames, 3, 3):
            cells = cell
        else:
            raise ValueError(
                f"cell must have shape (3, 3) or ({n_frames}, 3, 3), got {cell.shape}"
            )
    elif cells is None:
        logger.warning("%s: no cell data, frames will be non-periodic", h5_path.name)

    pbc = cells is not None

    frames: List[Atoms] = []
    for i in range(n_frames):
        atoms = Atoms(
            numbers=numbers,
            positions=positions[i],
            cell=cells[i] if pbc else None,
            pbc=pbc,
        )
        if energies is not None:
            atoms.info["energy"] = energies[i]
        if temperature_K is not None:
            atoms.info["temperature_K"] = temperature_K
        if velocities is not None:
            atoms.set_velocities(velocities[i])
        if forces is not None:
            atoms.arrays["forces"] = forces[i]
        frames.append(atoms)

    logger.info("%s -> %d Atoms frames", h5_path.name, n_frames)
    return frames

# ---------------------------------------------------------------------------
# Single-file HDF5 -> extxyz
# ---------------------------------------------------------------------------

def h5_to_extxyz(
    h5_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    temperature_K: Optional[int] = None,
    verbose:bool = False
) -> Path:
    """Convert a single torchsim HDF5 file to an extxyz trajectory.

    Parameters
    ----------
    h5_path:
        Path to the ``.h5`` file.
    out_path:
        Destination extxyz file. Defaults to ``h5_path`` with ``.extxyz``
        extension.
    temperature_K:
        Passed through to :func:`h5_to_atoms`.

    Returns
    -------
    Path to the written extxyz file.
    """
    h5_path  = Path(h5_path)
    out_path = Path(out_path) if out_path is not None else h5_path.with_suffix(".extxyz")

    frames = h5_to_atoms(h5_path, temperature_K=temperature_K)
    ase_write(str(out_path), frames, format="extxyz")
    if verbose:
        logger.info("%s -> %s (%d frames)", h5_path.name, out_path.name, len(frames))
        
    return out_path


# ---------------------------------------------------------------------------
# Cooldown merge: all per-temperature HDF5 files -> one extxyz
# ---------------------------------------------------------------------------

def cooldown_h5_to_atoms(
    sim_dir: Union[str, Path],
    cell: Optional[np.ndarray] = None,
) -> List[Atoms]:
    """Load the full cooldown trajectory from a simulation directory.

    Discovers all ``*_T{temp}.h5`` files, sorts them in descending temperature
    order (matching the cooldown sequence), converts each to ``Atoms``, and
    concatenates them into a single list. Each frame carries
    ``atoms.info["temperature_K"]``.

    Parameters
    ----------
    sim_dir:
        Directory produced by the cooldown MD script, containing one
        ``*_T{temp}.h5`` file per temperature stage.
    cell:
        Fallback ``(3, 3)`` cell applied to every stage whose HDF5 file stores
        no cell data. Ignored for stages that do contain cells.

    Returns
    -------
    Flat list of ``Atoms`` frames across all stages, in cooldown order.
    """
    sim_dir  = Path(sim_dir)
    h5_files = _discover_cooldown_h5(sim_dir)

    all_frames: List[Atoms] = []
    for h5_path in h5_files:
        temperature = _parse_temperature(h5_path)
        stage_frames = h5_to_atoms(h5_path, temperature_K=temperature, cell=cell)
        logger.info("  %s: %d frames at %d K", h5_path.name, len(stage_frames), temperature)
        all_frames.extend(stage_frames)

    logger.info(
        "cooldown_h5_to_atoms: %d total frames from %d stages",
        len(all_frames), len(h5_files),
    )
    return all_frames


def cooldown_to_extxyz(
    sim_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Merge all per-temperature HDF5 files from a cooldown run into one extxyz.

    Files are discovered via ``*_T{temp}.h5`` and sorted in descending
    temperature order (1000 K first, 500 K last).

    Parameters
    ----------
    sim_dir:
        Directory containing the per-temperature ``.h5`` files.
    out_path:
        Output extxyz path. Defaults to
        ``sim_dir / {tag}_full_trajectory.extxyz``.

    Returns
    -------
    Path to the written extxyz file.
    """
    sim_dir  = Path(sim_dir)
    h5_files = _discover_cooldown_h5(sim_dir)

    tag      = h5_files[0].stem.rsplit("_T", 1)[0]
    out_path = (
        Path(out_path) if out_path is not None
        else sim_dir / f"{tag}_full_trajectory.extxyz"
    )

    logger.info(
        "Merging %d stages into %s: %s",
        len(h5_files), out_path.name, [f.name for f in h5_files],
    )

    all_frames = cooldown_h5_to_atoms(sim_dir)
    ase_write(str(out_path), all_frames, format="extxyz")
    logger.info("Written %d frames -> %s", len(all_frames), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# inspection helpers
# ---------------------------------------------------------------------------



def trajectory_info(frames: List[Atoms]) -> Dict[str, object]:
    """Return a summary dict describing a trajectory.

    Parameters
    ----------
    frames:
        Trajectory as a list of ``Atoms``.

    Returns
    -------
    Dict with keys ``n_frames``, ``n_atoms``, ``has_velocities``,
    ``has_forces``, ``temperatures_K`` (sorted unique values, if tagged).
    """
    if not frames:
        return {"n_frames": 0}

    temps = sorted({f.info["temperature_K"] for f in frames if "temperature_K" in f.info})
    return {
        "n_frames":      len(frames),
        "n_atoms":       len(frames[0]),
        "has_velocities": frames[0].get_velocities() is not None,
        "has_forces":    "forces" in frames[0].arrays,
        "temperatures_K": temps or None,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _discover_cooldown_h5(sim_dir: Path) -> List[Path]:
    """Return *_T{temp}.h5 files sorted by descending temperature."""
    h5_files: List[Path] = sorted(
        sim_dir.glob("*T*.h5"),
        key=_parse_temperature,
        reverse=True,
    )
    if not h5_files:
        raise FileNotFoundError(f"No *_T*.h5 files found in {sim_dir}")
    return h5_files

import re

def _parse_temperature(h5_path: Path) -> int:
    """Extract the integer temperature from a filename like ``tag_T1000.h5`` or ``T1000.h5``."""
    m = re.search(r"T(\d+)$", h5_path.stem)
    if m is None:
        raise ValueError(f"No temperature found in {h5_path.name}")
    return int(m.group(1))