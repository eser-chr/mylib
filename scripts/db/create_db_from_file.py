#!/usr/bin/env python3
import os
import shutil
import argparse
import numpy as np
from jinja2 import Template
import ase
from ase.io import read, write
from pathlib import Path

DEFAULT_KSPACING = 0.1
DEFAULT_VAC_THRESHOLD = 4.0
INCAR_MAP = {
    "bulk": "INCAR_bulk",
    "surface": "INCAR_surface",
    "cluster": "INCAR_bulk",
}

POTCAR_PREFIX = "{elem}_potcar"


def classify_from_atoms(
    atoms: ase.Atoms, vac_threshold: float = DEFAULT_VAC_THRESHOLD
) -> tuple[str, list[int]]:
    """
    Determines if an ase Atoms structure is a bulk, surface or cluster.

    Works in fractional coordinates, so it is valid for non-orthogonal cells
    and for vacuum regions that straddle the periodic boundary.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to be classified
    vac_threshold : float
        Minimum gap [Å] between atomic planes, measured perpendicular to the
        lattice planes, for a lattice direction to count as non-periodic.

    Returns
    -------
    (kind, vac_axes)
        kind is one of {'bulk', 'surface', 'cluster'};
        vac_axes lists the lattice directions (0=a, 1=b, 2=c) with vacuum.
    """
    cell = atoms.cell
    frac = atoms.get_scaled_positions(wrap=True)

    # perpendicular height of the cell along each lattice direction
    vol = abs(cell.volume)
    heights = np.array(
        [
            vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
            for i in range(3)
        ]
    )

    vac_axes = []
    for i in range(3):
        s = np.sort(frac[:, i])
        gaps = np.diff(np.append(s, s[0] + 1.0))  # includes wrap-around gap
        if gaps.max() * heights[i] > vac_threshold:
            vac_axes.append(i)

    n = len(vac_axes)
    if n == 0:
        kind = "bulk"
    elif n == 1:
        kind = "surface"
    else:
        kind = "cluster"
    return kind, vac_axes


def theo_kpoints(cell: np.ndarray, kspacing: float = DEFAULT_KSPACING) -> np.ndarray:
    """
    Calculate k-point mesh based on reciprocal lattice norms and spacing.
    """
    rec_lat = 2 * np.pi * np.linalg.inv(cell).T
    norms = np.linalg.norm(rec_lat, axis=1)
    points = np.ceil(norms / kspacing).astype(int)
    return np.maximum(points, 1)


def get_kpoints(
    atom: ase.Atoms,
    template: Template,
    sim_type: str,
    vac_axes: list[int],
    kspacing: float = DEFAULT_KSPACING,
) -> str:
    """
    Render KPOINTS text for each simulation type.
    Directions with vacuum get a single k-point.
    """
    if sim_type == "cluster":
        mesh = [1, 1, 1]
    else:
        mesh = theo_kpoints(atom.get_cell(), kspacing)
        for ax in vac_axes:
            mesh[ax] = 1

    kpts_str = " ".join(map(str, mesh))
    return template.render({"kpoints": kpts_str})


def get_potcar(settings_dir: Path, atom: ase.Atoms) -> str:
    """
    Concatenate POTCARs in alphabetical element order
    (matches POSCAR written with sort=True).
    """
    unique = sorted(set(atom.get_chemical_symbols()))
    files = [POTCAR_PREFIX.format(elem=element) for element in unique]

    total = ""
    for file_path in files:
        with open(settings_dir / file_path, "r") as fin:
            total += fin.read()
    return total


def create_db(
    extxyz_file: Path,
    output_dir: Path,
    settings_dir: Path,
    vac_threshold: float = DEFAULT_VAC_THRESHOLD,
):
    """
    Generate VASP input directories from snapshots in an extxyz file.
    """
    kpt_tpl_path = settings_dir / "KPOINTS"

    # Read template once
    with open(kpt_tpl_path) as f:
        template = Template(f.read())

    snapshots = read(extxyz_file, index=":")
    os.makedirs(output_dir, exist_ok=True)

    counts = {"bulk": 0, "surface": 0, "cluster": 0}

    for idx, atom in enumerate(snapshots):
        sim_type, vac_axes = classify_from_atoms(atom, vac_threshold=vac_threshold)
        counts[sim_type] += 1
        print(
            f"{idx:5d}  {atom.get_chemical_formula():<20s} "
            f"{sim_type:<8s} vac_axes={vac_axes}  "
            f"cellpar={np.round(atom.cell.cellpar(), 1).tolist()}"
        )

        job_dir = output_dir / str(idx)
        os.makedirs(job_dir, exist_ok=True)

        # POSCAR
        write(job_dir / "POSCAR", atom, format="vasp", sort=True)

        # KPOINTS
        ktxt = get_kpoints(atom, template, sim_type, vac_axes)
        with open(job_dir / "KPOINTS", "w") as f:
            f.write(ktxt)

        # POTCAR
        potcar_txt = get_potcar(settings_dir, atom)
        with open(job_dir / "POTCAR", "w") as f:
            f.write(potcar_txt)

        # INCAR
        incar_file = INCAR_MAP.get(sim_type, INCAR_MAP["bulk"])
        shutil.copy(settings_dir / incar_file, job_dir / "INCAR")

    print(f"Summary: {counts}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a VASP database from extxyz snapshots"
    )
    parser.add_argument("extxyz", type=Path, help="Path to extxyz file")
    parser.add_argument("out_dir", type=Path, help="Output directory for simulations")
    parser.add_argument(
        "settings", type=Path, help="Directory with INCAR/POTCAR/KPOINTS templates"
    )
    parser.add_argument(
        "--vac_threshold",
        type=float,
        default=DEFAULT_VAC_THRESHOLD,
        help="Minimum perpendicular gap [Å] to count a direction as vacuum",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_db(args.extxyz, args.out_dir, args.settings, args.vac_threshold)
    print("Simulation database created.")
