# #!/usr/bin/env python3
# import os
# import shutil
# import argparse
# import numpy as np
# import numpy.typing as npt
# from jinja2 import Template
# import ase
# from ase.io import read, write
# from ase import Atoms
# from pathlib import Path

# DEFAULT_KSPACING = 0.1
# INCAR_MAP = {
#     "bulk": "INCAR_bulk",
#     "surface": "INCAR_surface",
#     "cluster": "INCAR_bulk",
# }

# POTCAR_PREFIX = "{elem}_potcar"


# def classify_from_atoms(atoms: Atoms, vac_threshold: float = 3.0) -> str:
#     """
#     Determines if an ase Atoms structure is a bulk, surface or cluster

#     Parameters
#     ----------
#     atoms : ase.Atoms
#         The structure to be determined
#     vac_threshold : float
#         Minimum “vacuum” thickness [Å] along an axis to count as
#         non-periodic in that direction.

#     Returns
#     -------
#     one of { 'bulk', 'surface', 'cluster' }
#     """
#     lengths = atoms.get_cell_lengths_and_angles()[:3] 
#     pos = atoms.get_positions()  # N×3 array
#     bbox = pos.max(axis=0) - pos.min(axis=0)  # atom extents along x,y,z

#     vacuum = lengths - bbox  # how much empty space in each dir

#     nonperiodic = np.count_nonzero(vacuum > vac_threshold)
#     if nonperiodic == 0:
#         return "bulk"
#     elif nonperiodic == 1:
#         return "surface"
#     else:
#         return "cluster"


# def theo_kpoints(cell: npt.NDArray, kspacing: float = DEFAULT_KSPACING) -> npt.NDArray:
#     """
#     Calculate k-point mesh based on reciprocal lattice norms and spacing.
#     """
#     rec_lat = 2 * np.pi * np.linalg.inv(cell).T
#     norms = np.linalg.norm(rec_lat, axis=1)
#     points = np.ceil(norms / kspacing).astype(int)
#     return np.maximum(points, 1)


# def get_kpoints(
#     atom, template: Template, sim_type: str, kspacing: float = DEFAULT_KSPACING
# ) -> str:
#     """
#     Render KPOINTS text for each simulation type.
#     """
#     if sim_type == "cluster":
#         mesh = [1, 1, 1]
#     else:
#         mesh = theo_kpoints(atom.get_cell(), kspacing)
#         if sim_type == "surface":
#             mesh[2] = 1

#     kpts_str = " ".join(map(str, mesh))
#     return template.render({"kpoints": kpts_str})


# def get_potcar(atom: ase.Atoms, potcar_dir: Path) -> str:
#     """
#     Choose POTCAR file based on number of unique elements.
#     """
#     symbols = atom.get_chemical_symbols()
#     unique = list(set(symbols))
#     unique.sort()
    
#     title = "".join(unique)
#     return POTCAR_PREFIX.format(elem=title)


# def create_db(extxyz_file: Path, output_dir: Path, settings_dir: Path):
#     """
#     Generate VASP input directories from snapshots in an extxyz file.
#     """
#     # Prepare paths
#     kpt_tpl_path = settings_dir / "KPOINTS"
#     potcar_dir = settings_dir

#     # Read template once
#     with open(kpt_tpl_path) as f:
#         template = Template(f.read())

#     snapshots = read(extxyz_file, index=":")
#     os.makedirs(output_dir, exist_ok=True)

#     for idx, atom in enumerate(snapshots):
#         sim_type = classify_from_atoms(atom, vac_threshold=3.0)
#         job_dir = output_dir / str(idx)
#         os.makedirs(job_dir, exist_ok=True)

#         # POSCAR
#         write(job_dir / "POSCAR", atom, format="vasp", sort=True)

#         # KPOINTS
#         ktxt = get_kpoints(atom, template, sim_type)
#         with open(job_dir / "KPOINTS", "w") as f:
#             f.write(ktxt)

#         # POTCAR
#         potcar_src = potcar_dir / get_potcar(atom, potcar_dir)
#         shutil.copy(potcar_src, job_dir / "POTCAR")

#         # INCAR
#         incar_file = INCAR_MAP.get(sim_type, INCAR_MAP["bulk"])
#         shutil.copy(settings_dir / incar_file, job_dir / "INCAR")


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Create a VASP database from extxyz snapshots"
#     )
#     parser.add_argument("extxyz", type=Path, help="Path to extxyz file")
#     parser.add_argument("out_dir", type=Path, help="Output directory for simulations")
#     parser.add_argument(
#         "settings", type=Path, help="Directory with INCAR/POTCAR/KPOINTS templates"
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     create_db(args.extxyz, args.out_dir, args.settings)
#     print("Simulation database created.")
