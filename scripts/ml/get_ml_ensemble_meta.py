"""
I call this script to calculate some metadata after generating a ensemble of models.

Metadata:
-> Elements in this model
-> Dimer energies
-> Error of all models over the validation set
-> Uncertainty over the validation set
"""
from dataclasses import dataclass, field
from pathlib import Path
import itertools
import json
import ase
from ase import Atoms
from ase.io import read, write
import pickle

import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Set, List, Dict, Any
from mace.calculators import MACECalculator
import logging
logging.basicConfig(level=logging.INFO)
import argparse
from time import time



# ===============
# UNIQUE ELEMENTS
# ===============
def get_all_elements(path_of_all_structures: Path) -> List[str]:
    atoms: List[Atoms] = read(path_of_all_structures, index=":")  # type: ignore
    unique_elements_in_db = set()
    for atom in atoms:
        unique_elements_in_db.update(atom.get_chemical_symbols())

    to_rtn = list(unique_elements_in_db)
    return sorted(to_rtn)


# ===============
# DIMER ENERGY
# ===============
def generate_dimer_traj(
    el1: str, el2: str, start: float, end: float, nsteps: int
) -> List[Atoms]:
    """
    Generate an ASE trajectory of a dimer going from start -> end distance.
    """
    traj = []
    distances = np.linspace(start, end, nsteps)
    for d in distances:
        dimer = Atoms(f"{el1}{el2}", positions=[[0, 0, 0], [0, 0, d]])
        traj.append(dimer)
    return traj


def evaluate_trajectory(traj: List[Atoms], model: MACECalculator) -> npt.NDArray:
    """
    Compute energies for each structure in the trajectory using MACE.
    Returns a numpy array of energies in eV.
    """
    energies = []
    for atoms in traj:
        energy = model.get_potential_energy(atoms)
        energies.append(energy)
    return np.array(energies)


def compute_all_pairs(
    elements: List[str], model: MACECalculator, start: float, end: float, nsteps: int
) -> Dict[str, npt.NDArray]:
    """
    Compute dimer energy curves for all unique pairs of given elements.
    Returns dict { "El1-El2": (distances, energies) }
    """
    distances = np.linspace(start, end, nsteps)
    results = {"distances": distances}
    for el1, el2 in itertools.combinations_with_replacement(elements, 2):
        key = f"{el1}-{el2}"
        traj = generate_dimer_traj(el1, el2, start, end, nsteps)
        energies = evaluate_trajectory(traj, model)
        results[key] = energies

    return results


# =================
# VALIDATION ERROR
# =================


def files_to_use_as_validation(
    path_to_container: Path, path_to_description: Path
) -> List[Path]:
    with open(path_to_description, "r") as f:
        settings = json.load(f)
    if "validation" not in settings:
        raise KeyError(f"Key 'validation' missing in {path_to_description}")
    return [path_to_container / Path(p) for p in settings["validation"]]


def predict(atoms: Atoms, mace_model):
    new = atoms.copy()
    new.calc = mace_model
    return new.get_potential_energy(), new.get_forces()


def evaluate_models(atom: Atoms, models):
    n_models = len(models)
    n_atoms = len(atom)
    E = np.empty(n_models)
    F = np.empty((n_models, n_atoms, 3))
    for i, model in enumerate(models):
        E[i], F[i] = predict(atom, model)
    return E, F


def compute_uncertainties(E: np.ndarray, F: np.ndarray):
    uE = E.std() / F.shape[1]  # divide by n_atoms
    forces_norm = np.linalg.norm(F, axis=2)  # (n_models, n_atoms)
    uF = forces_norm.std(axis=0)  # (n_atoms,)
    return dict(
        uncertainty_E=uE,
        mean_uncertainty_F=uF.mean(),
        max_uncertainty_F=uF.max(),
        per_atom_uncertainty_F=uF,
    )


def set_validation(
    path: Path, models: List[MACECalculator], extxyz_uncertainty_tag: str
):
    atoms_list = read(path, index=":")
    if isinstance(atoms_list, Atoms):
        raise RuntimeError

    results = []
    for atom in atoms_list:
        E, F = evaluate_models(atom, models)
        u = compute_uncertainties(E, F)
        atom.arrays[extxyz_uncertainty_tag] = u.pop("per_atom_uncertainty_F")
        results.append(u)

    return pd.DataFrame(results), atoms_list


# =================
# PARSER
# =================



@dataclass
class Arguments:
    main_folder: Path
    db_folder: Path
    uncertainty_tag: str

    path_of_all_structures: Path = field(init=False)
    paths_of_validation_sets: List[Path] = field(init=False)
    paths_of_valid_models: List[Path] = field(init=False)

    output_csv_uncertainty: Path = field(init=False)
    output_extxyz_uncertainty: Path = field(init=False)

    start: float = 0.5
    end: float = 4.0
    nsteps: int = 50

    def __init__(self, parser: argparse.ArgumentParser):
        args = parser.parse_args()
        self.main_folder = Path(args.main_folder)
        self.db_folder = Path(args.db_folder)
        self.uncertainty_tag = args.uncertainty_tag
        self.__post_init__()  # fill in derived fields

    def __post_init__(self):
        # --- validation ---
        if not self.main_folder.exists():
            raise FileNotFoundError(f"Main folder not found: {self.main_folder}")
        if not self.db_folder.exists():
            raise FileNotFoundError(f"DB folder not found: {self.db_folder}")

        desc_file = self.main_folder / "description.json"
        if not desc_file.exists():
            raise FileNotFoundError(f"Missing description.json in {self.main_folder}")

        # --- derived paths ---
        self.path_of_all_structures = self.main_folder / "all.extxyz"

        self.paths_of_valid_models = list(self.main_folder.rglob("model.model"))
        if not self.paths_of_valid_models:
            raise FileNotFoundError("No valid models found under main folder")

        with open(desc_file, "r") as f:
            settings = json.load(f)
        if "validation" not in settings:
            raise KeyError(f"'validation' key missing in {desc_file}")
        self.paths_of_validation_sets = [
            self.db_folder / "validation" / Path(p) for p in settings["validation"]
        ]

        self.output_csv_uncertainty = self.main_folder / "validation_data.csv"
        self.output_extxyz_uncertainty = self.main_folder


def main_parser() -> Arguments:
    parser = argparse.ArgumentParser(
        description="Creates metadata of a mlip generation."
    )
    parser.add_argument("main_folder", type=Path, help="The path to the main folder.")
    parser.add_argument("db_folder", type=Path, help="The path to the database.")
    parser.add_argument("uncertainty_tag", type=str,
                        help="The tag stored in the extxyz file for the uncertainty.")
    return Arguments(parser)



def main(args: Arguments) -> None:
    elements = get_all_elements(args.path_of_all_structures)

    models = [
        MACECalculator(model_paths=str(model_path), device="cuda")
        for model_path in args.paths_of_valid_models
    ]

    res_per_model = [
        compute_all_pairs(elements, model, args.start, args.end, args.nsteps)
        for model in models
    ]

    pairs = [k for k in res_per_model[0].keys() if k != "distances"]
    distances = res_per_model[0]["distances"]

    mean_profiles: Dict[str, npt.NDArray] = {"distances": distances}
    std_profiles: Dict[str, npt.NDArray] = {"distances": distances}

    for pair in pairs:
        arr = np.stack([res[pair] for res in res_per_model], axis=0)
        mean_profiles[pair] = arr.mean(axis=0)
        std_profiles[pair] = arr.std(axis=0)

    out_path = args.main_folder / "dimer_profiles.pickle"
    with open(out_path, "wb") as f:
        pickle.dump({"mean": mean_profiles, "std": std_profiles}, f)

    fig_path = args.main_folder / "dimer_profiles.png"
    fig, ax = plt.subplots()

    for key, item in mean_profiles.items():
        if key == "distances":
            continue
        std = std_profiles[key]
        ax.plot(distances, item, label=key)
        ax.fill_between(distances, item - std, item + std, alpha=0.2)

    ax.set(
        title=f"Dimer energy of pairs in ensemble {args.main_folder.absolute().name}",
        xlabel=r"distance [$\AA$]",
        ylabel=r"energy [$eV$]",
    )
    ax.grid()
    ax.legend(fontsize="small", ncol=2)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)

    total_df = pd.DataFrame(
        columns=["uncertainty_E", "mean_uncertainty_F", "max_uncertainty_F"]
    )

    for validation_set in args.paths_of_validation_sets:
        logging.info("Working on set %s", validation_set)

        df, atoms_extended = set_validation(
            validation_set, models, args.uncertainty_tag
        )
        df["set"] = validation_set.name
        total_df = pd.concat([total_df, df])

        write(
            args.output_extxyz_uncertainty / validation_set.name,
            atoms_extended,
            format="extxyz",
        )

    total_df.to_csv(args.output_csv_uncertainty, index=False)



if __name__ == "__main__":
    logging.info("Start generation of meta for mace ensemble")
    args = main_parser()
    start =time()
    main(args)
    end =time()
    logging.info(f"Done with generation of meta in {(end-start)/60} minutes.")