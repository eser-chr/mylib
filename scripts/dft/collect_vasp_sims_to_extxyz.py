import os
from os.path import join, abspath
import ase
from ase.io import read, write
import logging
from typing import List
import argparse
from tqdm import tqdm

def find_sims(folder_path: str):
    RES = []
    for path, subdirs, files in os.walk(folder_path):
        if "vasprun.xml" in files and not subdirs:
            RES.append(path)
    return RES


def collect_data(folder_path: str) -> List[ase.Atoms]:
    sims_paths = find_sims(folder_path)
    atoms_list = []
    for sim_path in tqdm(sims_paths):
        try:
            atom = read(join(sim_path, "vasprun.xml"))
            atom.info["REF_energy"] = atom.get_potential_energy() # type: ignore
            atom.arrays["REF_forces"] = atom.get_forces() # type: ignore
            atoms_list.append(atom)
        except Exception as e:
            logging.warning(f"Error of simulation {sim_path}: {e}")
    return atoms_list


def main_parser():
    parser = argparse.ArgumentParser(
        description="Collect atoms from vasprun.xml in simulation folders."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder of simulations to be extracted!",
    )

    parser.add_argument(
        "output_file_path",
        type=str,
        help="The path to the output file where atoms will be written in extxyz format.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("Hi")

    args = main_parser()
    atoms_list = collect_data(args.folder_path)
    write(args.output_file_path, atoms_list, format="extxyz")
