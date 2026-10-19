from ase.io import read, write
from ase import Atoms
from mylib.ml.evaluation import get_traj_uncertainty_F_per_atom
from mylib.ml.utils import load_models, append_uncertainty_to_traj
from pathlib import Path
import argparse
from typing import Dict, List


def parser():
    parser = argparse.ArgumentParser(
        description="A script that calculates the uncertainty of a list of atomic \
            configurations based on a MACE ensemble. \n\
            Args: [input file][output file][tag][mace potentials folder]  "
        )

    parser.add_argument(
        "input_file",
        type=Path,
        help="The input extxyz file that contains the snapshots to be evaluated",
    )
    
    parser.add_argument(
        "output_file",
        type= Path,
        help = "The output extxyz file that will contain the snapshots and also the uncertainty."        
    )
    
    parser.add_argument(
        "tag",
        type=str,
        help = "It will be used as identifier in the extxyz"
    )
    
    parser.add_argument(
        "mace_folder",
        type=Path,
        help = "The path to the mace potentials. The script will collect any file with *.model"
    )
    
    return parser.parse_args()
    
    



if __name__=="__main__":
    args = parser()
    print("1/5) Start conversion ✔️")
    models = load_models(args.mace_folder.rglob("*.model"))
    models = [p for p in models if "compiled" not in p.name]
    print(f"I loaded {len(models)} models!")
    print("2/5) Load models ✔️")
    traj = read(args.input_file, index=":")
    print("3/5) Read file ✔️")
    uncertainty = get_traj_uncertainty_F_per_atom(traj, models)
    print("4/5) Calc uncertainty ✔️")
    append_uncertainty_to_traj(traj, uncertainty, suffix=args.tag)
    print("5/5) Write output ✔️")
    write(args.output_file, traj)
    print("Done ✔️")
    print("========")