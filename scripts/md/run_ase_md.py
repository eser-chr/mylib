from pathlib import Path
import ase
from ase import Atoms
from ase import units
from ase.io import read
from mace.calculators import MACECalculator
from ase.md.nose_hoover_chain import NoseHooverChainNVT
import argparse


def my_md(
    init_pos: Atoms,
    time_fs: float,
    timestep: float,
    temp_K: float,
    writing_time: float,
    calculator: MACECalculator,
    output_file: Path,
) -> None:
    steps = int(time_fs / timestep)
    writing_steps = int(writing_time / timestep)
    init_pos.calc = calculator
    dyn = NoseHooverChainNVT(
        init_pos, timestep * units.fs, temperature_K=temp_K, tdamp=50 * units.fs
    )

    def write_frame():
        dyn.atoms.write(output_file, append=True)

    dyn.attach(write_frame, interval=writing_steps)
    dyn.run(steps)
    print("MD finished !!")


def main_parser():
    p = argparse.ArgumentParser(
        "Run an mD with ase. Heavy but nice easy script for small mds"
    )
    p.add_argument(
        "input_path",
        type=Path,
        help="the path to the extxyz file of the initial structure",
    )
    p.add_argument(
        "output_path",
        type=Path,
        help="The path to where the trajectory will be saved",
    )

    p.add_argument(
        "mace_model_path",
        type=Path,
        help="The path of the mace potentials.",
    )
    
    return p.parse_args()


if __name__ == "__main__":
    print("I am starting an MD")
    print("REad initial structure")

    args = main_parser()

    init_pos = read(args.input_path, index=0)
    assert isinstance(init_pos, Atoms), "Expected a single Atoms object"
    calc = MACECalculator(args.mace_model_path, device="cuda")

    my_md(init_pos, 100_000, 0.5, 1000, 100, calc, args.output_path)