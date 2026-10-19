# 23-7-2025
# Christoforos Eseroglou

import numpy as np
import matplotlib.pyplot as plt
from wulffpack import SingleCrystal
import ase
from typing import Tuple, Dict, Any


class Convert:
    @staticmethod
    def tuple_to_str(t):
        return "".join((str(i) for i in t))

    @staticmethod
    def str_to_tuple(str_obj: str) -> Tuple[int, ...]:
        try:
            return tuple(int(char) for char in str_obj)
        except ValueError as e:
            raise ValueError(f"Invalid character in input string '{str_obj}'") from e


    @staticmethod
    def strKeys_to_intTuple(dict_obj: Dict[str, Any]) -> Dict[Tuple[int, ...], Any]:
        return {Convert.str_to_tuple(key): item for key, item in dict_obj.items()}

    @staticmethod
    def intTupleKeys_to_str(dict_obj: Dict[Tuple[int, ...], Any]) -> Dict[str, Any]:
        return {Convert.tuple_to_str(key): item for key, item in dict_obj.items()}


class CellCalculator:
    @staticmethod
    def add_extra_vacuum(pos: np.ndarray, perc=0.5) -> np.ndarray:
        """
        Calculates a new cell for the nanoparticle. It adds vacuum as a percentage of the size of nanoparticle.

        Args:
        pos: The positions of the atoms.
        perc: The percentage. Default 0.5 = 50 %

        Returns:
        --------

        """
        M = np.max(pos, axis=0)
        m = np.min(pos, axis=0)
        Dx = M - m
        return (1 + perc) * Dx


def create_nanoparticle(
    prim_st: ase.Atoms, surface_energies: dict[tuple, float], N: int
) -> ase.Atoms:
    particle = SingleCrystal(
        surface_energies=surface_energies, primitive_structure=prim_st, natoms=N
    )

    atoms = particle.atoms
    box_size = CellCalculator.add_extra_vacuum(atoms.positions)
    atoms.set_cell(np.diag(box_size))
    atoms.center()

    return atoms


def plot_wulff_geometry(surface_energies: dict[tuple, float]) -> None:
    wulff = SingleCrystal(surface_energies)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    wulff.make_plot(ax, linewidth=0.5, alpha=0.8)
    # ax.set_box_aspect([1,1,1])

    plt.tight_layout()
    plt.show()
