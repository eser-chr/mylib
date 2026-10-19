from __future__ import annotations
from typing import Sequence, List
from pathlib import Path
from ase import Atoms
import numpy as np
from numpy import typing as npt
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from mace.calculators import MACECalculator


def calc_uncertainty(trajectory:Sequence[Atoms], models:Sequence[MACECalculator])-> List[npt.NDArray]:
    """ 
    Calculates the ensemble uncertainty per atom.
    """
    U = []
    for frame in trajectory:
        F_frame = np.stack([model.get_forces(frame) for model in models], axis=0)  # (n_models, n_atoms, 3) # type:ignore
        F_std = F_frame.std(axis=0)
        U.append(np.mean(F_std, axis = 1))                                              # (n_atoms, 3)

    assert len(U) == len(trajectory)
    assert len(U[0]) == len(trajectory[0])
    return U


from ase.neighborlist import neighbor_list

def calc_local_uncertainty(trajectory: Sequence[Atoms], models: Sequence[MACECalculator],
                           cutoff: float = 5.0) -> List[npt.NDArray]:
    """
    Per-atom force uncertainty, averaged over each atom's neighborhood (including itself) within `cutoff`.
    """
    U = []
    for frame in trajectory:
        F_frame = np.stack([model.get_forces(frame) for model in models], axis=0)  # (n_models, n_atoms, 3) # type:ignore
        u = F_frame.std(axis=0).mean(axis=1)                                       # (n_atoms,)

        i, j = neighbor_list("ij", frame, cutoff)
        n = len(frame)
        u_sum = u + np.bincount(i, weights=u[j], minlength=n)
        count = 1 + np.bincount(i, minlength=n)
        U.append(u_sum / count)                                                    # (n_atoms,)

    assert len(U) == len(trajectory)
    assert len(U[0]) == len(trajectory[0])
    return U