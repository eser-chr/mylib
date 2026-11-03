import ase
from ase import Atoms
from ase.io import read, write
from pathlib import Path
from mace.calculators import MACECalculator
import numpy as np
from scipy.constants import physical_constants
from typing import Sequence, List

# =====================================================
# BULK
# =====================================================


def get_density(atoms: Atoms) -> float:
    """
    Returns the total (average) density of structure in grams/cm^3
    Params:
    atoms: Atoms
    """
    mass = (
        np.sum(atoms.get_masses()) * physical_constants["atomic mass constant"][0] * 1e3
    )  # In grams
    volume = atoms.get_volume() * 1e-24  # Convert to cm^3
    return mass / volume


def deform_and_rattle(
    atoms,
    num=10,
    max_deform=0.05,
    max_rattle=0.02,
    seed: int | None = None,
    surface: bool = False,
):
    """
    Deform and rattle an ASE Atoms object repeatedly.

    If surface=True, only the XY components of the cell are deformed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        D = np.eye(3)
        if surface:
            D[:2, :2] += (rng.random((2, 2)) - 0.5) * 2 * max_deform
        else:
            D += (rng.random((3, 3)) - 0.5) * 2 * max_deform

        A = atoms.copy()
        A.set_cell(D @ A.get_cell(), scale_atoms=True)
        A.rattle(max_rattle, seed=rng.integers(1, 1_000_000))
        out.append(A)
    return out


def mix_structures(atoms1, atoms2, percentage, seed):
    """
    Mixes two structures as an interpolation, between structures and number of particles.
    """
    N1 = len(atoms1)
    symbol_to = atoms2.get_chemical_symbols()[0]

    n_atoms_to_be_replaced = int(percentage * N1)

    rng = np.random.default_rng(seed)
    indices_to_be_replaced = rng.choice(N1, size=n_atoms_to_be_replaced, replace=False)

    new_cell = (1 - percentage) * np.array(atoms1.get_cell()) + percentage * np.array(
        atoms2.get_cell()
    )

    new_atoms = atoms1.copy()
    new_atoms.set_cell(new_cell, scale_atoms=True)
    for index in indices_to_be_replaced:
        new_atoms[index].symbol = symbol_to

    return new_atoms


# def load_models(paths: Sequence[Path]) -> List[MACECalculator]:
#     return [
#         MACECalculator(model_paths=path, device="cuda", default_dtype="float32") # type: ignore
#         for path in paths
#     ]


def append_uncertainty_to_traj(
    traj: Sequence[Atoms], uncertainties: np.ndarray, suffix: str = ""
) -> None:
    """Attach per-atom uncertainties in place; errors if shapes disagree."""
    arr = np.asarray(uncertainties)
    expected = (len(traj), len(traj[0]))
    if arr.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {arr.shape}")
    for snap, u in zip(traj, arr):
        snap.arrays[f"uncertainty{suffix}"] = u


def load_with_ref_data(path: str) -> List[Atoms]:
    atoms: List[Atoms] = read(path, index=":")  # type: ignore

    for atom in atoms:
        atom.info["REF_energy"] = atom.get_potential_energy()
        atom.arrays["REF_forces"] = atom.get_forces()

    return atoms


def calculator_from_checkpoint(
    checkpoint_path: str | Path,
    template_model_path: str | Path,
    device: str = "cuda",
    default_dtype: str = "float64",
    enable_cueq=True
    out_path=None,
) -> MACECalculator:

    import torch
    from mace.calculators import MACECalculator

    model = torch.load(template_model_path, map_location=device, weights_only=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)

    if out_path is not None:
        torch.save(model, out_path)
        return MACECalculator(
            model_paths=out_path, device=device, default_dtype=default_dtype
        )

    return MACECalculator(
        models=model, device=device, default_dtype=default_dtype, enable_cueq=enable_cueq
    )
