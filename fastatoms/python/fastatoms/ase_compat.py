"""Conversion between fastatoms.Atoms and ase.Atoms. ASE is only imported when needed."""

import warnings

import numpy as np

from ._core import Atoms

__all__ = ["to_ase", "from_ase"]

# calculator results that are per atom (stored as per-atom arrays)
_PER_ATOM_RESULTS = {"forces", "energies", "magmoms", "charges", "stresses"}
# calculator results that are per frame (stored in info)
_PER_FRAME_RESULTS = {"energy", "free_energy", "stress", "magmom", "dipole"}


def _to_info_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.astype(float).ravel().tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return value


def _set_info(out, key, value):
    try:
        out.info[key] = _to_info_value(value)
    except (TypeError, ValueError, RuntimeError):
        warnings.warn(f"fastatoms: skipping info[{key!r}] of unsupported type {type(value).__name__}")


def from_ase(atoms, include_calculator=True):
    """Build a fastatoms.Atoms from an ase.Atoms."""
    out = Atoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=np.asarray(atoms.get_cell()),
        pbc=atoms.get_pbc(),
    )
    for key, value in atoms.info.items():
        _set_info(out, key, value)
    for name, values in atoms.arrays.items():
        if name in ("numbers", "positions"):
            continue
        out.set_array(name, values)

    calc = getattr(atoms, "calc", None)
    if include_calculator and calc is not None:
        for key, value in getattr(calc, "results", {}).items():
            if key in _PER_ATOM_RESULTS:
                out.set_array(key, np.asarray(value))
            elif key in _PER_FRAME_RESULTS or np.ndim(value) == 0:
                _set_info(out, key, value)
    return out


def to_ase(atoms, attach_calculator=True):
    """Build an ase.Atoms from a fastatoms.Atoms.

    With attach_calculator=True, energy/free_energy/stress in info and forces in the
    per-atom arrays are moved into a SinglePointCalculator, which is what
    ase.io.read does for extxyz files.
    """
    from ase import Atoms as AseAtoms

    out = AseAtoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    info = {k: (np.array(v) if isinstance(v, list) else v) for k, v in atoms.info.items()}
    arrays = {name: atoms.get_array(name) for name in atoms.array_names()}

    results = {}
    if attach_calculator:
        for key in ("energy", "free_energy"):
            value = info.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                results[key] = float(info.pop(key))
        if "stress" in info:
            stress = np.asarray(info["stress"], dtype=float)
            if stress.size == 6:
                results["stress"] = stress
                info.pop("stress")
            elif stress.size == 9:
                from ase.stress import full_3x3_to_voigt_6_stress

                results["stress"] = full_3x3_to_voigt_6_stress(stress.reshape(3, 3))
                info.pop("stress")
        if "forces" in arrays:
            results["forces"] = np.asarray(arrays.pop("forces"))

    out.info.update(info)
    for name, values in arrays.items():
        out.set_array(name, np.asarray(values))

    if results:
        from ase.calculators.singlepoint import SinglePointCalculator

        out.calc = SinglePointCalculator(out, **results)
    return out
