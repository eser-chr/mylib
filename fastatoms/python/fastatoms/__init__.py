"""fastatoms: an ASE-like Atoms container implemented in C++ with fast extxyz I/O."""

from ._core import (
    Atom,
    Atoms,
    InfoDict,
    __version__,
    format_extxyz,
    has_fast_float,
    has_openmp,
    parse_extxyz,
)
from . import ase_compat as _ase_compat
from .io import count_frames, read, write
from .ase_compat import from_ase, to_ase


def _atoms_to_ase(self, attach_calculator=True):
    """Convert to ase.Atoms (energy/forces/stress go into a SinglePointCalculator)."""
    return _ase_compat.to_ase(self, attach_calculator=attach_calculator)


Atoms.to_ase = _atoms_to_ase
Atoms.from_ase = staticmethod(_ase_compat.from_ase)

__all__ = [
    "Atom",
    "Atoms",
    "InfoDict",
    "__version__",
    "count_frames",
    "format_extxyz",
    "from_ase",
    "has_fast_float",
    "has_openmp",
    "parse_extxyz",
    "read",
    "to_ase",
    "write",
]
