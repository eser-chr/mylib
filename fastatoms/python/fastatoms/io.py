"""ase.io-like read/write for extxyz."""

import os

from . import _core

__all__ = ["read", "write", "count_frames"]


def _normalize_index(index):
    """Accept the same index forms as ase.io.read: int, slice, None, "-1", ":", "10:20:2"."""
    if index is None or isinstance(index, slice):
        return index
    if isinstance(index, str):
        index = index.strip()
        if ":" not in index:
            return int(index)
        parts = index.split(":")
        if len(parts) > 3:
            raise ValueError(f"invalid index string {index!r}")
        return slice(*(int(p) if p.strip() else None for p in parts))
    return int(index)


def _check_format(fmt):
    if fmt not in (None, "extxyz", "xyz"):
        raise ValueError(f"fastatoms only supports extxyz, got format={fmt!r}")


def read(filename, index=-1, format=None):
    """Read frames from an extxyz file.

    index=-1 (default) returns the last frame as Atoms, like ase.io.read.
    index=":" or a slice returns a list of Atoms.
    Frames are parsed in parallel (OpenMP) with the GIL released.
    """
    _check_format(format)
    path = os.fspath(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return _core.read_extxyz(path, _normalize_index(index))


def write(filename, images, append=False, format=None):
    """Write one Atoms or an iterable of Atoms to an extxyz file."""
    _check_format(format)
    _core.write_extxyz(os.fspath(filename), images, append)


def count_frames(filename):
    """Number of frames in an extxyz file (only scans frame boundaries)."""
    path = os.fspath(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return _core.count_extxyz_frames(path)
