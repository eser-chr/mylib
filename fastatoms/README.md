# fastatoms

An ASE-like `Atoms` class implemented in C++ and exposed to Python with pybind11,
with a fast, parallel reader/writer for extended XYZ (extxyz) files.

The goal is to remove the extxyz bottleneck in `ase.io` (per-line Python parsing and
per-frame object construction) while keeping an API that feels like ASE, plus cheap
conversion to/from `ase.Atoms` when you need the rest of ASE.

## Build and install

Requirements: a C++17 compiler, CMake >= 3.18, Python >= 3.9.

```bash
pip install .                 # regular install
pip install -e . -v           # editable install (rebuilds on import with scikit-build-core)
pip install ".[test]" && pytest
```

Build options (pass through scikit-build-core):

```bash
# No network at build time: fall back to strtod instead of fast_float
pip install . -C cmake.define.FASTATOMS_USE_FAST_FLOAT=OFF

# Disable OpenMP, or tune for the local CPU (not portable)
pip install . -C cmake.define.FASTATOMS_USE_OPENMP=OFF
pip install . -C cmake.define.FASTATOMS_NATIVE=ON
```

On macOS with Apple clang, OpenMP needs `brew install libomp` (and possibly
`-C cmake.define.OpenMP_ROOT=$(brew --prefix libomp)`). If it is not found the build
silently falls back to single-threaded. Check with `fastatoms.has_openmp`.

On HPC clusters just load a recent GCC and CMake module before `pip install`.

## Usage

```python
import numpy as np
import fastatoms as fa

atoms = fa.Atoms("Cu4Ga2", positions=np.random.rand(6, 3) * 5, cell=[10, 10, 10], pbc=True)
atoms.append("O", [1.0, 2.0, 3.0])
atoms.positions += 0.1                     # works like ASE (getter copy + setter)
atoms.info["energy"] = -42.0               # info is a live mapping
atoms.set_array("forces", np.zeros((len(atoms), 3)))

sub = atoms[atoms.numbers == 29]           # int, slice, index list, bool mask
del atoms[0]
atoms.wrap()
d = atoms.get_all_distances(mic=True)

fa.write("traj.extxyz", [atoms] * 1000)
frames = fa.read("traj.extxyz", ":")       # list[Atoms], parsed in parallel
last = fa.read("traj.extxyz")              # index=-1 by default, like ase.io.read
some = fa.read("traj.extxyz", "100:200:5")
n = fa.count_frames("traj.extxyz")

ase_atoms = frames[0].to_ase()             # energy/forces/stress -> SinglePointCalculator
back = fa.from_ase(ase_atoms)
```

Benchmark against ASE:

```bash
python benchmarks/bench_extxyz.py --frames 2000 --atoms 200
```

## API overview

| Area | Methods / attributes |
| --- | --- |
| Construction | `Atoms(symbols, positions, numbers, cell, pbc, info)` where symbols is a formula string, list of symbols/numbers, list of `Atom`, or another `Atoms` |
| Atoms | `len`, `append`, `extend`, `+`, `+=`, `[i]`, `[slice/list/mask]`, `del`, `pop`, `copy`, pickling |
| Positions | `get_positions`, `set_positions`, `positions`, `get_scaled_positions`, `set_scaled_positions` |
| Species | `get_chemical_symbols`, `set_chemical_symbols`, `symbols`, `get_atomic_numbers`, `set_atomic_numbers`, `numbers`, `get_chemical_formula` |
| Cell | `get_cell`, `set_cell(scale_atoms=)`, `cell`, `get_pbc`, `set_pbc`, `pbc`, `get_volume` |
| Geometry | `translate`, `wrap`, `get_distance(mic=)`, `get_all_distances(mic=)` |
| Data | `info` (live dict-like), `get_array`, `set_array`, `has`, `delete_array`, `array_names`, `arrays` (snapshot) |
| I/O | `fa.read`, `fa.write`, `fa.count_frames`, `fa.parse_extxyz`, `fa.format_extxyz`, `Atoms.write` |
| ASE | `fa.to_ase`, `fa.from_ase`, `Atoms.to_ase()`, `Atoms.from_ase()` |

## Design

* Storage is contiguous C++ vectors: `numbers` (int), `positions` (N x 3 double),
  a 3x3 cell (rows are lattice vectors, same as ASE), `pbc`, an `info` map, and typed
  per-atom arrays (`R`, `I`, `L`, `S` columns, matching extxyz `Properties`).
* Per-atom arrays are kept consistent automatically on `append`, `extend`, `del`, slicing.
* Reading happens in three steps with the GIL released:
  1. the whole file is read into memory with one call,
  2. a fast scan (`memchr` over newlines) locates frame boundaries,
  3. selected frames are parsed in parallel with OpenMP, numbers via `fast_float`.
  This means `read(path, 5)` or `read(path, "::10")` only parses the frames you ask for.
* Writing formats frames in parallel in batches, with a custom fixed-point formatter
  for positions/real columns (same `%.8f` precision as ASE) and shortest round-trip
  formatting for lattice and info floats.

## Semantics that differ from ASE

* Getters return copies. `atoms.positions[0] = x` modifies a temporary and has no
  effect. Use `atoms.positions = p`, `atoms.positions += d`, or `set_positions`.
  `atoms.info` is the exception, it is a live mapping.
* `atoms[i]` returns a detached `Atom` copy.
* info values can be bool, int, float, str or a flat list of floats. Multi-number
  values such as `stress="..."` are read as a list of floats.
* Calculator results stay in `info`/arrays when reading. `to_ase()` moves them into a
  `SinglePointCalculator`.
* No masses, constraints, tags, momenta or calculators on the C++ side (add them as
  per-atom arrays if needed).
* `get_distance(mic=True)` uses fractional rounding, which is exact for reasonably
  shaped cells but not for very skewed ones.

## Possible next steps

* Memory-mapped reading for files larger than RAM, and an `iread` generator.
* Neighbour lists (cell lists) in C++.
* Returning zero-copy NumPy views with explicit lifetime handling.
