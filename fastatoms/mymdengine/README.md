# mymdengine

Lennard-Jones MD (Velocity Verlet NVE + BAOAB Langevin) with a serial CPU engine and a
CUDA port of the same engine, both exposed to Python through pybind11.

```
src/common.hpp      MDParams, cell grid, Maxwell-Boltzmann init (shared)
src/atoms_cpu.hpp   CPU engine: cell list -> Verlet half neighbour list (Newton 3rd law)
src/atoms_gpu.cu    GPU engine: same algorithm, one thread per atom, full list (no atomics)
src/atoms_gpu.cuh   GPU class declaration (no CUDA types, so g++ can include it)
src/bindings.cpp    pybind11 module mymd._mymd
src/main.cpp        standalone demo + timing (make cpu / make gpu)
python/mymd/        Python package
python/validate.py  CPU vs GPU, nlist vs all-pairs correctness checks
python/benchmark.py timing sweep -> benchmark.csv / benchmark.png
```

## Build

```bash
pip install -e .                      # CPU + GPU if nvcc is found
MYMD_NO_CUDA=1 pip install -e .       # CPU only
make cpu gpu                          # standalone executables md_cpu / md_gpu
```

nvcc 12.0 does not accept gcc 13; `setup.py` detects this and passes `-ccbin g++-12`
(override with `NVCC_CCBIN=...`). GPU architecture defaults to `native` (override with `CUDA_ARCH=sm_89`).

## Use

```python
import mymd
x, box = mymd.lj_liquid(32, rho=0.8)          # 32768 atoms, periodic
cpu = mymd.AtomsCPU(x, box=box, kT=1.0)
gpu = mymd.AtomsGPU(x, box=box, kT=1.0)       # same seed -> same initial velocities
gpu.run(1000, langevin=True)
print(gpu.temperature, gpu.epot, gpu.positions.shape)
```

`box=(0,0,0)` (default) means open boundaries; `use_nlist=False` gives the original O(N²) loop.

## Notes

- Everything is FP64 on both sides. Consumer GPUs (e.g. RTX 4060) run FP64 at 1/64 of FP32
  rate, so the speed-up is far below what an FP32 or mixed-precision kernel would give.
- The CPU engine is single-threaded.
- The potential is truncated but not shifted, so NVE energy jumps slightly when pairs cross rcut.
