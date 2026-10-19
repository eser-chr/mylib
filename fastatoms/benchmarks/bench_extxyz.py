"""Compare extxyz read/write speed of ase.io and fastatoms.

    python benchmarks/bench_extxyz.py --frames 2000 --atoms 200
    OMP_NUM_THREADS=1 python benchmarks/bench_extxyz.py   # single-threaded fastatoms
"""

import argparse
import os
import tempfile
import time

import numpy as np

import fastatoms as fa


def build_frames(nframes, natoms, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(nframes):
        a = fa.Atoms(
            numbers=rng.choice([8, 29, 31], size=natoms),
            positions=rng.random((natoms, 3)) * 20.0,
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        a.info["energy"] = float(rng.normal())
        a.set_array("forces", rng.normal(size=(natoms, 3)))
        frames.append(a)
    return frames


def best_of(fn, repeat):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--atoms", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print(f"fastatoms {fa.__version__}  openmp={fa.has_openmp}  fast_float={fa.has_fast_float}")
    frames = build_frames(args.frames, args.atoms)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bench.extxyz")
        rows = []

        t = best_of(lambda: fa.write(path, frames), args.repeat)
        rows.append(("fastatoms write", t))
        size_mb = os.path.getsize(path) / 1e6
        t = best_of(lambda: fa.read(path, ":"), args.repeat)
        rows.append(("fastatoms read", t))

        try:
            import ase.io
        except ImportError:
            ase = None
        if ase is not None:
            ase_frames = ase.io.read(path, ":")
            path_ase = os.path.join(tmp, "bench_ase.extxyz")
            rows.append(("ase write", best_of(lambda: ase.io.write(path_ase, ase_frames), args.repeat)))
            rows.append(("ase read", best_of(lambda: ase.io.read(path, ":"), args.repeat)))

    print(f"{args.frames} frames x {args.atoms} atoms, file size {size_mb:.1f} MB\n")
    timings = dict(rows)
    for name, t in rows:
        print(f"{name:18s} {t:8.3f} s")
    if "ase read" in timings:
        print(f"\nread speedup  {timings['ase read'] / timings['fastatoms read']:.1f}x")
        print(f"write speedup {timings['ase write'] / timings['fastatoms write']:.1f}x")


if __name__ == "__main__":
    main()
