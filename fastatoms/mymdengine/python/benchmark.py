"""CPU vs GPU timing: ms per Langevin step vs number of atoms.

    python python/benchmark.py                 # default sizes, writes benchmark.png/.csv
    python python/benchmark.py --max-n 64      # up to 64^3 = 262144 atoms

System: periodic LJ liquid, rho = 0.8, kT = 1, rcut = 2.5, skin = 0.3, dt = 0.005.
Times are wall-clock per step inside engine.run(), which excludes Python overhead
and, for the GPU, includes the final cudaDeviceSynchronize.
"""
import argparse
import csv
import time

import numpy as np

import fastatoms.mymdengine.python.mymd as mymd

# Budget: skip a variant once a single step exceeds this many seconds.
MAX_STEP_SECONDS = 0.5


def time_per_step(engine, target_seconds=1.0):
    engine.run(2, langevin=True)                    # warm-up (+ first list builds)
    t0 = time.perf_counter()
    engine.run(1, langevin=True)
    one = max(time.perf_counter() - t0, 1e-6)
    nsteps = int(np.clip(target_seconds / one, 3, 5000))
    t0 = time.perf_counter()
    engine.run(nsteps, langevin=True)
    return (time.perf_counter() - t0) / nsteps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=48, help="largest lattice side (N = n^3)")
    ap.add_argument("--out", default="benchmark")
    args = ap.parse_args()

    variants = [("CPU", "nlist"), ("CPU", "N^2")]
    if mymd.has_cuda:
        variants += [("GPU", "nlist"), ("GPU", "N^2")]
    classes = {"CPU": mymd.AtomsCPU, "GPU": getattr(mymd, "AtomsGPU", None)}

    sides = [n for n in (6, 8, 10, 13, 16, 20, 25, 32, 40, 48, 56, 64, 80) if n <= args.max_n]
    results = {v: [] for v in variants}
    dropped = set()

    print(f"{'N':>8s}" + "".join(f"{b + ' ' + a:>14s}" for b, a in variants) + "   [ms/step]")
    for n in sides:
        x, box = mymd.lj_liquid(n)
        row = f"{len(x):8d}"
        for v in variants:
            if v in dropped:
                row += f"{'-':>14s}"
                continue
            backend, algo = v
            e = classes[backend](x, box=box, use_nlist=(algo == "nlist"))
            t = time_per_step(e)
            results[v].append((len(x), t))
            row += f"{1e3 * t:14.4f}"
            if t > MAX_STEP_SECONDS:
                dropped.add(v)
            del e
        print(row, flush=True)

    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "algorithm", "N", "seconds_per_step"])
        for (b, a), rows in results.items():
            for N, t in rows:
                w.writerow([b, a, N, t])
    print(f"wrote {args.out}.csv")

    if mymd.has_cuda:
        cpu = dict(results[("CPU", "nlist")])
        gpu = dict(results[("GPU", "nlist")])
        print("\nGPU speed-up (neighbour list):")
        for N in sorted(set(cpu) & set(gpu)):
            print(f"  N = {N:8d}   x{cpu[N] / gpu[N]:6.1f}")

    plot(results, args.out + ".png")


def plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    # Colour = backend, line style = algorithm.
    color = {"CPU": "#2a78d6", "GPU": "#eb6834"}
    style = {"nlist": "-", "N^2": "--"}
    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3dc"

    has_speedup = ("GPU", "nlist") in results
    fig, axes = plt.subplots(1, 2 if has_speedup else 1, figsize=(11 if has_speedup else 6, 4.4),
                             squeeze=False)
    for ax in axes[0]:
        ax.set_facecolor("white")
        ax.grid(True, which="major", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelcolor=ink)

    ax = axes[0][0]
    for (b, a), rows in results.items():
        if not rows:
            continue
        N, t = np.array(rows).T
        label = f"{b}, {'neighbour list' if a == 'nlist' else 'all pairs O(N²)'}"
        ax.loglog(N, 1e3 * t, style[a], color=color[b], lw=2, marker="o", ms=5, label=label)
        # direct label at the line end, in text ink; CPU N^2 ends next to GPU N^2, so it goes left
        left = (b, a) == ("CPU", "N^2")
        ax.annotate(f"{b} {'nlist' if a == 'nlist' else 'N²'}", (N[-1], 1e3 * t[-1]),
                    xytext=(-8 if left else 6, 0), textcoords="offset points", va="center",
                    ha="right" if left else "left", fontsize=9, color=ink)
    ax.set_xlabel("atoms N", color=ink)
    ax.set_ylabel("time per step [ms]", color=ink)
    ax.set_title("Cost per Langevin step", color=ink, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)
    ax.margins(x=0.12)

    if has_speedup:
        ax = axes[0][1]
        cpu = dict(results[("CPU", "nlist")])
        gpu = dict(results[("GPU", "nlist")])
        N = np.array(sorted(set(cpu) & set(gpu)))
        s = np.array([cpu[n] / gpu[n] for n in N])
        ax.semilogx(N, s, "-", color=color["GPU"], lw=2, marker="o", ms=5)
        ax.axhline(1.0, color=muted, lw=1)
        ax.annotate(f"×{s[-1]:.0f}", (N[-1], s[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=ink)
        ax.set_xlabel("atoms N", color=ink)
        ax.set_ylabel("CPU time / GPU time", color=ink)
        ax.set_title("GPU speed-up, neighbour list (both FP64)", color=ink, loc="left")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.12)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
