"""Check that all four variants (CPU/GPU x neighbour-list/all-pairs) agree.

    python python/validate.py
"""
import numpy as np

import fastatoms.mymdengine.python.mymd as mymd

ENGINES = [("CPU", mymd.AtomsCPU)] + ([("GPU", mymd.AtomsGPU)] if mymd.has_cuda else [])


def check(name, a, b, tol):
    err = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    print(f"  {name:<40s} max|diff| = {err:.2e}  {'OK' if err < tol else 'FAIL'}")
    return err < tol


def compare_forces(label, x, **kw):
    """Forces and energy of one configuration, all variants vs CPU all-pairs."""
    print(f"{label}: forces/energy on one configuration")
    ref = mymd.AtomsCPU(x, use_nlist=False, **kw)
    ok = True
    for name, cls in ENGINES:
        for nl in (True, False):
            if name == "CPU" and not nl:
                continue
            a = cls(x, use_nlist=nl, **kw)
            tag = f"{name} {'nlist' if nl else 'N^2  '}"
            ok &= check(f"{tag} forces vs CPU N^2", a.forces, ref.forces, 1e-9)
            ok &= check(f"{tag} epot   vs CPU N^2", a.epot, ref.epot, 1e-9)
    return ok


def compare_trajectory(label, x, nsteps, **kw):
    """Short NVE run: identical start, so trajectories agree until chaos amplifies round-off."""
    print(f"{label}: {nsteps} NVE steps from identical initial state")
    ref = mymd.AtomsCPU(x, use_nlist=False, **kw)
    ref.run(nsteps)
    ok = True
    for name, cls in ENGINES:
        a = cls(x, use_nlist=True, **kw)
        a.run(nsteps)
        # compare wrapped positions (nlist builds wrap atoms into the box)
        box = np.asarray(kw.get("box", (0, 0, 0)), dtype=float)
        d = a.positions - ref.positions
        if box.all():
            d -= box * np.round(d / box)
        ok &= check(f"{name} nlist positions vs CPU N^2", d, 0.0, 1e-6)
        ok &= check(f"{name} nlist Etot       vs CPU N^2", a.epot + a.ekin, ref.epot + ref.ekin, 1e-6)
        print(f"  ({name} neighbour-list builds: {a.n_builds})")
    return ok


def energy_drift(cls, nsteps=20000, **kw):
    x, box = mymd.lj_liquid(8)
    a = cls(x, box=box, **kw)
    e0 = a.epot + a.ekin
    es = []
    for _ in range(20):
        a.run(nsteps // 20)
        es.append(a.epot + a.ekin)
    return (np.array(es) - e0) / a.N


if __name__ == "__main__":
    print(f"mymd.has_cuda = {mymd.has_cuda}\n")
    rng = np.random.default_rng(0)
    ok = True

    # open cluster (the original main.cpp system), slightly perturbed
    x = mymd.cubic_lattice(6, 1.2) + 0.05 * rng.standard_normal((216, 3))
    ok &= compare_forces("Open cluster N=216", x, kT=0.5)
    ok &= compare_trajectory("Open cluster N=216", x, 200, kT=0.5)

    # periodic liquid, large enough that the cell list is used
    x, box = mymd.lj_liquid(12)
    x = x + 0.05 * rng.standard_normal(x.shape)
    ok &= compare_forces(f"Periodic liquid N={len(x)}", x, box=box)
    ok &= compare_trajectory(f"Periodic liquid N={len(x)}", x, 200, box=box)

    print("\nEnergy drift per atom, NVE, N=512, dt=0.005 (truncated, unshifted LJ):")
    for name, cls in ENGINES:
        d = energy_drift(cls)
        print(f"  {name}: final {d[-1]:+.2e}, max |dE/N| {np.abs(d).max():.2e}")

    print("\nLangevin thermostat, target kT = 1.5:")
    x, box = mymd.lj_liquid(8)
    for name, cls in ENGINES:
        a = cls(x, box=box, kT=1.5)
        a.run(2000, langevin=True)
        T = []
        for _ in range(100):
            a.run(20, langevin=True)
            T.append(a.temperature)
        print(f"  {name}: <T> = {np.mean(T):.3f} +- {np.std(T) / np.sqrt(len(T)):.3f}")

    print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
