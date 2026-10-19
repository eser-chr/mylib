"""Lennard-Jones MD engine with a serial CPU backend and a CUDA backend.

Both classes share the same constructor and API:

    AtomsCPU / AtomsGPU(positions, box=(0,0,0), kT=1.0, mass=1.0, dt=0.005, gamma=1.0,
                        epsilon=1.0, sigma=1.0, rcut=2.5, skin=0.3, seed=42, use_nlist=True)

    .run(nsteps, langevin=False)   .verlet_step()   .langevin_step()
    .positions  .velocities  .forces  .epot  .ekin  .temperature  .n_builds
"""
import numpy as np

from .....mymdengine.python.mymd._mymd import AtomsCPU, has_cuda

if has_cuda:
    from .....mymdengine.python.mymd._mymd import AtomsGPU

__all__ = ["AtomsCPU", "AtomsGPU", "has_cuda", "cubic_lattice", "lj_liquid"]


def cubic_lattice(n, a):
    """n x n x n simple cubic lattice with spacing a, shape (n^3, 3)."""
    r = (np.arange(n) + 0.5) * a
    return np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)


def lj_liquid(n, rho=0.8):
    """Periodic LJ system of N = n^3 atoms at density rho. Returns (positions, box)."""
    a = (1.0 / rho) ** (1.0 / 3.0)
    return cubic_lattice(n, a), (n * a,) * 3
