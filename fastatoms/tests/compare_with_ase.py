import time
import numpy as np
import ase.io
import fastatoms as fa

path = "3000.extxyz"   # any extxyz you already have

# --- read ---
t0 = time.perf_counter()
ase_frames = ase.io.read(path, ":")
t_ase_read = time.perf_counter() - t0

t0 = time.perf_counter()
fa_frames = fa.read(path, ":")
t_fa_read = time.perf_counter() - t0

# --- write ---
t0 = time.perf_counter()
ase.io.write("out_ase.extxyz", ase_frames)
t_ase_write = time.perf_counter() - t0

t0 = time.perf_counter()
fa.write("out_fa.extxyz", fa_frames)
t_fa_write = time.perf_counter() - t0

print(f"{len(fa_frames)} frames")
print(f"read   ase {t_ase_read:.3f} s   fastatoms {t_fa_read:.3f} s   speedup {t_ase_read / t_fa_read:.1f}x")
print(f"write  ase {t_ase_write:.3f} s   fastatoms {t_fa_write:.3f} s   speedup {t_ase_write / t_fa_write:.1f}x")

# --- sanity check that both read the same thing ---
assert len(ase_frames) == len(fa_frames)
for a, f in zip(ase_frames, fa_frames):
    assert a.get_chemical_symbols() == f.get_chemical_symbols()
    np.testing.assert_allclose(a.positions, f.positions, atol=1e-8)
    np.testing.assert_allclose(a.cell[:], f.cell, atol=1e-8)
    if a.calc is not None and "forces" in a.calc.results:
        np.testing.assert_allclose(a.get_forces(), f.get_array("forces"), atol=1e-8)
print("contents match")