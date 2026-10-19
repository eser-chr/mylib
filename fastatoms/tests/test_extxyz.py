import numpy as np
import pytest

import fastatoms as fa

SAMPLE = """3
Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" Properties=species:S:1:pos:R:3:forces:R:3:tag:I:1 energy=-12.5 step=7 config_type="bulk cu" converged=T stress="1 2 3 4 5 6" pbc="T T F"
Cu 0.0 0.0 0.0 0.1 0.2 0.3 1
Cu 2.0 0.0 0.0 -0.1 -0.2 -0.3 2
Ga 0.0 2.0 0.0 0.0 0.0 0.0 3
"""


def random_frames(n=5, natoms=18, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        a = fa.Atoms("Cu10Ga5O3", positions=rng.random((natoms, 3)) * 10, cell=[10, 10, 10], pbc=True)
        a.info["energy"] = float(rng.normal())
        a.set_array("forces", rng.normal(size=(natoms, 3)))
        frames.append(a)
    return frames


def test_parse_sample():
    a = fa.parse_extxyz(SAMPLE, index=0)
    assert a.get_chemical_symbols() == ["Cu", "Cu", "Ga"]
    assert a.get_pbc().tolist() == [True, True, False]
    assert np.allclose(a.cell, np.diag([4.0, 4.0, 4.0]))
    assert a.info["energy"] == -12.5
    assert a.info["step"] == 7
    assert a.info["config_type"] == "bulk cu"
    assert a.info["converged"] is True
    assert a.info["stress"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert np.allclose(a.get_array("forces")[1], [-0.1, -0.2, -0.3])
    assert a.get_array("tag").tolist() == [1, 2, 3]


def test_roundtrip(tmp_path):
    frames = random_frames()
    path = tmp_path / "frames.extxyz"
    fa.write(path, frames)

    back = fa.read(path, ":")
    assert len(back) == len(frames)
    for x, y in zip(frames, back):
        assert x.get_chemical_symbols() == y.get_chemical_symbols()
        np.testing.assert_allclose(x.positions, y.positions, rtol=0, atol=1e-8)
        np.testing.assert_allclose(x.get_array("forces"), y.get_array("forces"), rtol=0, atol=1e-8)
        assert x.info["energy"] == y.info["energy"]  # info floats round-trip exactly

    assert fa.read(path).info["energy"] == frames[-1].info["energy"]
    assert len(fa.read(path, "1:4")) == 3
    assert len(fa.read(path, slice(None, None, 2))) == 3
    assert fa.count_frames(path) == 5


def test_append(tmp_path):
    frames = random_frames()
    path = tmp_path / "append.extxyz"
    fa.write(path, frames[:2])
    fa.write(path, frames[2:], append=True)
    assert fa.count_frames(path) == 5


def test_format_parse_string():
    frames = random_frames(3)
    text = fa.format_extxyz(frames)
    back = fa.parse_extxyz(text)
    assert len(back) == 3


def test_errors(tmp_path):
    with pytest.raises(RuntimeError):
        fa.parse_extxyz("3\n\nCu 0 0 0\n")
    with pytest.raises(RuntimeError):
        fa.parse_extxyz("1\n\nCu 0 0\n")
    with pytest.raises(FileNotFoundError):
        fa.read(tmp_path / "does-not-exist.xyz")
    path = tmp_path / "f.extxyz"
    fa.write(path, random_frames(2))
    with pytest.raises(IndexError):
        fa.read(path, 5)


def test_compatible_with_ase(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3))
    atoms.rattle(stdev=0.05, seed=1)
    atoms.info["foo"] = 1.25

    p = tmp_path / "ase.extxyz"
    ase_io.write(p, [atoms] * 3, format="extxyz")
    ours = fa.read(p, ":")
    assert len(ours) == 3
    np.testing.assert_allclose(ours[0].positions, atoms.positions, atol=1e-7)
    np.testing.assert_allclose(ours[0].cell, atoms.cell[:], atol=1e-7)
    assert ours[0].info["foo"] == 1.25

    q = tmp_path / "ours.extxyz"
    fa.write(q, ours)
    back = ase_io.read(q, ":")
    np.testing.assert_allclose(back[0].positions, atoms.positions, atol=1e-7)

    converted = ours[0].to_ase()
    np.testing.assert_allclose(converted.positions, atoms.positions, atol=1e-7)
    again = fa.from_ase(converted)
    assert len(again) == len(atoms)
