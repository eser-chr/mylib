import copy
import pickle

import numpy as np
import pytest

import fastatoms as fa


def make():
    return fa.Atoms("Cu2Ga", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[4, 4, 4], pbc=True)


def test_basic_properties():
    a = make()
    assert len(a) == 3
    assert a.get_chemical_symbols() == ["Cu", "Cu", "Ga"]
    assert a.numbers.tolist() == [29, 29, 31]
    assert a.get_chemical_formula() == "Cu2Ga"
    assert np.allclose(a.get_cell(), np.diag([4.0, 4.0, 4.0]))
    assert a.get_pbc().tolist() == [True, True, True]
    assert a.get_volume() == pytest.approx(64.0)


def test_append_and_positions():
    a = make()
    a.append("O", [0.5, 0.5, 0.5])
    assert len(a) == 4
    assert a[-1].symbol == "O"

    a.positions += 1.0  # getter copy, in-place add, setter
    assert np.allclose(a.positions[0], [1, 1, 1])

    pos = a.get_positions()
    pos[0] = 99.0
    assert not np.allclose(a.positions[0], 99.0)  # get_positions returns a copy

    a.set_positions(np.zeros((4, 3)))
    assert np.allclose(a.positions, 0.0)
    with pytest.raises(ValueError):
        a.set_positions(np.zeros((3, 3)))


def test_construct_from_atom_objects():
    a = fa.Atoms([fa.Atom("H", [0, 0, 0]), fa.Atom("O", [0, 0, 1])])
    assert a.get_chemical_symbols() == ["H", "O"]
    assert np.allclose(a.positions[1], [0, 0, 1])


def test_indexing_and_delete():
    a = make()
    sub = a[[0, 2]]
    assert sub.get_chemical_symbols() == ["Cu", "Ga"]
    assert len(a[a.numbers == 29]) == 2
    assert len(a[1:]) == 2
    assert [at.symbol for at in a] == ["Cu", "Cu", "Ga"]

    del a[0]
    assert len(a) == 2
    at = a.pop()
    assert at.symbol == "Ga"
    assert len(a) == 1


def test_arrays_follow_atoms():
    a = make()
    a.set_array("forces", np.arange(9.0).reshape(3, 3))
    a.set_array("tag", np.array([1, 2, 3]))
    del a[1]
    assert np.allclose(a.get_array("forces"), [[0, 1, 2], [6, 7, 8]])
    assert a.get_array("tag").tolist() == [1, 3]
    a.append("H", [0, 0, 0])
    assert a.get_array("tag").tolist() == [1, 3, 0]
    with pytest.raises(ValueError):
        a.set_array("forces", np.zeros((2, 3)))


def test_extend_and_add():
    a = make()
    b = a + a
    assert len(b) == 6
    a += a
    assert len(a) == 6


def test_wrap_and_mic():
    a = fa.Atoms("H2", positions=[[0.1, 0, 0], [3.9, 0, 0]], cell=[4, 4, 4], pbc=True)
    assert a.get_distance(0, 1, mic=True) == pytest.approx(0.2)
    assert a.get_distance(0, 1) == pytest.approx(3.8)
    assert np.allclose(a.get_all_distances(mic=True), [[0, 0.2], [0.2, 0]])

    a.positions = [[-0.5, 0, 0], [4.5, 0, 0]]
    a.wrap()
    assert np.allclose(a.positions[:, 0], [3.5, 0.5])


def test_scaled_positions_and_set_cell():
    a = make()
    a.set_cell([8, 8, 8], scale_atoms=True)
    assert np.allclose(a.positions[1], [2, 0, 0])
    assert np.allclose(a.get_scaled_positions()[1], [0.25, 0, 0])


def test_info_pickle_copy():
    a = make()
    a.info["energy"] = -1.5
    a.info["step"] = 3
    a.info["label"] = "bulk"
    a.set_array("forces", np.ones((3, 3)))

    b = pickle.loads(pickle.dumps(a))
    assert b.info["energy"] == -1.5
    assert b.info["step"] == 3
    assert b.info["label"] == "bulk"
    assert np.allclose(b.get_array("forces"), 1.0)

    c = copy.deepcopy(a)
    c.positions += 1.0
    assert not np.allclose(a.positions, c.positions)
