import re
import numpy as np
import numpy.typing as npt
import pandas as pd
import ase
from ase import Atoms
from ase.build import (
    bulk,
    bcc100,
    hcp10m10,
    bcc110,
    hcp0001,
    bcc111,
    hcp0001_root,
    bcc111_root,
    fcc100,
    fcc110,
    fcc111,
    fcc111_root,
)
from wulffpack import SingleCrystal
from typing import Callable, Dict, Optional, Tuple




# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------



def _extract_lattice_params(df, element, cs):
    row = df[(df["element"] == element) & (df["crystalstructure"] == cs)]
    return float(row["a"].iloc[0]), float(row["c"].iloc[0])


def extract_lattice_params(
    df: pd.DataFrame, from_elem: str, to_elem: str, composition: float, structure: str
) -> Tuple[float, float]:
    """
    Look up and interpolate lattice parameters (a, c) from `df` based on elements
    and desired composition and crystal structure.
    """
    a1, c1 = _extract_lattice_params(df, from_elem, structure)
    a2, c2 = _extract_lattice_params(df, to_elem, structure)
    a = (1 - composition) * a1 + composition * a2
    c = (1 - composition) * c1 + composition * c2
    return a, c


def _extract_surf_energy(cs, miller, df, element_a, element_b, concentration_of_b):
    a = df.loc[
        (df["element"] == element_a) & (df["cs"] == cs) & (df["miller"] == miller),
        "gamma",
    ].iloc[0]
    b = df.loc[
        (df["element"] == element_b) & (df["cs"] == cs) & (df["miller"] == miller),
        "gamma",
    ].iloc[0]
    return a * (1 - concentration_of_b) + b * concentration_of_b


def find_common_surfaces(cs, df, element_a, element_b):
    tmpa = df[(df["element"] == element_a) & (df["cs"] == cs)]["miller"].unique()
    tmpb = df[(df["element"] == element_b) & (df["cs"] == cs)]["miller"].unique()
    return list(set(tmpa).intersection(set(tmpb)))


def miller_to_tuple(miller):
    return tuple([int(i) for i in miller])


def extract_surface_energies(cs, df, element_a, element_b, concentration_of_b):
    common_surfaces = find_common_surfaces(cs, df, element_a, element_b)
    return {
        miller_to_tuple(miller): _extract_surf_energy(
            cs, miller, df, element_a, element_b, concentration_of_b
        )
        for miller in common_surfaces
    }


def substitute_random_atoms(
    atoms: ase.Atoms,
    frac: float,
    new_symbol: str,
    rng: Optional[np.random.Generator] = None,
) -> ase.Atoms:
    """
    Replace `frac` fraction of `atoms`’ symbols at random with `new_symbol`.
    Returns the modified Atoms object (in place).
    """
    rng = rng or np.random.default_rng()
    total = len(atoms)
    n_replace = int(round(total * frac))
    indices = rng.choice(total, size=n_replace, replace=False)
    for i in indices:
        atoms[i].symbol = new_symbol # type: ignore
    return atoms


# -------------------------------------------------------------------
# Core Builders
# -------------------------------------------------------------------


def build_bulk_structure(
    structure: str,
    df: pd.DataFrame,
    composition: float,
    symbol_from: str,
    symbol_to: str,
    supercell: Tuple[int, int, int] = (4, 4, 4),
) -> ase.Atoms:
    """
    Create a bulk supercell of `structure` using `symbol_from`, then
    randomly substitute `composition` fraction with `symbol_to`.
    """
    a, c = extract_lattice_params(df, symbol_from, symbol_to, composition, structure)
    base = bulk(symbol_from, crystalstructure=structure, a=a, c=c)
    supercell_atoms = base * supercell
    return substitute_random_atoms(supercell_atoms, composition, symbol_to)


def build_cluster(
    crystal_structure: str,
    df: pd.DataFrame,
    composition: float,
    symbol_from: str,
    symbol_to: str,
    repeat: int = 3,
    radius: float = 6.0,
    cell_size: float = 20.0,
) -> ase.Atoms:
    """
    Build a spherical nanoparticle (“cluster”) of radius `radius` Å
    from a repeated bulk supercell.

    Params:
    ---------
    crystal_structure: "fcc", "bcc", "hcp", .. as supported by ase and the passed dataframe


    """
    # 1) Build & substitute bulk
    atoms:Atoms = build_bulk_structure(
        structure=crystal_structure,
        df=df,
        composition=composition,
        symbol_from=symbol_from,
        symbol_to=symbol_to,
        supercell=(repeat, repeat, repeat),
    )

    # 2) Carve out a sphere
    com = atoms.get_center_of_mass()
    mask = np.linalg.norm(atoms.get_positions() - com, axis=1) <= radius
    cluster = atoms[mask].copy() # type: ignore

    # 3) Center in a cubic cell
    cluster.set_cell((cell_size,) * 3 + (90, 90, 90))
    cluster.center()

    return cluster


_SURFACE_GENS: Dict[str, Callable] = {
    "bcc100": bcc100,
    "hcp10m10": hcp10m10,
    "bcc110": bcc110,
    "hcp0001": hcp0001,
    "bcc111": bcc111,
    "hcp0001_root": hcp0001_root,
    "bcc111_root": bcc111_root,
    "fcc100": fcc100,
    "fcc110": fcc110,
    "fcc111": fcc111,
    "fcc111_root": fcc111_root,
}


def parse_surface_name(name: str) -> Tuple[str, str, bool]:
    """
    "fcc111_root" → ("fcc", "111", True)
    """
    pattern = re.compile(r"^([A-Za-z]{3})([0-9m]+)(_root)?$")
    m = pattern.match(name)
    if not m:
        raise ValueError(f"Invalid surface name: {name!r}")
    return m.group(1), m.group(2), bool(m.group(3))


def build_surface(
    structure: str,
    df: pd.DataFrame,
    composition: float,
    symbol_from: str,
    symbol_to: str,
    repeat: Tuple[int, int, int] = (2, 2, 10),
    vacuum: float = 10.0,
    root_layers: int = 3,
) -> ase.Atoms:
    """
    Generate a slab (surface) of type `structure`, randomly substitute
    `composition` fraction, and return the ASE Atoms.
    """
    cs, miller, is_root = parse_surface_name(structure)
    gen = _SURFACE_GENS.get(structure)
    if gen is None:
        raise KeyError(f"No generator for surface '{structure}'")

    # lattice params
    a, c = extract_lattice_params(df, symbol_from, symbol_to, composition, cs)

    # build slab
    kwargs = dict(a=a, vacuum=vacuum)
    if cs == "hcp":
        kwargs["c"] = c

    slab = gen(symbol_from, *([root_layers] if is_root else []), repeat, **kwargs)

    return substitute_random_atoms(slab, composition, symbol_to)


# -------------------------------------------------------------------
# NANO-PARTICLE
# -------------------------------------------------------------------


def calc_box_size(pos: npt.NDArray) -> npt.NDArray:
    M = np.max(pos, axis=0)
    m = np.min(pos, axis=0)
    Dx = M - m
    return Dx


def create_nanoparticle(
    prim_st: ase.Atoms, surface_energies: dict[tuple, float], N: int
) -> ase.Atoms:
    particle = SingleCrystal(
        surface_energies=surface_energies, primitive_structure=prim_st, natoms=N
    )

    atoms = particle.atoms
    box_size = calc_box_size(atoms.positions)
    atoms.set_cell(np.diag(box_size))
    atoms.center()

    return atoms


def create_nnpt(
    structure,
    element_a,
    element_b,
    concentration_of_b,
    df_bulk: pd.DataFrame,
    df_surface: pd.DataFrame,
    vacuum:float = 10.0,
    N: int = 256
):

    a, c = extract_lattice_params(
        df_bulk, element_a, element_b, concentration_of_b, structure
    )
    base = bulk(element_a, crystalstructure=structure, a=a, c=c)

    surface_energies = extract_surface_energies(
        structure, df_surface, element_a, element_b, concentration_of_b
    )

    particle = SingleCrystal(
        surface_energies=surface_energies, primitive_structure=base, natoms=N
    )


    atoms = particle.atoms
    box_size = calc_box_size(atoms.positions)
    atoms.set_cell(np.diag(box_size+vacuum))
    atoms.center()


    return substitute_random_atoms(atoms, concentration_of_b, element_b)
