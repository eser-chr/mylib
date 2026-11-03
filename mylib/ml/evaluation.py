import logging
from typing import Sequence, Callable, List, Tuple, Literal, TypeAlias

import numpy as np
from ase import Atoms
from joblib import Parallel, delayed
# from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Calculator:TypeAlias = Callable[[Atoms], None]


def compute_energies(trajectory: Sequence[Atoms], calculator: Calculator) -> np.ndarray:
    """
    Compute potential energies for each snapshot in a trajectory.

    Parameters
    ----------
    trajectory : Sequence[Atoms]
        Iterable of ASE Atoms snapshots.
    calculator : Calculator
        ASE-compatible calculator instance.

    Returns
    -------
    np.ndarray
        Energies of shape (n_snapshots,).
    """
    n_snap = len(trajectory)
    energies = np.empty(n_snap)
    for i, atoms in enumerate(trajectory):
        if len(atoms) == 0:
            raise ValueError(f"Snapshot {i} has no atoms.")
        atoms.calc = calculator
        try:
            energies[i] = atoms.get_potential_energy()
        except Exception as e:
            logger.error(f"Error computing energy for snapshot {i}: {e}")
            raise
    return energies


def compute_forces(trajectory: Sequence[Atoms], calculator: Calculator) -> np.ndarray:
    """
    Compute forces for each snapshot in a trajectory.

    Parameters
    ----------
    trajectory : Sequence[Atoms]
        Iterable of ASE Atoms snapshots.
    calculator : Calculator
        ASE-compatible calculator instance.

    Returns
    -------
    np.ndarray
        Forces of shape (n_snapshots, n_atoms, 3).
    """
    n_snap = len(trajectory)
    n_atoms = len(trajectory[0])
    forces = np.empty((n_snap, n_atoms, 3))
    for i, atoms in enumerate(trajectory):
        if len(atoms) != n_atoms:
            raise ValueError(
                f"Inconsistent atom count: snapshot 0 has {n_atoms}, snapshot {i} has {len(atoms)}"
            )
        atoms.calc = calculator
        try:
            forces[i] = atoms.get_forces()
        except Exception as e:
            logger.error(f"Error computing forces for snapshot {i}: {e}")
            raise
    return forces


def _parallel_model_eval(
    trajectory: Sequence[Atoms],
    calculators: Sequence[Calculator],
    eval_fn: Callable[[Sequence[Atoms], Calculator], np.ndarray],
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Run `eval_fn(trajectory, calc)` in parallel for each calculator.

    Returns
    -------
    np.ndarray
        Stacked results with shape (n_models, ...) matching eval_fn output.
    """
    logger.info("Starting parallel evaluation for %d models...", len(calculators))
    results = Parallel(n_jobs=n_jobs)(
        delayed(eval_fn)(trajectory, calc) for calc in calculators
    )
    return np.stack(results, axis=0) # type: ignore


class EnsembleEvaluator:
    """
    Evaluate ensembles of calculators on an ASE trajectory to compute energies,
    forces, and their uncertainties.
    """

    def __init__(
        self,
        trajectory: Sequence[Atoms],
        calculators: Sequence[Calculator],
        n_jobs: int = -1,
    ):
        if len(trajectory) == 0 or len(calculators) == 0:
            raise ValueError("Trajectory and calculators must be non-empty.")
        self.traj = trajectory
        self.calculators = calculators
        self.n_jobs = n_jobs
        self._n_atoms = len(trajectory[0])

    def energies(self) -> np.ndarray:
        """
        Compute energies for each snapshot and model.

        Returns
        -------
        np.ndarray
            Array of shape (n_snapshots, n_models).
        """
        em = _parallel_model_eval(
            self.traj, self.calculators, compute_energies, self.n_jobs
        )
        return em.T  # (models, steps) -> (steps, models)

    def forces(self) -> np.ndarray:
        """
        Compute forces for each snapshot and model.

        Returns
        -------
        np.ndarray
            Array of shape (n_snapshots, n_models, n_atoms, 3).
        """
        fm = _parallel_model_eval(
            self.traj, self.calculators, compute_forces, self.n_jobs
        )
        return fm.transpose(1, 0, 2, 3)

    def uncertainty(
        self,
        what: Literal["energy", "force"],
        agg: Literal["std", "mean", "max", "median"] = "std",
        normalize_energy: bool = True,
    ) -> np.ndarray:
        """
        Compute uncertainty across models for energy or force.

        Parameters
        ----------
        what : 'energy' or 'force'
            Quantity to compute uncertainty for.
        agg : 'std', 'mean', 'max', or 'median'
            Aggregation method over model axis.
        normalize_energy : bool
            If True and what='energy', divide energies by n_atoms before agg.

        Returns
        -------
        np.ndarray
            For 'energy': shape (n_snapshots,).
            For 'force': shape (n_snapshots,) after aggregating over atoms and components.
        """
        if what == "energy":
            E = self.energies()  # (steps, models)
            if normalize_energy:
                E = E / self._n_atoms
            if agg == "std":
                return E.std(axis=1)
            elif agg == "mean":
                return E.mean(axis=1)
            elif agg == "max":
                return E.max(axis=1)
            elif agg == "median":
                return np.median(E, axis=1)
            else:
                raise ValueError(f"Unknown agg: {agg}")

        elif what == "force":
            F = self.forces()  # (steps, models, atoms, 3)
            # std over models
            f_std = F.std(axis=1)  # (steps, atoms, 3)
            # aggregate over atoms & components
            if agg == "std":
                out = f_std.std(axis=(1, 2))
            elif agg == "mean":
                out = f_std.mean(axis=(1, 2))
            elif agg == "max":
                out = f_std.max(axis=(1, 2))
            elif agg == "median":
                out = np.median(f_std, axis=(1, 2))
            else:
                raise ValueError(f"Unknown agg: {agg}")
            return out

        else:
            raise ValueError(f"Unknown quantity: {what}")

    def select_snapshots(
        self,
        fraction: float = 0.25,
        what: Literal["energy", "force"] = "force",
        agg: Literal["std", "mean"] = "mean",
        normalize_energy: bool = True,
    ) -> Tuple[List[Atoms], int]:
        """
        Select a subset of snapshots whose cumulative uncertainty covers the given fraction
        of total uncertainty in ascending order.

        Parameters
        ----------
        fraction : float
            Fraction of total uncertainty to cover (0 < fraction < 1).
        what : 'energy' or 'force'
        agg : 'std' or 'mean'
        normalize_energy : bool

        Returns
        -------
        selected : List[Atoms]
            Snapshots selected.
        count : int
            Number of snapshots selected.
        """
        s = self.uncertainty(what=what, agg=agg, normalize_energy=normalize_energy)
        total = s.sum()
        target = fraction * total
        sorted_idx = np.argsort(s)
        csum = np.cumsum(s[sorted_idx])
        count = int(np.searchsorted(csum, target) + 1)
        return [self.traj[i] for i in sorted_idx[:count]], count


class LogicUncertainty:
    """ 
    Defines how the uncertainty of a snapshot is calculated based on the atom uncertainties.
    
    For example, the uncertainty of the snapshot is equal to the maximum or average of all single atom uncertainties.
    """
    @staticmethod
    def maximum(u: np.ndarray) -> float:
        return u.max()

    @staticmethod
    def average(u: np.ndarray) -> float:
        return u.mean()



def get_traj_uncertainty_F_per_atom(md:Sequence[Atoms], calculators:List[Calculator])->np.ndarray:
    """ Returns the per atom uncertainty as described in literature"""
    
    F = EnsembleEvaluator(md, calculators).forces()
    s = np.std(F, axis = 1)
    return np.mean(s, axis = -1)


def get_traj_uncertainty(md, calculators):
    return np.mean(get_traj_uncertainty_F_per_atom(md, calculators), axis = -1)


def get_uncertain_snapshots(md:List[Atoms], s_mean):
    target = s_mean.sum()/4
    I = np.argsort(s_mean)
    partial_sorted_sum = np.cumsum(np.sort(s_mean))
    index = np.argwhere(partial_sorted_sum>=target)[0][0]
    collected_snapshots = [md[i] for i in I[:index]]
    return collected_snapshots, index



def compare_calculator(calc: MACECalculator, DB: List[Atoms]) -> dict:
    """Compares a MACE calculator with the reference forces of a database.
    
    Assumes each Atoms object in DB has reference forces stored in arrays.get_forces()
    or info/arrays dicts. Computes RMSE and MAE over all atoms and frames.
    """
    sq_errors = []
    abs_errors = []

    for frame in DB:
        ref_forces = frame.arrays["REF_forces"]  # adjust key as needed
        frame.calc = calc
        pred_forces = calc.get_forces(frame)

        diff = pred_forces - ref_forces          # (N_atoms, 3)
        sq_errors.append((diff ** 2).sum(axis=1))   # per-atom squared norm? No — per component
        abs_errors.append(np.abs(diff).flatten())

    sq_errors = np.concatenate(sq_errors)
    abs_errors = np.concatenate(abs_errors)

    return {
        "rmse_f": np.sqrt(sq_errors.mean()),
        "mae_f": abs_errors.mean(),
        "n_frames": len(DB),
        "n_atoms_total": len(sq_errors),
    }


def compare_calculators(calc_a: MACECalculator, calc_b: MACECalculator, DB: List[Atoms]) -> dict:
    """Compares two calculators on a dataset by computing RMSE of their predicted forces.
    
    Does not require reference forces — compares calc_a vs calc_b directly.
    """
    sq_errors = []
    abs_errors = []

    for frame in DB:
        frame_a = frame.copy()
        frame_b = frame.copy()

        frame_a.calc = calc_a
        frame_b.calc = calc_b

        forces_a = calc_a.get_forces(frame_a)
        forces_b = calc_b.get_forces(frame_b)

        diff = forces_a - forces_b               # (N_atoms, 3)
        sq_errors.append((diff ** 2).flatten())
        abs_errors.append(np.abs(diff).flatten())

    sq_errors = np.concatenate(sq_errors)
    abs_errors = np.concatenate(abs_errors)

    return {
        "rmse_f": np.sqrt(sq_errors.mean()),
        "mae_f": abs_errors.mean(),
        "n_frames": len(DB),
        "n_atoms_total": len(sq_errors),
    }

