import argparse
import gzip
import os
import shutil

import numpy as np
from ase.io import read, write

SPLITS = ["train", "valid", "test"]
MASTER_NAME = "dataset.extxyz"

parser = argparse.ArgumentParser()
parser.add_argument("root", help="folder containing the model folders")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--pack", action="store_true",
                  help="delete split files, failed models and gzip the dataset")
mode.add_argument("--unpack", action="store_true",
                  help="rebuild split files from the dataset and indices")
args = parser.parse_args()

ROOT = args.root
MASTER = os.path.join(ROOT, MASTER_NAME)


def subdirs():
    return sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))


def split_path(folder, s, suffix=".extxyz"):
    return os.path.join(ROOT, folder, f"{s}{suffix}")


def key(atoms):
    pos = np.round(atoms.positions, 6) + 0.0
    cell = np.round(atoms.cell.array, 6) + 0.0
    return atoms.numbers.tobytes() + pos.tobytes() + cell.tobytes()


def labels(atoms):
    out = dict(atoms.info)
    if atoms.calc is not None:
        out.update(atoms.calc.results)
    out.update(atoms.arrays)
    return out


def same(a, b):
    la, lb = labels(a), labels(b)
    if la.keys() != lb.keys():
        return False
    for k in la:
        va, vb = np.asarray(la[k]), np.asarray(lb[k])
        if np.issubdtype(va.dtype, np.number):
            if va.shape != vb.shape or not np.allclose(va, vb):
                return False
        elif not np.array_equal(va, vb):
            return False
    return True


def unpack():
    folders = [d for d in subdirs() if os.path.isfile(split_path(d, SPLITS[0], "_idx.txt"))]
    master = read(MASTER + ".gz", ":")
    print(f"Master: {len(master)} frames, models: {folders}")
    for folder in folders:
        for s in SPLITS:
            idx = np.loadtxt(split_path(folder, s, "_idx.txt"), dtype=int, ndmin=1)
            write(split_path(folder, s), [master[i] for i in idx])
            print(f"{folder}/{s}: {len(idx)} frames rebuilt")


def pack(delete):
    folders = [d for d in subdirs() if os.path.isdir(os.path.join(ROOT, d, "checkpoints"))]
    failed = [d for d in subdirs() if d not in folders]
    print(f"Models: {folders}")
    print(f"No checkpoints (to delete): {failed}")

    # 1. build full dataset from the first model
    frames = []
    for s in SPLITS:
        frames += read(split_path(folders[0], s), ":")
    write(MASTER, frames)

    # 2. reread from disk, so we check what is actually saved
    master = read(MASTER, ":")
    lookup = {}
    for i, atoms in enumerate(master):
        lookup.setdefault(key(atoms), i)
    print(f"Master: {len(master)} frames, {len(lookup)} unique structures")

    # 3. find and check indices for every model
    all_ok = True
    for folder in folders:
        for s in SPLITS:
            split = read(split_path(folder, s), ":")
            idx = [lookup[key(a)] for a in split]
            bad = sum(not same(master[i], a) for i, a in zip(idx, split))
            np.savetxt(split_path(folder, s, "_idx.txt"), idx, fmt="%d")
            print(f"{folder}/{s}: {len(idx)} frames, {bad} mismatches")
            all_ok = all_ok and bad == 0

    # 4. clean up only if everything matched
    if not all_ok:
        print("Mismatches found, nothing deleted.")
    elif delete:
        for folder in failed:
            shutil.rmtree(os.path.join(ROOT, folder))
            print(f"Deleted folder {folder}")
        for folder in folders:
            for s in SPLITS:
                os.remove(split_path(folder, s))
        with open(MASTER, "rb") as f_in, gzip.open(MASTER + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(MASTER)
        print(f"Packed: deleted split files, saved {MASTER}.gz")
    else:
        print("All checks passed. Run with --pack to delete and compress.")


if args.unpack:
    unpack()
else:
    pack(args.pack)