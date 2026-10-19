#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import pandas as pd
import sys
from tqdm import tqdm

TIME_PATTERNS = {
    "wall_time": re.compile(r"Elapsed time \(sec\):\s*([\d.]+)"),
    "cpu_time":  re.compile(r"Total CPU time used \(sec\):\s*([\d.]+)"),
}


def parse_outcar_times(outcar: Path, tail_lines: int = 20) -> dict[str, float]:
    """
    Read the last `tail_lines` of an OUTCAR and extract wall_time & cpu_time.
    Returns -1.0 for any missing value.
    """
    # Initialize both times to a sentinel
    times = {"cpu_time": -1.0, "wall_time": -1.0}

    # Read the last `tail_lines` lines directly
    with outcar.open() as f:
        lines = f.readlines()[-tail_lines:]

    for line in lines:
        for key, pattern in TIME_PATTERNS.items():
            match = pattern.search(line)
            if match:
                times[key] = float(match.group(1))
        # stop early if we've found both
        if times["cpu_time"] != -1.0 and times["wall_time"] != -1.0:
            break

    return times


def collect_times(root: Path) -> list[dict]:
    """
    Walk `root`, look for subdirs containing both OUTCAR & vasprun.xml,
    parse times, and return a list of dicts.
    """
    results = []
    for subdir in tqdm(root.rglob("*")):
        if subdir.is_dir() and {"OUTCAR", "vasprun.xml"} <= {p.name for p in subdir.iterdir()}:
            times = parse_outcar_times(subdir / "OUTCAR")
            results.append({"path": str(subdir), **times})
    return results


def main():
    p = argparse.ArgumentParser(description="Extract OUTCAR timings from a simulation tree")
    p.add_argument("root", type=Path, help="Top-level directory to scan")
    p.add_argument("-o", "--out", type=Path, default=Path("times.csv"),
                   help="CSV file to write (default: times.csv)")
    args = p.parse_args()

    data = collect_times(args.root)
    if not data:
        print(f"⚠️  No simulations found under {args.root}", file=sys.stderr)
        return

    df = pd.DataFrame(data)
    df.to_csv(args.out, index=False)
    print(f"✔️  Wrote {len(df)} records to {args.out}")


if __name__ == "__main__":
    main()

