#!/usr/bin/env python3

"""
Calculate solvent-accessible surface area (SASA) for an ensemble of
MD replicas arranged as e.g.:

ROOT/
  replica1_conf1/
    rep1_md.xtc   (or md.xtc)
    rep1_md.tpr   (or md.tpr/topol.tpr)
  replica2_conf1/
    ...
  replica1_conf2/
    ...

Requirements:
    pip install MDAnalysis==2.10.0 mdakit-sasa

The script calculates SASA independently for each trajectory and then
aggregates the replica-level means. Frames are NOT pooled across replicas,
so each independent MD replica has equal statistical weight. Independent
replica/conf directories can be analyzed in parallel with --workers N.

Outputs (ROOT/sasa_ensemble by default):
    SASA_per_replica.csv
    SASA_skipped_replicas.csv   (only if any replica is skipped)
    SASA_by_conformation.csv
    SASA_summary.txt

SASA values reported by mdakit-sasa/FreeSASA are in Angstrom^2.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
from scipy.stats import t as student_t


RUN_RE = re.compile(r"^replica(?P<replica>\d+)_conf(?P<conf>\d+)$", re.IGNORECASE)

# Preferred trajectory names, in order.
TRAJ_PATTERNS = (
    "rep*_md.xtc",
    "md.xtc",
    "*.xtc",
)

# Fallback topology names, in order. A topology with coordinates is not
# required for XTC if a TPR is available, so TPR is strongly preferred.
TOP_PATTERNS = (
    "rep*_md.tpr",
    "md.tpr",
    "topol.tpr",
    "*.tpr",
    "*.gro",
    "*.pdb",
)


class NonFiniteSASAError(RuntimeError):
    """Raised when a replica contains NaN or infinite SASA values."""
    pass


def natural_key(value: str | Path):
    """Natural sort key: replica2 before replica10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def unique_sorted(paths):
    return sorted(set(paths), key=natural_key)


def find_trajectory(run_dir: Path) -> Path:
    """Find the preferred XTC trajectory in a replica directory."""
    for pattern in TRAJ_PATTERNS:
        matches = unique_sorted(run_dir.glob(pattern))
        if matches:
            if len(matches) > 1:
                print(
                    f"WARNING: multiple trajectories match '{pattern}' in {run_dir}; "
                    f"using {matches[0].name}",
                    file=sys.stderr,
                )
            return matches[0]
    raise FileNotFoundError(f"No XTC trajectory found in {run_dir}")


def find_topology(run_dir: Path, trajectory: Path) -> Path:
    """Find topology, preferring a TPR with the same stem as the XTC."""
    same_stem_tpr = trajectory.with_suffix(".tpr")
    if same_stem_tpr.exists():
        return same_stem_tpr

    for pattern in TOP_PATTERNS:
        matches = unique_sorted(run_dir.glob(pattern))
        if matches:
            if len(matches) > 1:
                print(
                    f"WARNING: multiple topology files match '{pattern}' in {run_dir}; "
                    f"using {matches[0].name}",
                    file=sys.stderr,
                )
            return matches[0]

    raise FileNotFoundError(
        f"No TPR/GRO/PDB topology found in {run_dir}. "
        "A .cpt or .itp file alone is not sufficient for this analysis."
    )


def parse_run_name(run_dir: Path) -> tuple[int, int]:
    match = RUN_RE.match(run_dir.name)
    if not match:
        raise ValueError(f"Unexpected run directory name: {run_dir.name}")
    return int(match.group("replica")), int(match.group("conf"))


def summarize(values: np.ndarray) -> dict[str, float]:
    """Summary where each value is one independent replica mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean = float(np.mean(values))
    median = float(np.median(values))

    if n >= 2:
        sd = float(np.std(values, ddof=1))
        sem = sd / np.sqrt(n)
        tcrit = float(student_t.ppf(0.975, df=n - 1))
        ci_low = mean - tcrit * sem
        ci_high = mean + tcrit * sem
    else:
        sd = sem = ci_low = ci_high = float("nan")

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "sd": sd,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def analyze_run(
    run_dir: Path,
    selection: str,
    stride: int,
) -> dict:
    replica, conf = parse_run_name(run_dir)
    trajectory = find_trajectory(run_dir)
    topology = find_topology(run_dir, trajectory)

    print(f"\n[{run_dir.name}]")
    print(f"  topology   : {topology.name}")
    print(f"  trajectory : {trajectory.name}")
    print(f"  selection  : {selection}")
    print(f"  stride     : {stride}")

    universe = mda.Universe(str(topology), str(trajectory))
    atoms = universe.select_atoms(selection)

    if len(atoms) == 0:
        raise ValueError(
            f"Selection '{selection}' contains zero atoms in {run_dir}"
        )

    analysis = SASAAnalysis(universe, select=selection)
    analysis.run(step=stride)

    sasa = np.asarray(analysis.results.total_area, dtype=float).reshape(-1)

    if sasa.size == 0:
        raise RuntimeError(f"No SASA values were produced for {run_dir}")
    if not np.all(np.isfinite(sasa)):
        bad = np.count_nonzero(~np.isfinite(sasa))
        raise NonFiniteSASAError(
            f"Non-finite SASA values detected for {run_dir} "
            f"({bad}/{sasa.size} analyzed frames)"
        )

    row = {
        "directory": run_dir.name,
        "replica": replica,
        "conf": conf,
        "topology": topology.name,
        "trajectory": trajectory.name,
        "selection": selection,
        "selected_atoms": len(atoms),
        "analyzed_frames": int(sasa.size),
        "stride": stride,
        "mean_sasa_A2": float(np.mean(sasa)),
        "sd_frames_A2": float(np.std(sasa, ddof=1)) if sasa.size > 1 else float("nan"),
        "median_sasa_A2": float(np.median(sasa)),
        "min_sasa_A2": float(np.min(sasa)),
        "max_sasa_A2": float(np.max(sasa)),
        "mean_sasa_nm2": float(np.mean(sasa) / 100.0),
        "sd_frames_nm2": float(np.std(sasa, ddof=1) / 100.0) if sasa.size > 1 else float("nan"),
    }

    print(
        f"  mean SASA  : {row['mean_sasa_A2']:.3f} A^2 "
        f"({row['mean_sasa_nm2']:.3f} nm^2)"
    )
    print(f"  frames      : {row['analyzed_frames']}")

    return row


def _worker_analyze_run(run_dir_str: str, selection: str, stride: int) -> dict:
    """Top-level multiprocessing worker (must remain picklable)."""
    return analyze_run(
        run_dir=Path(run_dir_str),
        selection=selection,
        stride=stride,
    )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate SASA for all replicaN_confM directories using "
            "MDAnalysis + mdakit-sasa."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing replica*_conf* subdirectories",
    )
    parser.add_argument(
        "--selection",
        default="protein",
        help="MDAnalysis atom selection (default: protein)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Analyze every Nth trajectory frame (default: 1)",
    )
    parser.add_argument(
        "--output",
        default="sasa_ensemble",
        help="Output directory name inside ROOT (default: sasa_ensemble)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of replica directories to analyze in parallel using "
            "separate processes (default: 1). For example: --workers 8"
        ),
    )
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Root directory does not exist: {root}")

    run_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and RUN_RE.match(path.name)
    ]
    run_dirs = sorted(run_dirs, key=natural_key)

    if not run_dirs:
        raise RuntimeError(
            f"No directories matching replicaN_confM were found in {root}"
        )

    print(f"Root: {root}")
    print(f"Found {len(run_dirs)} replica/conf directories")

    rows = []
    skipped_rows = []

    if args.workers == 1:
        # Serial mode; useful for debugging and for storage with poor parallel I/O.
        for run_dir in run_dirs:
            try:
                rows.append(
                    analyze_run(
                        run_dir=run_dir,
                        selection=args.selection,
                        stride=args.stride,
                    )
                )
            except NonFiniteSASAError as exc:
                print(
                    f"WARNING: {exc}. Replica {run_dir.name} will be ignored.",
                    file=sys.stderr,
                    flush=True,
                )
                replica, conf = parse_run_name(run_dir)
                skipped_rows.append(
                    {
                        "directory": run_dir.name,
                        "replica": replica,
                        "conf": conf,
                        "reason": str(exc),
                    }
                )
    else:
        n_workers = min(args.workers, len(run_dirs))
        print(f"Parallel mode: {n_workers} worker processes")
        print(
            "Each worker analyzes one complete replica/conf trajectory. "
            "This avoids pooling frames and preserves replica-level statistics."
        )

        # Parallelize across independent replica directories. This is more robust
        # than attempting to share one MDAnalysis Universe/trajectory reader
        # between processes.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_run = {
                executor.submit(
                    _worker_analyze_run,
                    str(run_dir),
                    args.selection,
                    args.stride,
                ): run_dir
                for run_dir in run_dirs
            }

            completed = 0
            for future in as_completed(future_to_run):
                run_dir = future_to_run[future]
                try:
                    row = future.result()
                except NonFiniteSASAError as exc:
                    print(
                        f"WARNING: {exc}. Replica {run_dir.name} will be ignored.",
                        file=sys.stderr,
                        flush=True,
                    )
                    replica, conf = parse_run_name(run_dir)
                    skipped_rows.append(
                        {
                            "directory": run_dir.name,
                            "replica": replica,
                            "conf": conf,
                            "reason": str(exc),
                        }
                    )
                    completed += 1
                    print(
                        f"Completed {completed}/{len(run_dirs)}: {run_dir.name} SKIPPED",
                        flush=True,
                    )
                    continue
                except Exception as exc:
                    # Other failures are still treated as fatal because they can
                    # indicate a broken topology/trajectory or another real problem.
                    for other in future_to_run:
                        other.cancel()
                    raise RuntimeError(
                        f"SASA analysis failed for {run_dir}: {exc}"
                    ) from exc

                rows.append(row)
                completed += 1
                print(
                    f"Completed {completed}/{len(run_dirs)}: {run_dir.name} "
                    f"mean SASA = {row['mean_sasa_A2']:.3f} A^2",
                    flush=True,
                )

        # as_completed() returns jobs in completion order; restore deterministic
        # replica/conf ordering in CSV files and summaries.
        rows.sort(key=lambda row: (row["conf"], row["replica"], row["directory"]))

    if not rows:
        raise RuntimeError(
            "No replicas with finite SASA values were available for the ensemble summary."
        )

    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    per_replica_csv = output_dir / "SASA_per_replica.csv"
    write_csv(per_replica_csv, rows)

    skipped_csv = output_dir / "SASA_skipped_replicas.csv"
    if skipped_rows:
        skipped_rows.sort(key=lambda row: (row["conf"], row["replica"], row["directory"]))
        write_csv(skipped_csv, skipped_rows)

    # ------------------------------------------------------------------
    # Per-conformation summaries: each MD replica contributes one value.
    # ------------------------------------------------------------------
    by_conf = defaultdict(list)
    for row in rows:
        by_conf[row["conf"]].append(row["mean_sasa_A2"])

    conf_rows = []
    for conf in sorted(by_conf):
        stats = summarize(np.asarray(by_conf[conf], dtype=float))
        conf_rows.append(
            {
                "conf": conf,
                "n_replicas": stats["n"],
                "mean_sasa_A2": stats["mean"],
                "sd_between_replicas_A2": stats["sd"],
                "sem_A2": stats["sem"],
                "ci95_low_A2": stats["ci95_low"],
                "ci95_high_A2": stats["ci95_high"],
                "mean_sasa_nm2": stats["mean"] / 100.0,
                "sd_between_replicas_nm2": stats["sd"] / 100.0,
            }
        )

    per_conf_csv = output_dir / "SASA_by_conformation.csv"
    write_csv(per_conf_csv, conf_rows)

    # ------------------------------------------------------------------
    # Overall summary.
    # Primary summary = mean of replica means, not mean of pooled frames.
    # ------------------------------------------------------------------
    replica_means = np.asarray([row["mean_sasa_A2"] for row in rows], dtype=float)
    overall = summarize(replica_means)

    # Equal-conformation mean is also reported. This differs from the overall
    # replica mean only if different conformations have different numbers of
    # successfully analyzed replicas.
    conf_means = np.asarray([row["mean_sasa_A2"] for row in conf_rows], dtype=float)
    equal_conf_mean = float(np.mean(conf_means))

    print("\n" + "=" * 78)
    print("ENSEMBLE SASA SUMMARY")
    print("=" * 78)
    print(f"Replicas analyzed       : {overall['n']}")
    print(f"Replicas skipped        : {len(skipped_rows)}")
    print(f"Starting conformations  : {len(conf_rows)}")
    print(f"Mean of replica means   : {overall['mean']:.3f} A^2")
    print(f"SD between replicas     : {overall['sd']:.3f} A^2")
    print(f"SEM                     : {overall['sem']:.3f} A^2")
    print(
        f"95% CI                  : [{overall['ci95_low']:.3f}, "
        f"{overall['ci95_high']:.3f}] A^2"
    )
    print(f"Mean                    : {overall['mean'] / 100.0:.3f} nm^2")
    print(f"SD                      : {overall['sd'] / 100.0:.3f} nm^2")
    print(f"Equal-weight conf mean  : {equal_conf_mean:.3f} A^2")

    summary_path = output_dir / "SASA_summary.txt"
    with summary_path.open("w") as handle:
        handle.write(f"root = {root}\n")
        handle.write(f"selection = {args.selection}\n")
        handle.write(f"stride = {args.stride}\n")
        handle.write(f"workers = {args.workers}\n")
        handle.write(f"replicas = {overall['n']}\n")
        handle.write(f"replicas_skipped_nonfinite = {len(skipped_rows)}\n")
        handle.write(f"conformations = {len(conf_rows)}\n\n")

        handle.write("[Replica-level ensemble summary]\n")
        handle.write(f"mean_SASA_A2 = {format_float(overall['mean'])}\n")
        handle.write(f"SD_between_replicas_A2 = {format_float(overall['sd'])}\n")
        handle.write(f"SEM_A2 = {format_float(overall['sem'])}\n")
        handle.write(f"CI95_low_A2 = {format_float(overall['ci95_low'])}\n")
        handle.write(f"CI95_high_A2 = {format_float(overall['ci95_high'])}\n")
        handle.write(f"mean_SASA_nm2 = {format_float(overall['mean'] / 100.0)}\n")
        handle.write(f"SD_between_replicas_nm2 = {format_float(overall['sd'] / 100.0)}\n\n")

        handle.write("[Equal-weight starting-conformation summary]\n")
        handle.write(f"mean_of_conf_means_A2 = {format_float(equal_conf_mean)}\n")
        handle.write(f"mean_of_conf_means_nm2 = {format_float(equal_conf_mean / 100.0)}\n")

    print("\nOutput:")
    print(f"  {per_replica_csv}")
    if skipped_rows:
        print(f"  {skipped_csv}")
    print(f"  {per_conf_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
