#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.sasa import ShrakeRupley


TRAJ_PATTERNS = (
    "rep*_md.xtc",
    "md.xtc",
)

TPR_PATTERNS = (
    "rep*_md.tpr",
    "md.tpr",
    "topol.tpr",
    "*.tpr",
)


def natural_key(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(path))
    ]


def find_one(directory, patterns):
    for pattern in patterns:
        files = sorted(directory.glob(pattern), key=natural_key)
        if files:
            return files[0]
    return None


def parse_replica_conf(name):
    m = re.match(r"replica(\d+)_conf(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def mean_ci95(values):
    values = np.asarray(values, dtype=float)

    n = len(values)
    mean = np.mean(values)

    if n < 2:
        return mean, np.nan, np.nan, np.nan, np.nan

    sd = np.std(values, ddof=1)
    sem = sd / np.sqrt(n)

    # Normal approximation is enough for summary;
    # inferential statistics should still use replica-level tests separately.
    ci_low = mean - 1.96 * sem
    ci_high = mean + 1.96 * sem

    return mean, sd, sem, ci_low, ci_high


def analyse_replica(
    replica_dir,
    selection="protein",
    probe_radius=1.4,
    n_sphere_points=960,
    stride=1,
):
    xtc = find_one(replica_dir, TRAJ_PATTERNS)
    tpr = find_one(replica_dir, TPR_PATTERNS)

    if xtc is None:
        raise FileNotFoundError(f"No XTC found in {replica_dir}")

    if tpr is None:
        raise FileNotFoundError(f"No TPR found in {replica_dir}")

    print(f"\n[{replica_dir.name}]")
    print(f"  topology:   {tpr.name}")
    print(f"  trajectory: {xtc.name}")

    u = mda.Universe(str(tpr), str(xtc))

    atoms = u.select_atoms(selection)

    if len(atoms) == 0:
        raise ValueError(
            f"Selection '{selection}' contains zero atoms "
            f"in {replica_dir}"
        )

    # Shrake-Rupley radii are in Å in MDAnalysis.
    sr = ShrakeRupley(
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points,
    )

    sasa_values = []

    for iframe, ts in enumerate(u.trajectory[::stride]):

        # Run SASA for the current frame.
        sr.run(
            atoms,
            start=ts.frame,
            stop=ts.frame + 1,
        )

        # Current MDAnalysis ShrakeRupley stores
        # per-atom SASA in atoms.tempfactors/results depending on version.
        #
        # The robust route is to read results.total_area if present.
        if hasattr(sr.results, "total_area"):
            value = np.asarray(sr.results.total_area).ravel()[0]

        elif hasattr(sr.results, "areas"):
            value = np.sum(sr.results.areas)

        else:
            raise RuntimeError(
                "Cannot locate SASA output in this MDAnalysis version. "
                "Check MDAnalysis.analysis.sasa.ShrakeRupley API."
            )

        sasa_values.append(float(value))

    sasa_values = np.asarray(sasa_values)

    replica, conf = parse_replica_conf(replica_dir.name)

    result = {
        "directory": replica_dir.name,
        "replica": replica,
        "conf": conf,
        "trajectory": xtc.name,
        "topology": tpr.name,
        "selection": selection,
        "n_atoms": len(atoms),
        "n_frames": len(sasa_values),
        "stride": stride,

        # Å²
        "mean_sasa_A2": np.mean(sasa_values),
        "std_sasa_A2": np.std(sasa_values, ddof=1)
        if len(sasa_values) > 1 else np.nan,
        "median_sasa_A2": np.median(sasa_values),
        "min_sasa_A2": np.min(sasa_values),
        "max_sasa_A2": np.max(sasa_values),

        # nm²
        "mean_sasa_nm2": np.mean(sasa_values) / 100.0,
        "std_sasa_nm2": (
            np.std(sasa_values, ddof=1) / 100.0
            if len(sasa_values) > 1 else np.nan
        ),
    }

    print(
        f"  SASA = {result['mean_sasa_A2']:.2f} "
        f"± {result['std_sasa_A2']:.2f} Å²"
    )

    return result


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate mean protein SASA for all replica*_conf* "
            "directories using MDAnalysis."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root directory, e.g. 1LZ7_SNACLEC",
    )

    parser.add_argument(
        "--selection",
        default="protein",
        help="MDAnalysis selection (default: protein)",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Analyse every Nth trajectory frame",
    )

    parser.add_argument(
        "--probe-radius",
        type=float,
        default=1.4,
        help="Probe radius in Å (default: 1.4)",
    )

    parser.add_argument(
        "--sphere-points",
        type=int,
        default=960,
        help="Shrake-Rupley sphere points (default: 960)",
    )

    parser.add_argument(
        "--output",
        default="sasa_ensemble",
        help="Output directory name",
    )

    args = parser.parse_args()

    root = args.root.resolve()

    dirs = sorted(
        [
            p for p in root.iterdir()
            if p.is_dir()
            and re.match(r"replica\d+_conf\d+", p.name)
        ],
        key=natural_key,
    )

    if not dirs:
        raise RuntimeError(
            f"No replica*_conf* directories found in {root}"
        )

    print(f"Root: {root}")
    print(f"Found {len(dirs)} replica directories")

    results = []

    for directory in dirs:
        try:
            result = analyse_replica(
                directory,
                selection=args.selection,
                probe_radius=args.probe_radius,
                n_sphere_points=args.sphere_points,
                stride=args.stride,
            )
            results.append(result)

        except Exception as e:
            print(f"\nERROR in {directory.name}: {e}")

    if not results:
        raise RuntimeError("No replicas were successfully analysed")

    output_dir = root / args.output
    output_dir.mkdir(exist_ok=True)

    write_csv(
        output_dir / "SASA_per_replica.csv",
        results,
    )

    replica_means = np.asarray(
        [x["mean_sasa_A2"] for x in results]
    )

    mean, sd, sem, ci_low, ci_high = mean_ci95(
        replica_means
    )

    print("\n" + "=" * 72)
    print("ENSEMBLE SASA")
    print("=" * 72)

    print(f"Replicas: {len(results)}")
    print(f"Mean SASA: {mean:.3f} Å²")
    print(f"SD between replicas: {sd:.3f} Å²")
    print(f"SEM: {sem:.3f} Å²")
    print(
        f"95% CI of replica means: "
        f"[{ci_low:.3f}, {ci_high:.3f}] Å²"
    )

    print()
    print(f"Mean SASA: {mean / 100:.4f} nm²")
    print(f"SD:        {sd / 100:.4f} nm²")

    summary_file = output_dir / "SASA_summary.txt"

    with open(summary_file, "w") as fh:
        fh.write(f"root = {root}\n")
        fh.write(f"selection = {args.selection}\n")
        fh.write(f"replicas = {len(results)}\n")
        fh.write(f"probe_radius_A = {args.probe_radius}\n")
        fh.write(f"sphere_points = {args.sphere_points}\n")
        fh.write(f"stride = {args.stride}\n\n")

        fh.write(f"mean_SASA_A2 = {mean:.6f}\n")
        fh.write(f"SD_between_replicas_A2 = {sd:.6f}\n")
        fh.write(f"SEM_A2 = {sem:.6f}\n")
        fh.write(f"CI95_low_A2 = {ci_low:.6f}\n")
        fh.write(f"CI95_high_A2 = {ci_high:.6f}\n\n")

        fh.write(f"mean_SASA_nm2 = {mean / 100:.6f}\n")
        fh.write(f"SD_between_replicas_nm2 = {sd / 100:.6f}\n")

    print("\nOutput:")
    print(f"  {output_dir / 'SASA_per_replica.csv'}")
    print(f"  {summary_file}")


if __name__ == "__main__":
    main()