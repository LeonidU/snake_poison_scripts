#!/usr/bin/env python3
"""
Stable inter-protein residue contacts across MD ensembles.

Expected layout, for example:
    1LZ7_SNACLEC/
      replica1_conf1/
        rep1_md.xtc or md.xtc
        rep1_md.tpr / md.tpr / another compatible .tpr
        Protein_chain_A.itp
        Protein_chain_B.itp
      replica2_conf1/
      replica1_conf2/
      ...

A residue-residue contact is present in a frame if ANY pair of heavy atoms,
one from chain A and one from chain B, is within --cutoff A.

Stability is assessed hierarchically:
  1) within a run: occupancy + persistence across contiguous time blocks;
  2) within each starting conformation (conf): reproducibility across replicas;
  3) across starting conformations: reproducibility across confs.

The script uses the two protein ITPs to identify the two GROMACS molecule types.
With a TPR topology, MDAnalysis usually exposes moltype directly. If not, the
script falls back to matching a contiguous atom-name/residue-name block.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance


@dataclass
class ITPInfo:
    path: Path
    moltype: str
    atom_names: np.ndarray
    resnames: np.ndarray
    resnrs: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.atom_names)


@dataclass
class RunResult:
    run_name: str
    replica: str
    conf: str
    trajectory: Path
    topology: Path
    chain_a_name: str
    chain_b_name: str
    a_meta: list[dict]
    b_meta: list[dict]
    occupancy: np.ndarray
    block_fraction: np.ndarray
    min_block_occupancy: np.ndarray
    median_block_occupancy: np.ndarray
    run_stable: np.ndarray
    n_frames: int
    analyzed_ns: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find stable residue contacts between two protein chains across replicas/conformations."
    )
    p.add_argument("root", type=Path, help="Root directory, e.g. 1LZ7_SNACLEC")
    p.add_argument("--run-glob", default="replica*_conf*",
                   help="Glob for run directories relative to root (default: replica*_conf*)")
    p.add_argument("--itp-a", default=None,
                   help="ITP filename/glob for chain A in every run, e.g. 'Protein_chain_A.itp'")
    p.add_argument("--itp-b", default=None,
                   help="ITP filename/glob for chain B in every run, e.g. 'Protein_chain_B.itp'")
    p.add_argument("--cutoff", type=float, default=4.5,
                   help="Heavy-atom contact cutoff in Angstrom (default: 4.5)")
    p.add_argument("--start-ns", type=float, default=0.0,
                   help="Ignore trajectory before this time, ns (default: 0)")
    p.add_argument("--end-ns", type=float, default=None,
                   help="Ignore trajectory after this time, ns")
    p.add_argument("--stride", type=int, default=1,
                   help="Analyze every Nth trajectory frame (default: 1)")
    p.add_argument("--blocks", type=int, default=10,
                   help="Number of contiguous time blocks per run (default: 10)")

    # Hierarchical stability criteria.
    p.add_argument("--run-occupancy", type=float, default=0.50,
                   help="Minimum whole-run occupancy for a run to support a contact (default: 0.50)")
    p.add_argument("--block-occupancy", type=float, default=0.50,
                   help="Minimum occupancy inside a time block for that block to support a contact (default: 0.50)")
    p.add_argument("--min-time-block-fraction", type=float, default=0.80,
                   help="Fraction of time blocks that must support the contact within a run (default: 0.80)")
    p.add_argument("--min-replica-fraction", type=float, default=0.80,
                   help="Fraction of replicas within each conf that must support the contact (default: 0.80)")
    p.add_argument("--min-conf-fraction", type=float, default=1.00,
                   help="Fraction of starting conformations that must support the contact (default: 1.00 = all confs)")

    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: ROOT/stable_contacts_analysis)")
    return p.parse_args()


def parse_itp(path: Path) -> ITPInfo:
    section = None
    moltype = None
    atoms: list[tuple[int, str, str]] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^\[\s*([^\]]+)\s*\]$", line)
            if m:
                section = m.group(1).strip().lower()
                continue

            if section == "moleculetype" and moltype is None:
                fields = line.split()
                if fields:
                    moltype = fields[0]
            elif section == "atoms":
                fields = line.split()
                # nr type resnr residue atom cgnr charge mass ...
                if len(fields) >= 5 and fields[0].isdigit():
                    try:
                        resnr = int(fields[2])
                    except ValueError:
                        continue
                    atoms.append((resnr, fields[3], fields[4]))

    if moltype is None or not atoms:
        raise ValueError(f"{path}: could not parse [ moleculetype ] and [ atoms ]")

    return ITPInfo(
        path=path,
        moltype=moltype,
        atom_names=np.asarray([x[2] for x in atoms], dtype=object),
        resnames=np.asarray([x[1] for x in atoms], dtype=object),
        resnrs=np.asarray([x[0] for x in atoms], dtype=int),
    )


def resolve_one_glob(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"{run_dir}: pattern {pattern!r} matched {len(matches)} files; expected exactly one"
        )
    return matches[0]


def auto_find_itps(run_dir: Path, arg_a: Optional[str], arg_b: Optional[str]) -> tuple[ITPInfo, ITPInfo]:
    if arg_a and arg_b:
        return parse_itp(resolve_one_glob(run_dir, arg_a)), parse_itp(resolve_one_glob(run_dir, arg_b))
    if bool(arg_a) != bool(arg_b):
        raise RuntimeError("Use both --itp-a and --itp-b, or neither")

    infos: list[ITPInfo] = []
    for p in sorted(run_dir.glob("*.itp")):
        low = p.name.lower()
        if "posre" in low or "restraint" in low:
            continue
        try:
            info = parse_itp(p)
        except Exception:
            continue
        # Avoid tiny cofactors/ions if they happen to have their own ITP.
        if info.n_atoms >= 50:
            infos.append(info)

    if len(infos) < 2:
        raise RuntimeError(f"{run_dir}: found fewer than two protein-like ITP files")

    def is_a(info: ITPInfo) -> bool:
        s = (info.path.name + " " + info.moltype).lower()
        return bool(re.search(r"chain[_\- ]*a(?:\.|_|\b)", s))

    def is_b(info: ITPInfo) -> bool:
        s = (info.path.name + " " + info.moltype).lower()
        return bool(re.search(r"chain[_\- ]*b(?:\.|_|\b)", s))

    aa = [x for x in infos if is_a(x)]
    bb = [x for x in infos if is_b(x)]
    if len(aa) == 1 and len(bb) == 1 and aa[0].path != bb[0].path:
        return aa[0], bb[0]

    if len(infos) == 2:
        return infos[0], infos[1]

    names = ", ".join(x.path.name for x in infos)
    raise RuntimeError(
        f"{run_dir}: found multiple candidate ITPs ({names}). "
        "Specify --itp-a and --itp-b explicitly."
    )


def choose_trajectory(run_dir: Path) -> Path:
    preferred: list[Path] = []
    preferred.extend(sorted(run_dir.glob("rep*_md.xtc")))
    if (run_dir / "md.xtc").exists():
        preferred.append(run_dir / "md.xtc")
    preferred.extend(sorted(run_dir.glob("*_md.xtc")))

    # Unique while preserving order.
    seen = set()
    preferred = [p for p in preferred if not (p in seen or seen.add(p))]
    if preferred:
        return preferred[0]

    xtcs = [p for p in sorted(run_dir.glob("*.xtc"))
            if not any(k in p.name.lower() for k in ("nojump", "center", "fit", "prepared", "rmsd"))]
    if len(xtcs) == 1:
        return xtcs[0]
    raise RuntimeError(f"{run_dir}: cannot choose trajectory; found {[p.name for p in xtcs]}")


def choose_topology(run_dir: Path, xtc: Path) -> Path:
    same_stem = xtc.with_suffix(".tpr")
    if same_stem.exists():
        return same_stem

    candidates: list[Path] = []
    candidates.extend(sorted(run_dir.glob("rep*_md.tpr")))
    for name in ("md.tpr", "topol.tpr"):
        p = run_dir / name
        if p.exists():
            candidates.append(p)
    candidates.extend(sorted(run_dir.glob("*.tpr")))

    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]
    if candidates:
        return candidates[0]

    # Last-resort topology; moltype may not be available, but atom-block matching can work.
    gros = sorted(run_dir.glob("*.gro"))
    if gros:
        return gros[-1]

    raise RuntimeError(f"{run_dir}: no compatible .tpr or .gro topology found")


def atomgroup_from_itp(u: mda.Universe, info: ITPInfo):
    # Preferred: TPR provides the GROMACS molecule type for every atom.
    try:
        moltypes = np.asarray(u.atoms.moltypes, dtype=object)
        mask = moltypes == info.moltype
        idx = np.flatnonzero(mask)
        if len(idx) == info.n_atoms:
            return u.atoms[idx]
        if len(idx) > 0:
            raise RuntimeError(
                f"moltype {info.moltype!r}: topology contains {len(idx)} atoms, "
                f"ITP contains {info.n_atoms}; expected one molecule instance"
            )
    except (AttributeError, mda.exceptions.NoDataError):
        pass

    # Fallback: find one contiguous atom block matching atom names + residue names.
    names = np.asarray(u.atoms.names, dtype=object)
    resnames = np.asarray(u.atoms.resnames, dtype=object)
    n = info.n_atoms
    candidate_starts = np.flatnonzero((names == info.atom_names[0]) & (resnames == info.resnames[0]))
    matches = []
    for s in candidate_starts:
        e = s + n
        if e > len(names):
            continue
        if np.array_equal(names[s:e], info.atom_names) and np.array_equal(resnames[s:e], info.resnames):
            matches.append((s, e))
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely map {info.path.name} ({info.moltype}) into topology: "
            f"found {len(matches)} contiguous matches"
        )
    s, e = matches[0]
    return u.atoms[s:e]


def heavy_atoms(ag):
    names = np.asarray(ag.names, dtype=str)
    stripped = np.char.lstrip(np.char.upper(names), "0123456789")
    mask = np.array([not x.startswith("H") for x in stripped], dtype=bool)
    return ag[mask]


def local_residue_mapping(ag):
    """Return per-atom local residue ordinal and residue metadata in chain order."""
    global_resindices = np.asarray(ag.resindices, dtype=int)
    mapping: dict[int, int] = {}
    ordered: list[int] = []
    atom_to_local = np.empty(len(ag), dtype=int)
    for i, gri in enumerate(global_resindices):
        gri = int(gri)
        if gri not in mapping:
            mapping[gri] = len(ordered)
            ordered.append(gri)
        atom_to_local[i] = mapping[gri]

    meta = []
    for pos, gri in enumerate(ordered, start=1):
        r = ag.universe.residues[gri]
        meta.append({
            "pos": pos,
            "resid": int(r.resid),
            "resname": str(r.resname),
            "label": f"{r.resname}{int(r.resid)}",
        })
    return atom_to_local, meta


def parse_run_name(name: str) -> tuple[str, str]:
    m = re.search(r"replica(\d+)_conf(\d+)", name, flags=re.I)
    if m:
        return f"replica{m.group(1)}", f"conf{m.group(2)}"
    m = re.search(r"rep(?:lica)?[_-]?(\d+)", name, flags=re.I)
    replica = f"replica{m.group(1)}" if m else name
    m2 = re.search(r"conf[_-]?(\d+)", name, flags=re.I)
    conf = f"conf{m2.group(1)}" if m2 else "conf1"
    return replica, conf


def valid_box(ts):
    box = ts.dimensions
    if box is None:
        return None
    arr = np.asarray(box, dtype=float)
    if len(arr) < 3 or np.any(arr[:3] <= 0):
        return None
    return arr


def analyze_run(run_dir: Path, args: argparse.Namespace, out_dir: Path) -> RunResult:
    xtc = choose_trajectory(run_dir)
    top = choose_topology(run_dir, xtc)
    itp_a, itp_b = auto_find_itps(run_dir, args.itp_a, args.itp_b)

    print(f"\n[{run_dir.name}]", file=sys.stderr)
    print(f"  trajectory: {xtc.name}", file=sys.stderr)
    print(f"  topology:   {top.name}", file=sys.stderr)
    print(f"  chain A:    {itp_a.path.name} / {itp_a.moltype}", file=sys.stderr)
    print(f"  chain B:    {itp_b.path.name} / {itp_b.moltype}", file=sys.stderr)

    u = mda.Universe(str(top), str(xtc))
    ag_a_full = atomgroup_from_itp(u, itp_a)
    ag_b_full = atomgroup_from_itp(u, itp_b)

    ag_a = heavy_atoms(ag_a_full)
    ag_b = heavy_atoms(ag_b_full)
    if len(ag_a) == 0 or len(ag_b) == 0:
        raise RuntimeError(f"{run_dir}: one chain has zero heavy atoms")

    a_atom_to_res, a_meta = local_residue_mapping(ag_a)
    b_atom_to_res, b_meta = local_residue_mapping(ag_b)
    n_a, n_b = len(a_meta), len(b_meta)
    n_pairs = n_a * n_b

    selected_frames: list[int] = []
    selected_times_ns: list[float] = []
    for iframe in range(0, len(u.trajectory), args.stride):
        ts = u.trajectory[iframe]
        tns = float(ts.time) / 1000.0  # MDAnalysis/GROMACS time is ps
        if tns < args.start_ns:
            continue
        if args.end_ns is not None and tns > args.end_ns:
            continue
        selected_frames.append(iframe)
        selected_times_ns.append(tns)

    if not selected_frames:
        raise RuntimeError(f"{run_dir}: no trajectory frames after time filtering")

    n_blocks = min(args.blocks, len(selected_frames))
    counts = np.zeros(n_pairs, dtype=np.int64)
    block_counts = np.zeros((n_blocks, n_pairs), dtype=np.int64)
    block_nframes = np.zeros(n_blocks, dtype=np.int64)

    cutoff = float(args.cutoff)

    for k, iframe in enumerate(selected_frames):
        ts = u.trajectory[iframe]
        block = min((k * n_blocks) // len(selected_frames), n_blocks - 1)
        block_nframes[block] += 1

        atom_pairs = capped_distance(
            ag_a.positions,
            ag_b.positions,
            max_cutoff=cutoff,
            box=valid_box(ts),
            return_distances=False,
        )
        if len(atom_pairs) == 0:
            continue

        ra = a_atom_to_res[atom_pairs[:, 0]]
        rb = b_atom_to_res[atom_pairs[:, 1]]
        flat = np.unique(ra * n_b + rb)  # each residue pair counts once per frame
        counts[flat] += 1
        block_counts[block, flat] += 1

    n_frames = len(selected_frames)
    occupancy = counts / n_frames
    block_occ = block_counts / block_nframes[:, None]
    block_fraction = np.mean(block_occ >= args.block_occupancy, axis=0)
    min_block_occ = np.min(block_occ, axis=0)
    median_block_occ = np.median(block_occ, axis=0)

    run_stable = (
        (occupancy >= args.run_occupancy)
        & (block_fraction >= args.min_time_block_fraction)
    )

    replica, conf = parse_run_name(run_dir.name)
    analyzed_ns = max(selected_times_ns) - min(selected_times_ns) if n_frames > 1 else 0.0

    # Per-run output, only pairs observed at least once.
    observed = np.flatnonzero(counts > 0)
    rows = []
    for flat in observed:
        ia, ib = divmod(int(flat), n_b)
        rows.append({
            "run": run_dir.name,
            "replica": replica,
            "conf": conf,
            "A_pos": a_meta[ia]["pos"],
            "A_resid": a_meta[ia]["resid"],
            "A_resname": a_meta[ia]["resname"],
            "A_label": a_meta[ia]["label"],
            "B_pos": b_meta[ib]["pos"],
            "B_resid": b_meta[ib]["resid"],
            "B_resname": b_meta[ib]["resname"],
            "B_label": b_meta[ib]["label"],
            "occupancy": occupancy[flat],
            "time_block_fraction": block_fraction[flat],
            "min_block_occupancy": min_block_occ[flat],
            "median_block_occupancy": median_block_occ[flat],
            "run_stable": bool(run_stable[flat]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["run_stable", "occupancy", "time_block_fraction"],
            ascending=[False, False, False],
        )
    df.to_csv(out_dir / f"{run_dir.name}_contacts.csv", index=False)

    print(
        f"  heavy atoms A/B: {len(ag_a)}/{len(ag_b)}; residues A/B: {n_a}/{n_b}; "
        f"frames: {n_frames}; stable in this run: {int(run_stable.sum())}",
        file=sys.stderr,
    )

    return RunResult(
        run_name=run_dir.name,
        replica=replica,
        conf=conf,
        trajectory=xtc,
        topology=top,
        chain_a_name=itp_a.moltype,
        chain_b_name=itp_b.moltype,
        a_meta=a_meta,
        b_meta=b_meta,
        occupancy=occupancy,
        block_fraction=block_fraction,
        min_block_occupancy=min_block_occ,
        median_block_occupancy=median_block_occ,
        run_stable=run_stable,
        n_frames=n_frames,
        analyzed_ns=analyzed_ns,
    )


def assert_same_sequences(results: list[RunResult]):
    a0 = [x["resname"] for x in results[0].a_meta]
    b0 = [x["resname"] for x in results[0].b_meta]
    for r in results[1:]:
        a = [x["resname"] for x in r.a_meta]
        b = [x["resname"] for x in r.b_meta]
        if a != a0 or b != b0:
            raise RuntimeError(
                f"Residue sequence/order differs in {r.run_name}. "
                "Analyze systems with consistent chain definitions together."
            )


def aggregate(results: list[RunResult], args: argparse.Namespace, out_dir: Path):
    assert_same_sequences(results)

    n_runs = len(results)
    n_a = len(results[0].a_meta)
    n_b = len(results[0].b_meta)
    n_pairs = n_a * n_b

    occ = np.vstack([r.occupancy for r in results])
    block_frac = np.vstack([r.block_fraction for r in results])
    run_stable = np.vstack([r.run_stable for r in results])

    confs = sorted({r.conf for r in results}, key=lambda x: (int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 10**9, x))
    conf_masks = {c: np.array([r.conf == c for r in results], dtype=bool) for c in confs}

    conf_rows = []
    conf_stable_matrix = np.zeros((len(confs), n_pairs), dtype=bool)
    conf_replica_fraction = np.zeros((len(confs), n_pairs), dtype=float)

    for ci, conf in enumerate(confs):
        mask = conf_masks[conf]
        frac = np.mean(run_stable[mask], axis=0)
        conf_replica_fraction[ci] = frac
        cstable = frac >= args.min_replica_fraction
        conf_stable_matrix[ci] = cstable

        observed = np.flatnonzero(np.max(occ[mask], axis=0) > 0)
        for flat in observed:
            ia, ib = divmod(int(flat), n_b)
            conf_rows.append({
                "conf": conf,
                "n_replicas": int(mask.sum()),
                "A_pos": results[0].a_meta[ia]["pos"],
                "A_resid": results[0].a_meta[ia]["resid"],
                "A_resname": results[0].a_meta[ia]["resname"],
                "A_label": results[0].a_meta[ia]["label"],
                "B_pos": results[0].b_meta[ib]["pos"],
                "B_resid": results[0].b_meta[ib]["resid"],
                "B_resname": results[0].b_meta[ib]["resname"],
                "B_label": results[0].b_meta[ib]["label"],
                "replica_stable_fraction": frac[flat],
                "median_replica_occupancy": float(np.median(occ[mask, flat])),
                "mean_replica_occupancy": float(np.mean(occ[mask, flat])),
                "mean_time_block_fraction": float(np.mean(block_frac[mask, flat])),
                "conf_stable": bool(cstable[flat]),
            })

    conf_df = pd.DataFrame(conf_rows)
    if not conf_df.empty:
        conf_df = conf_df.sort_values(
            ["conf_stable", "replica_stable_fraction", "median_replica_occupancy"],
            ascending=[False, False, False],
        )
    conf_df.to_csv(out_dir / "contacts_by_conformation.csv", index=False)

    conf_fraction = np.mean(conf_stable_matrix, axis=0)
    core = conf_fraction >= args.min_conf_fraction

    observed = np.flatnonzero(np.max(occ, axis=0) > 0)
    rows = []
    for flat in observed:
        ia, ib = divmod(int(flat), n_b)
        row = {
            "A_pos": results[0].a_meta[ia]["pos"],
            "A_resid": results[0].a_meta[ia]["resid"],
            "A_resname": results[0].a_meta[ia]["resname"],
            "A_label": results[0].a_meta[ia]["label"],
            "B_pos": results[0].b_meta[ib]["pos"],
            "B_resid": results[0].b_meta[ib]["resid"],
            "B_resname": results[0].b_meta[ib]["resname"],
            "B_label": results[0].b_meta[ib]["label"],
            "n_runs": n_runs,
            "n_confs": len(confs),
            "mean_occupancy": float(np.mean(occ[:, flat])),
            "median_occupancy": float(np.median(occ[:, flat])),
            "min_occupancy": float(np.min(occ[:, flat])),
            "max_occupancy": float(np.max(occ[:, flat])),
            "stable_run_fraction": float(np.mean(run_stable[:, flat])),
            "mean_time_block_fraction": float(np.mean(block_frac[:, flat])),
            "min_time_block_fraction": float(np.min(block_frac[:, flat])),
            "stable_conf_fraction": float(conf_fraction[flat]),
            "core_stable_contact": bool(core[flat]),
        }
        for ci, conf in enumerate(confs):
            row[f"{conf}_stable_replica_fraction"] = float(conf_replica_fraction[ci, flat])
            row[f"{conf}_stable"] = bool(conf_stable_matrix[ci, flat])
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["core_stable_contact", "stable_conf_fraction", "stable_run_fraction", "median_occupancy"],
            ascending=[False, False, False, False],
        )
    summary.to_csv(out_dir / "contacts_summary.csv", index=False)

    stable = summary[summary["core_stable_contact"]].copy() if not summary.empty else summary.copy()
    stable.to_csv(out_dir / "stable_contacts.csv", index=False)

    # Run manifest for reproducibility.
    manifest = pd.DataFrame([
        {
            "run": r.run_name,
            "replica": r.replica,
            "conf": r.conf,
            "trajectory": str(r.trajectory),
            "topology": str(r.topology),
            "chain_A_moltype": r.chain_a_name,
            "chain_B_moltype": r.chain_b_name,
            "frames_analyzed": r.n_frames,
            "analyzed_span_ns": r.analyzed_ns,
        }
        for r in results
    ])
    manifest.to_csv(out_dir / "runs_manifest.csv", index=False)

    print("\n=== Ensemble summary ===", file=sys.stderr)
    print(f"runs: {n_runs}", file=sys.stderr)
    print(f"conformations: {', '.join(confs)}", file=sys.stderr)
    print(f"observed residue pairs: {len(summary)}", file=sys.stderr)
    print(f"core stable contacts: {len(stable)}", file=sys.stderr)
    print(f"output: {out_dir}", file=sys.stderr)


def main():
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")
    if args.stride < 1 or args.blocks < 1:
        raise SystemExit("--stride and --blocks must be >= 1")
    for name in (
        "run_occupancy", "block_occupancy", "min_time_block_fraction",
        "min_replica_fraction", "min_conf_fraction",
    ):
        x = getattr(args, name)
        if not (0.0 <= x <= 1.0):
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")

    out_dir = (args.output or (root / "stable_contacts_analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in root.glob(args.run_glob) if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run directories matched {args.run_glob!r} under {root}")

    results = []
    failures = []
    for run_dir in run_dirs:
        try:
            results.append(analyze_run(run_dir, args, out_dir))
        except Exception as exc:
            failures.append((run_dir, exc))
            print(f"ERROR [{run_dir.name}]: {exc}", file=sys.stderr)

    if failures:
        print("\nFailed runs:", file=sys.stderr)
        for d, exc in failures:
            print(f"  {d}: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if not results:
        raise SystemExit("No runs were successfully analyzed")

    aggregate(results, args, out_dir)


if __name__ == "__main__":
    main()
