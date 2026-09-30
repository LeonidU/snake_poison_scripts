#!/usr/bin/env python3
"""
RMSD plateau analysis for an MD trajectory.

Workflow
--------
1. Calculate RMSD from an XTC trajectory with MDAnalysis.
2. Detect the start of the statistically stationary/equilibrated RMSD region
   with pymbar.timeseries.detect_equilibration().
3. Estimate residual RMSD drift after t0 with linear regression and
   autocorrelation-robust (HAC/Newey-West) standard errors.
4. Test whether the slope is practically equivalent to zero using TOST.
5. As a sensitivity analysis, estimate a moving-block-bootstrap CI for slope.

Important
---------
An XTC file does not contain atom names/topology. Supply --top, or place an
unambiguous .tpr/.gro/.pdb topology next to the XTC.

RMSD stability is evidence for stability of this observable, not proof of full
thermodynamic convergence of the entire simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.transformations import unwrap
from pymbar import timeseries
import statsmodels.api as sm
from scipy.stats import norm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect and statistically test an RMSD plateau in an XTC trajectory."
    )
    p.add_argument("xtc", type=Path, help="Production trajectory (.xtc)")
    p.add_argument(
        "--top", type=Path, default=None,
        help="Topology/structure readable by MDAnalysis (.tpr, .gro, .pdb). "
             "If omitted, the script tries to find one next to the XTC."
    )
    p.add_argument(
        "--selection", default="protein and backbone",
        help='MDAnalysis selection used for fitting and RMSD (default: "protein and backbone").'
    )
    p.add_argument(
        "--ref-frame", type=int, default=0,
        help="Reference frame index in the original trajectory (default: 0)."
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Analyze every N-th trajectory frame (default: 1)."
    )
    p.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="Number of multiprocessing workers for RMSD calculation "
             "(default: min(8, available CPUs)). Use 1 for serial execution."
    )
    p.add_argument(
        "--parts", type=int, default=0,
        help="Number of trajectory chunks for parallel RMSD. "
             "0 = automatic (2 x workers)."
    )
    p.add_argument(
        "--no-unwrap", action="store_true",
        help="Do not try to unwrap the protein using bond information."
    )
    p.add_argument(
        "--nskip", type=int, default=0,
        help="Candidate t0 spacing for PyMBAR. 0 = automatic (~1000 candidate origins)."
    )
    p.add_argument(
        "--equiv-drift-A", type=float, default=0.10,
        help="Maximum practically negligible RMSD change, in Angstrom (default: 0.10)."
    )
    p.add_argument(
        "--equiv-window-ns", type=float, default=100.0,
        help="Time window corresponding to --equiv-drift-A, in ns (default: 100)."
    )
    p.add_argument(
        "--alpha", type=float, default=0.05,
        help="One-sided alpha for TOST (default: 0.05; corresponds to a 90%% CI)."
    )
    p.add_argument(
        "--hac-lags", type=int, default=0,
        help="HAC/Newey-West maximum lag in analyzed frames. 0 = estimate from residual autocorrelation."
    )
    p.add_argument(
        "--bootstrap", type=int, default=5000,
        help="Number of moving-block-bootstrap replicates (default: 5000; 0 disables)."
    )
    p.add_argument(
        "--seed", type=int, default=20260929,
        help="Random seed for bootstrap (default: 20260929)."
    )
    p.add_argument(
        "--prefix", type=Path, default=None,
        help="Output prefix (default: XTC path without extension)."
    )
    return p.parse_args()


def discover_topology(xtc: Path) -> Path:
    """Find a plausible topology next to the XTC, but refuse ambiguous choices."""
    direct = [xtc.with_suffix(ext) for ext in (".tpr", ".gro", ".pdb")]
    direct_existing = [p for p in direct if p.exists()]
    if len(direct_existing) == 1:
        return direct_existing[0]
    if len(direct_existing) > 1:
        raise RuntimeError(
            "Several same-stem topology files were found: "
            + ", ".join(map(str, direct_existing))
            + ". Please choose one with --top."
        )

    preferred_names = ["topol.tpr", "md.tpr", "production.tpr", "prod.tpr"]
    preferred = [xtc.parent / name for name in preferred_names if (xtc.parent / name).exists()]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError(
            "Several plausible topology files were found: "
            + ", ".join(map(str, preferred))
            + ". Please choose one with --top."
        )

    all_candidates = []
    for ext in ("*.tpr", "*.gro", "*.pdb"):
        all_candidates.extend(sorted(xtc.parent.glob(ext)))
    if len(all_candidates) == 1:
        return all_candidates[0]

    if not all_candidates:
        raise RuntimeError(
            "No .tpr/.gro/.pdb topology was found next to the XTC. Supply it with --top."
        )
    raise RuntimeError(
        "Topology auto-detection is ambiguous. Candidates: "
        + ", ".join(map(str, all_candidates))
        + ". Supply the desired file with --top."
    )


def add_unwrap_if_possible(u: mda.Universe, label: str) -> bool:
    """Unwrap bonded protein fragments; return True if transformation was added."""
    protein = u.select_atoms("protein")
    if len(protein) == 0:
        warnings.warn(f"[{label}] No atoms matched 'protein'; PBC unwrapping skipped.")
        return False
    try:
        u.trajectory.add_transformations(unwrap(protein))
        return True
    except Exception as exc:
        warnings.warn(
            f"[{label}] Could not enable automatic protein unwrapping ({exc}). "
            "For a PBC-broken complex, preprocess the trajectory with GROMACS before analysis."
        )
        return False


def calculate_rmsd(
    topology: Path,
    xtc: Path,
    selection: str,
    ref_frame: int,
    stride: int,
    do_unwrap: bool,
    workers: int,
    parts: int,
) -> tuple[pd.DataFrame, int, bool]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if parts < 0:
        raise ValueError("--parts must be >= 0")

    u = mda.Universe(str(topology), str(xtc))
    ref = mda.Universe(str(topology), str(xtc))

    if not (0 <= ref_frame < len(u.trajectory)):
        raise ValueError(
            f"--ref-frame={ref_frame} is outside trajectory range 0..{len(u.trajectory)-1}."
        )

    sel = u.select_atoms(selection)
    if len(sel) == 0:
        raise ValueError(f"Selection matched zero atoms: {selection!r}")

    unwrapped = False
    if do_unwrap:
        ok1 = add_unwrap_if_possible(u, "trajectory")
        ok2 = add_unwrap_if_possible(ref, "reference")
        unwrapped = ok1 and ok2

    analysis = rms.RMSD(
        u,
        reference=ref,
        select=selection,
        ref_frame=ref_frame,
        weights="mass",
    )

    if workers == 1:
        analysis.run(step=stride, backend="serial")
    else:
        n_parts = parts if parts > 0 else max(workers, 2 * workers)
        analysis.run(
            step=stride,
            backend="multiprocessing",
            n_workers=workers,
            n_parts=n_parts,
        )

    arr = np.asarray(analysis.results.rmsd)
    # columns: frame, time (ps in GROMACS trajectories), RMSD (Angstrom)
    df = pd.DataFrame(
        {
            "frame": arr[:, 0].astype(int),
            "time_ps": arr[:, 1],
            "time_ns": arr[:, 1] / 1000.0,
            "rmsd_A": arr[:, 2],
        }
    )

    if len(df) < 20:
        raise RuntimeError(
            f"Only {len(df)} analyzed frames are available. Use a smaller --stride or a longer trajectory."
        )
    if not np.all(np.isfinite(df["rmsd_A"])):
        raise RuntimeError("Non-finite RMSD values were produced.")
    if np.any(np.diff(df["time_ns"]) <= 0):
        raise RuntimeError("Trajectory time is not strictly increasing after applying stride.")

    return df, len(sel), unwrapped


def detect_plateau(rmsd_values: np.ndarray, user_nskip: int) -> tuple[int, float, float, int]:
    n = len(rmsd_values)
    nskip = user_nskip if user_nskip > 0 else max(1, n // 1000)
    t0, g, neff = timeseries.detect_equilibration(
        np.asarray(rmsd_values, dtype=float), nskip=nskip
    )
    return int(t0), float(g), float(neff), int(nskip)


def regression_with_hac(
    t_ns: np.ndarray,
    y: np.ndarray,
    hac_lags_user: int,
) -> dict:
    x = np.asarray(t_ns, dtype=float) - float(t_ns[0])
    y = np.asarray(y, dtype=float)
    X = sm.add_constant(x)

    ols = sm.OLS(y, X).fit()
    residuals = np.asarray(ols.resid, dtype=float)

    try:
        g_resid = float(timeseries.statistical_inefficiency(residuals, fft=True))
        if not np.isfinite(g_resid):
            g_resid = 1.0
    except Exception:
        g_resid = 1.0

    if hac_lags_user > 0:
        hac_lags = hac_lags_user
    else:
        # g is approximately 1 + 2*tau in units of analyzed frames.
        # Using ceil(g) is deliberately conservative for the HAC bandwidth here.
        hac_lags = max(1, int(np.ceil(g_resid)))

    # Avoid pathological bandwidths in short post-equilibration segments.
    hac_lags = min(hac_lags, max(1, len(y) // 4))

    hac = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags, "kernel": "bartlett", "use_correction": True},
        use_t=False,
    )

    return {
        "x": x,
        "y": y,
        "X": X,
        "ols": ols,
        "hac": hac,
        "residuals": residuals,
        "g_resid": g_resid,
        "hac_lags": hac_lags,
        "slope": float(hac.params[1]),
        "slope_se": float(hac.bse[1]),
        "intercept": float(hac.params[0]),
    }


def tost_slope(
    slope: float,
    se: float,
    delta: float,
    alpha: float,
) -> dict:
    if delta <= 0:
        raise ValueError("Equivalence slope margin must be > 0.")
    if not (0 < alpha < 0.5):
        raise ValueError("--alpha must be between 0 and 0.5.")
    if se <= 0 or not np.isfinite(se):
        raise RuntimeError("Invalid robust standard error for the slope.")

    lower_bound = -delta
    upper_bound = +delta

    # TOST:
    # H01: beta <= lower_bound  vs H11: beta > lower_bound
    # H02: beta >= upper_bound  vs H12: beta < upper_bound
    z_lower = (slope - lower_bound) / se
    p_lower = float(norm.sf(z_lower))
    z_upper = (slope - upper_bound) / se
    p_upper = float(norm.cdf(z_upper))
    p_tost = max(p_lower, p_upper)

    # A TOST at one-sided alpha is equivalent to checking the (1 - 2*alpha) CI.
    zcrit = float(norm.ppf(1.0 - alpha))
    ci_low = slope - zcrit * se
    ci_high = slope + zcrit * se

    # Conventional two-sided trend test, included only as descriptive information.
    z_zero = slope / se
    p_zero = float(2.0 * norm.sf(abs(z_zero)))

    equivalent = bool((p_tost < alpha) and (ci_low > lower_bound) and (ci_high < upper_bound))

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_level": float(1.0 - 2.0 * alpha),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": float(p_tost),
        "p_slope_zero_two_sided": p_zero,
        "equivalent": equivalent,
    }


def circular_block_resample(residuals: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    out = np.empty(n, dtype=float)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        take = min(block_len, n - pos)
        idx = (start + np.arange(take)) % n
        out[pos:pos + take] = residuals[idx]
        pos += take
    return out


def block_bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    block_len: int,
    n_boot: int,
    alpha: float,
    seed: int,
) -> dict | None:
    if n_boot <= 0:
        return None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals - residuals.mean()

    block_len = max(1, min(int(block_len), len(y)))
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot, dtype=float)
    X = np.column_stack([np.ones_like(x), x])

    for i in range(n_boot):
        e_star = circular_block_resample(residuals, block_len, rng)
        y_star = fitted + e_star
        beta_star, *_ = np.linalg.lstsq(X, y_star, rcond=None)
        slopes[i] = beta_star[1]

    q_low, q_high = np.quantile(slopes, [alpha, 1.0 - alpha])
    return {
        "n_boot": int(n_boot),
        "block_len_frames": int(block_len),
        "ci_low": float(q_low),
        "ci_high": float(q_high),
        "ci_level": float(1.0 - 2.0 * alpha),
        "median_slope": float(np.median(slopes)),
        "slopes": slopes,
    }


def make_plot(
    df: pd.DataFrame,
    t0_idx: int,
    slope: float,
    intercept: float,
    tost: dict,
    bootstrap: dict | None,
    output_png: Path,
) -> None:
    plateau = df.iloc[t0_idx:]
    t0 = float(plateau["time_ns"].iloc[0])
    x_post = plateau["time_ns"].to_numpy() - t0
    fit_post = intercept + slope * x_post

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(df["time_ns"], df["rmsd_A"], linewidth=0.8, alpha=0.75, label="RMSD")
    ax.axvline(t0, linestyle="--", linewidth=1.3, label=f"Detected t0 = {t0:.2f} ns")
    ax.plot(plateau["time_ns"], fit_post, linewidth=2.0, label="Post-t0 linear trend")

    eq_text = "YES" if tost["equivalent"] else "NO"
    boot_text = ""
    if bootstrap is not None:
        boot_text = (
            f"\nBlock-bootstrap {100*bootstrap['ci_level']:.0f}% CI: "
            f"[{bootstrap['ci_low']:.3g}, {bootstrap['ci_high']:.3g}] Å/ns"
        )

    text = (
        f"HAC slope = {slope:.3g} Å/ns\n"
        f"{100*tost['ci_level']:.0f}% CI = [{tost['ci_low']:.3g}, {tost['ci_high']:.3g}] Å/ns\n"
        f"TOST p = {tost['p_tost']:.3g}\n"
        f"Equivalent to zero within chosen margin: {eq_text}"
        f"{boot_text}"
    )
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD plateau analysis")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    xtc = args.xtc.resolve()
    if not xtc.exists():
        raise FileNotFoundError(xtc)

    topology = args.top.resolve() if args.top is not None else discover_topology(xtc)
    if not topology.exists():
        raise FileNotFoundError(topology)

    prefix = args.prefix if args.prefix is not None else xtc.with_suffix("")
    prefix = prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Trajectory : {xtc}")
    print(f"Topology   : {topology}")
    print(f"Selection  : {args.selection}")
    print(f"Workers    : {args.workers}")
    if args.workers > 1:
        print(f"RMSD backend: multiprocessing ({args.parts if args.parts > 0 else 2 * args.workers} parts)")
    else:
        print("RMSD backend: serial")

    try:
        df, n_selected, unwrapped = calculate_rmsd(
            topology=topology,
            xtc=xtc,
            selection=args.selection,
            ref_frame=args.ref_frame,
            stride=args.stride,
            do_unwrap=not args.no_unwrap,
            workers=args.workers,
            parts=args.parts,
        )
    except Exception as exc:
        # TPR is preferred because it carries bond topology, but very new
        # GROMACS TPX versions can occasionally outpace MDAnalysis support.
        # If an identically named GRO file exists, retry with it automatically.
        gro_fallback = topology.with_suffix(".gro")
        if topology.suffix.lower() == ".tpr" and gro_fallback.exists():
            print(
                f"WARNING: failed to read TPR topology ({exc}).\n"
                f"Retrying with matching GRO file: {gro_fallback}",
                file=sys.stderr,
            )
            topology = gro_fallback
            df, n_selected, unwrapped = calculate_rmsd(
                topology=topology,
                xtc=xtc,
                selection=args.selection,
                ref_frame=args.ref_frame,
                stride=args.stride,
                do_unwrap=not args.no_unwrap,
                workers=args.workers,
                parts=args.parts,
            )
        else:
            raise

    t0_idx, g_eq, neff_eq, nskip = detect_plateau(df["rmsd_A"].to_numpy(), args.nskip)
    n_post = len(df) - t0_idx
    if n_post < 10:
        raise RuntimeError(
            f"PyMBAR left only {n_post} frames after t0; this is too short for a reliable trend test."
        )
    if n_post < 50:
        warnings.warn(
            f"PyMBAR left only {n_post} frames after t0. "
            "The slope/equivalence result should be interpreted cautiously."
        )

    plateau = df.iloc[t0_idx:].copy()
    reg = regression_with_hac(
        plateau["time_ns"].to_numpy(),
        plateau["rmsd_A"].to_numpy(),
        args.hac_lags,
    )

    delta_slope = args.equiv_drift_A / args.equiv_window_ns
    tost = tost_slope(
        slope=reg["slope"],
        se=reg["slope_se"],
        delta=delta_slope,
        alpha=args.alpha,
    )

    block_len = max(1, int(np.ceil(reg["g_resid"])))
    bootstrap = block_bootstrap_slope(
        x=reg["x"],
        y=reg["y"],
        fitted=np.asarray(reg["ols"].fittedvalues),
        residuals=reg["residuals"],
        block_len=block_len,
        n_boot=args.bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )

    df["region"] = np.where(np.arange(len(df)) >= t0_idx, "post_t0", "pre_t0")
    csv_path = Path(str(prefix) + "_rmsd_timeseries.csv")
    json_path = Path(str(prefix) + "_rmsd_plateau_summary.json")
    txt_path = Path(str(prefix) + "_rmsd_plateau_summary.txt")
    png_path = Path(str(prefix) + "_rmsd_plateau.png")
    df.to_csv(csv_path, index=False)

    dt_ns = float(np.median(np.diff(df["time_ns"])))
    t0_time_ns = float(df["time_ns"].iloc[t0_idx])
    post_duration_ns = float(df["time_ns"].iloc[-1] - t0_time_ns)

    summary = {
        "input": {
            "trajectory": str(xtc),
            "topology": str(topology),
            "selection": args.selection,
            "selected_atoms": n_selected,
            "reference_frame": args.ref_frame,
            "stride": args.stride,
            "workers": args.workers,
            "parts": args.parts if args.parts > 0 else (2 * args.workers if args.workers > 1 else 1),
            "rmsd_backend": "multiprocessing" if args.workers > 1 else "serial",
            "automatic_unwrap_enabled": not args.no_unwrap,
            "automatic_unwrap_succeeded": unwrapped,
        },
        "rmsd": {
            "units": "Angstrom",
            "n_analyzed_frames": int(len(df)),
            "frame_spacing_ns": dt_ns,
        },
        "step1_equilibration_detection": {
            "method": "pymbar.timeseries.detect_equilibration",
            "nskip": nskip,
            "t0_analyzed_index": t0_idx,
            "t0_original_frame": int(df["frame"].iloc[t0_idx]),
            "t0_time_ns": t0_time_ns,
            "statistical_inefficiency_g": g_eq,
            "effective_uncorrelated_samples_Neff": neff_eq,
            "post_t0_frames": int(len(plateau)),
            "post_t0_duration_ns": post_duration_ns,
        },
        "step2_autocorrelation_robust_trend": {
            "method": "OLS slope with HAC/Newey-West covariance, Bartlett kernel",
            "slope_A_per_ns": reg["slope"],
            "slope_SE_HAC": reg["slope_se"],
            "residual_statistical_inefficiency_g": reg["g_resid"],
            "hac_maxlags_frames": reg["hac_lags"],
            "two_sided_p_for_slope_equal_zero": tost["p_slope_zero_two_sided"],
        },
        "step3_equivalence_TOST": {
            "equivalence_definition": (
                f"absolute RMSD drift < {args.equiv_drift_A} Angstrom per "
                f"{args.equiv_window_ns} ns"
            ),
            "slope_margin_A_per_ns": delta_slope,
            "alpha_one_sided": args.alpha,
            "confidence_level": tost["ci_level"],
            "slope_CI_low_A_per_ns": tost["ci_low"],
            "slope_CI_high_A_per_ns": tost["ci_high"],
            "p_lower": tost["p_lower"],
            "p_upper": tost["p_upper"],
            "p_TOST": tost["p_tost"],
            "equivalent_within_margin": tost["equivalent"],
        },
        "moving_block_bootstrap_sensitivity": None,
    }

    if bootstrap is not None:
        boot_equiv = bool(
            bootstrap["ci_low"] > -delta_slope and bootstrap["ci_high"] < delta_slope
        )
        summary["moving_block_bootstrap_sensitivity"] = {
            "replicates": bootstrap["n_boot"],
            "seed": args.seed,
            "block_length_frames": bootstrap["block_len_frames"],
            "confidence_level": bootstrap["ci_level"],
            "slope_CI_low_A_per_ns": bootstrap["ci_low"],
            "slope_CI_high_A_per_ns": bootstrap["ci_high"],
            "CI_entirely_inside_equivalence_margin": boot_equiv,
        }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        "RMSD PLATEAU ANALYSIS",
        "=====================",
        f"Trajectory: {xtc}",
        f"Topology:   {topology}",
        f"Selection:  {args.selection} ({n_selected} atoms)",
        "",
        "STEP 1 — plateau/equilibration detection",
        f"t0 = {t0_time_ns:.4f} ns (analyzed index {t0_idx}, original frame {int(df['frame'].iloc[t0_idx])})",
        f"g  = {g_eq:.4f}",
        f"Neff = {neff_eq:.2f}",
        f"post-t0 duration = {post_duration_ns:.4f} ns",
        "",
        "STEP 2 — residual temporal drift",
        f"slope = {reg['slope']:.8g} Å/ns",
        f"HAC SE = {reg['slope_se']:.8g} Å/ns",
        f"HAC maxlags = {reg['hac_lags']} analyzed frames",
        f"residual g = {reg['g_resid']:.4f}",
        f"two-sided p(slope = 0) = {tost['p_slope_zero_two_sided']:.6g}  [descriptive only]",
        "",
        "STEP 3 — TOST equivalence test",
        f"equivalence margin = ±{delta_slope:.8g} Å/ns",
        f"which corresponds to ±{args.equiv_drift_A:g} Å per {args.equiv_window_ns:g} ns",
        f"{100*tost['ci_level']:.1f}% HAC CI = [{tost['ci_low']:.8g}, {tost['ci_high']:.8g}] Å/ns",
        f"p_lower = {tost['p_lower']:.6g}",
        f"p_upper = {tost['p_upper']:.6g}",
        f"p_TOST  = {tost['p_tost']:.6g}",
        f"equivalent within chosen margin = {tost['equivalent']}",
    ]
    if bootstrap is not None:
        boot_equiv = (
            bootstrap["ci_low"] > -delta_slope and bootstrap["ci_high"] < delta_slope
        )
        lines += [
            "",
            "SENSITIVITY — moving-block bootstrap",
            f"replicates = {bootstrap['n_boot']}",
            f"block length = {bootstrap['block_len_frames']} analyzed frames",
            f"{100*bootstrap['ci_level']:.1f}% CI = [{bootstrap['ci_low']:.8g}, {bootstrap['ci_high']:.8g}] Å/ns",
            f"bootstrap CI entirely inside equivalence margin = {boot_equiv}",
        ]

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    make_plot(
        df=df,
        t0_idx=t0_idx,
        slope=reg["slope"],
        intercept=reg["intercept"],
        tost=tost,
        bootstrap=bootstrap,
        output_png=png_path,
    )

    print("\n" + "\n".join(lines[5:]))
    print("\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {json_path}")
    print(f"  {png_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
