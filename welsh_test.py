#!/usr/bin/env python3

import argparse
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Parsing gmx_MMPBSA FINAL_RESULT.dat
# ============================================================

FLOAT_RE = (
    r"[-+]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[Ee][-+]?\d+)?"
)

ROW_RE = re.compile(
    rf"^\s*(?P<component>.+?)\s{{2,}}"
    rf"(?P<average>{FLOAT_RE})"
    rf"(?:\s+{FLOAT_RE})*\s*$"
)


def normalize_component(name):

    name = name.strip()

    # gmx_MMPBSA uses Greek delta
    name = name.replace("Δ", "DELTA ")

    name = name.upper()

    name = re.sub(
        r"[^A-Z0-9]+",
        "_",
        name
    )

    return name.strip("_")


def detect_model(line, current_model):

    u = line.upper()

    if "GENERALIZED BORN" in u:
        return "GB"

    if "POISSON BOLTZMANN" in u:
        return "PB"

    if "GBNSR6" in u:
        return "GBNSR6"

    if "3D-RISM" in u or "3D RISM" in u:
        return "3D_RISM"

    return current_model

def parse_dat(filename):
    """
    Parse binding-energy terms from the

        Delta (Complex - Receptor - Ligand):

    sections of gmx_MMPBSA FINAL_RESULTS_MMPBSA.dat.

    The file may contain several models, e.g.
        GENERALIZED BORN
        POISSON BOLTZMANN

    Returns
    -------
    list of dicts with:
        model
        component
        component_raw
        average_kcal_mol
    """

    results = []

    model = "MMPBSA"
    in_delta = False
    rows_started = False

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as fh:

        for line in fh:

            # ------------------------------------------------
            # Detect calculation model
            # ------------------------------------------------

            model = detect_model(line, model)

            stripped = line.strip()
            u = stripped.upper()

            # ------------------------------------------------
            # Start of Delta section
            #
            # gmx_MMPBSA normally uses:
            #
            # Delta (Complex - Receptor - Ligand):
            #
            # but support "Differences" too.
            # ------------------------------------------------

            if (
                ("DELTA" in u or "DIFFERENCES" in u)
                and "COMPLEX" in u
                and "RECEPTOR" in u
                and "LIGAND" in u
            ):
                in_delta = True
                rows_started = False
                continue

            if not in_delta:
                continue

            # ------------------------------------------------
            # End of Delta section
            #
            # Important:
            # ignore the first separator before data,
            # but the separator AFTER rows means section end.
            # ------------------------------------------------

            if (
                rows_started
                and stripped
                and set(stripped) <= {"-"}
            ):
                in_delta = False
                rows_started = False
                continue

            # ------------------------------------------------
            # Ignore blanks.
            #
            # DO NOT finish section here:
            # gmx_MMPBSA places blank lines between
            # ΔESURF, ΔGGAS, ΔGSOLV and ΔTOTAL.
            # ------------------------------------------------

            if not stripped:
                continue

            # table separators
            if stripped.startswith("-"):
                continue

            # header
            if "ENERGY COMPONENT" in u:
                continue

            # ------------------------------------------------
            # Energy row
            # ------------------------------------------------

            m = ROW_RE.match(line)

            if m:

                component_raw = (
                    m.group("component").strip()
                )

                component = normalize_component(
                    component_raw
                )

                average = float(
                    m.group("average")
                )

                results.append(
                    {
                        "model": model,
                        "component": component,
                        "component_raw": component_raw,
                        "average_kcal_mol": average,
                    }
                )

                rows_started = True
                continue

            # ------------------------------------------------
            # Defensive stop if another named section appears
            # ------------------------------------------------

            if (
                rows_started
                and stripped.endswith(":")
                and not stripped.startswith("Δ")
            ):
                in_delta = False
                rows_started = False

    return results

# ============================================================
# Collect replicas
# ============================================================

def find_result_files(group_dir, filename):
    group_dir = Path(group_dir)

    files = sorted(group_dir.rglob(filename))

    if not files:
        raise FileNotFoundError(
            f"No {filename} files found below {group_dir}"
        )

    return files


def collect_group(group_name, group_dir, filename):
    files = find_result_files(group_dir, filename)

    rows = []

    for file in files:

        replica = str(file.parent.relative_to(group_dir))

        parsed = parse_dat(file)

        if not parsed:
            print(f"WARNING: nothing parsed from {file}")
            continue

        for x in parsed:

            rows.append(
                {
                    "group": group_name,
                    "replica": replica,
                    "source": str(file),
                    "model": x["model"],
                    "component": x["component"],
                    "component_raw": x["component_raw"],
                    "mean_kcal_mol": x["average_kcal_mol"],
                }
            )

    columns = [
        "group",
        "replica",
        "source",
        "model",
        "component",
        "component_raw",
        "mean_kcal_mol",
    ]

    return pd.DataFrame(
        rows,
        columns=columns
    )


# ============================================================
# Statistics
# ============================================================

def welch_ci(x, y, alpha=0.05):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    mx = np.mean(x)
    my = np.mean(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    diff = mx - my

    a = vx / nx
    b = vy / ny

    se = np.sqrt(a + b)

    if se == 0:
        return diff, diff, diff, np.inf

    df = (a + b) ** 2 / (
        a**2 / (nx - 1) +
        b**2 / (ny - 1)
    )

    tcrit = stats.t.ppf(1 - alpha / 2, df)

    low = diff - tcrit * se
    high = diff + tcrit * se

    return diff, low, high, df


def hedges_g(x, y):
    """
    Small-sample corrected standardized effect size.
    Positive: GTA > GTB
    Negative: GTA < GTB
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)

    pooled_var = (
        (nx - 1) * sx2 +
        (ny - 1) * sy2
    ) / (nx + ny - 2)

    if pooled_var <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_var)

    d = (np.mean(x) - np.mean(y)) / pooled_sd

    # Small-sample correction
    correction = 1 - 3 / (4 * (nx + ny) - 9)

    return correction * d


def permutation_test_mean(
    x,
    y,
    max_exact=200000,
    n_random=100000,
    seed=12345,
):
    """
    Two-sided permutation test for difference in means.

    For 9 GTA + 9 GTB:
        C(18, 9) = 48620

    so the exact permutation test is feasible.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    values = np.concatenate([x, y])

    nx = len(x)
    n = len(values)

    observed = abs(np.mean(x) - np.mean(y))

    total_sum = values.sum()

    n_combinations = math.comb(n, nx)

    tolerance = 1e-12

    # --------------------------------------------------------
    # Exact permutation test
    # --------------------------------------------------------

    if n_combinations <= max_exact:

        extreme = 0

        for inds in itertools.combinations(range(n), nx):

            s1 = values[list(inds)].sum()

            m1 = s1 / nx
            m2 = (total_sum - s1) / (n - nx)

            d = abs(m1 - m2)

            if d >= observed - tolerance:
                extreme += 1

        return extreme / n_combinations, n_combinations, "exact"

    # --------------------------------------------------------
    # Monte Carlo permutation test
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    extreme = 0

    for _ in range(n_random):

        inds = rng.choice(n, size=nx, replace=False)

        s1 = values[inds].sum()

        m1 = s1 / nx
        m2 = (total_sum - s1) / (n - nx)

        d = abs(m1 - m2)

        if d >= observed - tolerance:
            extreme += 1

    p = (extreme + 1) / (n_random + 1)

    return p, n_random, "monte_carlo"


def bh_fdr(pvalues):
    """
    Benjamini-Hochberg FDR correction.
    """

    pvalues = np.asarray(pvalues, dtype=float)

    qvalues = np.full(len(pvalues), np.nan)

    valid = np.isfinite(pvalues)

    p = pvalues[valid]

    if len(p) == 0:
        return qvalues

    order = np.argsort(p)
    ranked = p[order]

    m = len(ranked)

    q = ranked * m / np.arange(1, m + 1)

    # Enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]

    q = np.minimum(q, 1.0)

    reverse = np.empty(m, dtype=int)
    reverse[order] = np.arange(m)

    qvalues[valid] = q[reverse]

    return qvalues


def compare_groups(df, min_replicates=3):

    results = []

    keys = (
        df[["model", "component"]]
        .drop_duplicates()
        .sort_values(["model", "component"])
    )

    for _, key in keys.iterrows():

        model = key["model"]
        component = key["component"]

        sub = df[
            (df["model"] == model)
            & (df["component"] == component)
        ]

        x = (
            sub.loc[sub["group"] == "GTA", "mean_kcal_mol"]
            .dropna()
            .to_numpy()
        )

        y = (
            sub.loc[sub["group"] == "GTB", "mean_kcal_mol"]
            .dropna()
            .to_numpy()
        )

        nx = len(x)
        ny = len(y)

        if nx < min_replicates or ny < min_replicates:
            continue

        # Descriptive statistics
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        sd_x = np.std(x, ddof=1)
        sd_y = np.std(y, ddof=1)

        # Welch t-test
        test = stats.ttest_ind(
            x,
            y,
            equal_var=False,
            nan_policy="omit",
        )

        diff, ci_low, ci_high, welch_df = welch_ci(x, y)

        # Effect size
        g = hedges_g(x, y)

        # Permutation test
        perm_p, n_perm, perm_type = permutation_test_mean(x, y)

        results.append(
            {
                "model": model,
                "component": component,

                "n_GTA": nx,
                "mean_GTA": mean_x,
                "sd_GTA": sd_x,

                "n_GTB": ny,
                "mean_GTB": mean_y,
                "sd_GTB": sd_y,

                "difference_GTA_minus_GTB": diff,
                "difference_CI95_low": ci_low,
                "difference_CI95_high": ci_high,

                "welch_t": test.statistic,
                "welch_df": welch_df,
                "welch_p": test.pvalue,

                "hedges_g": g,

                "permutation_p": perm_p,
                "permutation_type": perm_type,
                "n_permutations": n_perm,
            }
        )

    out = pd.DataFrame(results)

    if len(out) == 0:
        return out

    # Multiple-testing correction
    out["welch_q_BH"] = bh_fdr(out["welch_p"].values)
    out["permutation_q_BH"] = bh_fdr(
        out["permutation_p"].values
    )

    out["significant_Welch_FDR05"] = (
        out["welch_q_BH"] < 0.05
    )

    out["significant_permutation_FDR05"] = (
        out["permutation_q_BH"] < 0.05
    )

    return out


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare replicate-level MMPBSA energies "
            "between GTA and GTB."
        )
    )

    parser.add_argument(
        "--gta",
        default="GTA",
        help="Directory containing GTA replicas",
    )

    parser.add_argument(
        "--gtb",
        default="GTB",
        help="Directory containing GTB replicas",
    )

    parser.add_argument(
        "--filename",
        default="FINAL_RESULTS_MMPBSA.dat",
        help="MMPBSA result filename",
    )

    parser.add_argument(
        "--out-prefix",
        default="GTA_vs_GTB_MMPBSA",
    )

    parser.add_argument(
        "--min-replicates",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    gta_dir = Path(args.gta)
    gtb_dir = Path(args.gtb)

    # --------------------------------------------------------
    # Read replicas
    # --------------------------------------------------------

    gta = collect_group(
        "GTA",
        gta_dir,
        args.filename,
    )

    gtb = collect_group(
        "GTB",
        gtb_dir,
        args.filename,
    )

    data = pd.concat(
        [gta, gtb],
        ignore_index=True,
    )

    if data.empty:
        raise RuntimeError(
            "No MMPBSA Delta energy terms were parsed. "
            "Check input files and their format."
        )

    # Save replicate-level table
    replica_file = (
        f"{args.out_prefix}_replica_means.csv"
    )

    data.to_csv(
        replica_file,
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    result = compare_groups(
        data,
        min_replicates=args.min_replicates,
    )

    stats_file = (
        f"{args.out_prefix}_statistics.csv"
    )

    result.to_csv(
        stats_file,
        index=False,
    )

    # --------------------------------------------------------
    # Extract total binding free energy
    # --------------------------------------------------------

    total_mask = result["component"].str.contains(
        r"(?:^|_)TOTAL$",
        regex=True,
        na=False,
    )

    totals = result[total_mask].copy()

    total_file = (
        f"{args.out_prefix}_TOTAL_statistics.csv"
    )

    totals.to_csv(
        total_file,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MMPBSA GTA vs GTB")
    print("=" * 80)

    print(
        f"GTA replicas: "
        f"{gta['replica'].nunique()}"
    )

    print(
        f"GTB replicas: "
        f"{gtb['replica'].nunique()}"
    )

    print()
    print(f"Replica means: {replica_file}")
    print(f"Statistics:    {stats_file}")
    print(f"TOTAL only:    {total_file}")

    if len(totals):

        print()
        print("=" * 80)
        print("PRIMARY ENDPOINT: TOTAL BINDING FREE ENERGY")
        print("=" * 80)

        for _, r in totals.iterrows():

            print()
            print(
                f"{r['model']} / {r['component']}"
            )

            print(
                f"  GTA: "
                f"{r['mean_GTA']:.3f} ± "
                f"{r['sd_GTA']:.3f} kcal/mol "
                f"(n={int(r['n_GTA'])})"
            )

            print(
                f"  GTB: "
                f"{r['mean_GTB']:.3f} ± "
                f"{r['sd_GTB']:.3f} kcal/mol "
                f"(n={int(r['n_GTB'])})"
            )

            print(
                f"  GTA - GTB: "
                f"{r['difference_GTA_minus_GTB']:.3f} "
                f"kcal/mol"
            )

            print(
                f"  95% CI: "
                f"[{r['difference_CI95_low']:.3f}, "
                f"{r['difference_CI95_high']:.3f}]"
            )

            print(
                f"  Welch p: "
                f"{r['welch_p']:.6g}"
            )

            print(
                f"  Welch q (BH): "
                f"{r['welch_q_BH']:.6g}"
            )

            print(
                f"  Permutation p: "
                f"{r['permutation_p']:.6g} "
                f"({r['permutation_type']})"
            )

            print(
                f"  Hedges g: "
                f"{r['hedges_g']:.3f}"
            )

            if r["difference_GTA_minus_GTB"] < 0:
                print(
                    "  Direction: GTA has the more negative "
                    "mean binding free energy."
                )
            elif r["difference_GTA_minus_GTB"] > 0:
                print(
                    "  Direction: GTB has the more negative "
                    "mean binding free energy."
                )

    print()


if __name__ == "__main__":
    main()