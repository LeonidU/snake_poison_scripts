#!/usr/bin/env bash
set -euo pipefail

IMAGE="${RMSD_PLATEAU_IMAGE:-rmsd-plateau:1.0}"

usage() {
    cat <<USAGE
Usage:
  $(basename "$0") /path/to/data_folder [rmsd_plateau.py options]

The folder must contain:
  - one production trajectory named rep*_md.xtc (for example rep1_md.xtc), and
  - preferably the matching rep*_md.tpr topology.

Other XTC files (NVT, NPT, MMPBSA-prepared trajectories, etc.) are ignored.
To override automatic selection, use environment variables:
  XTC_FILE=step5_production.xtc TOP_FILE=step5_production.tpr \\
    $(basename "$0") /path/to/data_folder --bootstrap 5000

Examples:
  $(basename "$0") ./replica_1

  $(basename "$0") ./replica_1 \\
    --selection "protein and backbone" \\
    --equiv-drift-A 0.10 \\
    --equiv-window-ns 100 \\
    --bootstrap 5000

  XTC_FILE=md_200ns.xtc TOP_FILE=md_200ns.tpr \\
    $(basename "$0") ./replica_1 --stride 10

PBC preprocessing:
  By default the container uses GROMACS 2025.4 to create
  rep*_md_rmsd_prepared.xtc with:
    1) trjconv -pbc nojump
    2) trjconv -center -pbc mol -ur compact
  The prepared XTC is cached and reused on later runs.
  Use --rebuild-pbc to regenerate it or --no-gmx-pbc to disable this step.

Parallel RMSD calculation:
  By default the Python script uses up to 8 worker processes.
  Override it either with --workers N or with RMSD_WORKERS=N.

  RMSD_WORKERS=16 $(basename "$0") ./replica_1

Image can be overridden with:
  RMSD_PLATEAU_IMAGE=my-image:tag $(basename "$0") ./replica_1
USAGE
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

DATA_DIR="$1"
shift

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: directory does not exist: $DATA_DIR" >&2
    exit 2
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd -P)"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker was not found in PATH." >&2
    exit 127
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: Docker image '$IMAGE' is not available locally." >&2
    echo "Build it first from the directory containing Dockerfile:" >&2
    echo "  docker build -t $IMAGE ." >&2
    exit 3
fi

choose_xtc() {
    if [[ -n "${XTC_FILE:-}" ]]; then
        local p="$DATA_DIR/$XTC_FILE"
        [[ -f "$p" ]] || { echo "ERROR: XTC_FILE not found: $p" >&2; exit 4; }
        printf '%s\n' "$XTC_FILE"
        return
    fi

    # Production trajectories in this project use the prefix rep*_md.
    # Deliberately ignore equilibration and derived trajectories such as
    # rep*_nvt.xtc, rep*_npt_*.xtc and mmpbsa_prepared*.xtc.
    local prod_files=()
    while IFS= read -r -d '' f; do
        prod_files+=("$(basename "$f")")
    done < <(find "$DATA_DIR" -maxdepth 1 -type f -iname 'rep*_md.xtc' -print0 | sort -z)

    if (( ${#prod_files[@]} == 1 )); then
        printf '%s\n' "${prod_files[0]}"
        return
    elif (( ${#prod_files[@]} > 1 )); then
        echo "ERROR: more than one production trajectory matching rep*_md.xtc was found:" >&2
        printf '  %s\n' "${prod_files[@]}" >&2
        echo "Set XTC_FILE=<basename> to choose one explicitly." >&2
        exit 4
    fi

    echo "ERROR: no production trajectory matching rep*_md.xtc was found in $DATA_DIR" >&2
    echo "Expected a file such as rep1_md.xtc." >&2
    echo "If this directory uses another naming scheme, set XTC_FILE=<basename>." >&2
    exit 4
}

choose_topology() {
    local xtc="$1"

    if [[ -n "${TOP_FILE:-}" ]]; then
        local p="$DATA_DIR/$TOP_FILE"
        [[ -f "$p" ]] || { echo "ERROR: TOP_FILE not found: $p" >&2; exit 5; }
        printf '%s\n' "$TOP_FILE"
        return
    fi

    local stem="${xtc%.*}"

    # Prefer the production TPR with exactly the same prefix as the XTC.
    # TPR contains the most complete topology information for MDAnalysis.
    if [[ -f "$DATA_DIR/$stem.tpr" ]]; then
        printf '%s\n' "$stem.tpr"
        return
    elif [[ -f "$DATA_DIR/$stem.gro" ]]; then
        printf '%s\n' "$stem.gro"
        return
    elif [[ -f "$DATA_DIR/$stem.pdb" ]]; then
        printf '%s\n' "$stem.pdb"
        return
    fi

    local preferred_names=("topol.tpr" "md.tpr" "production.tpr" "prod.tpr")
    local preferred=()
    local candidate
    for candidate in "${preferred_names[@]}"; do
        [[ -f "$DATA_DIR/$candidate" ]] && preferred+=("$candidate")
    done
    if (( ${#preferred[@]} == 1 )); then
        printf '%s\n' "${preferred[0]}"
        return
    elif (( ${#preferred[@]} > 1 )); then
        echo "ERROR: several preferred topology files found:" >&2
        printf '  %s\n' "${preferred[@]}" >&2
        echo "Set TOP_FILE=<basename> to choose one." >&2
        exit 5
    fi

    local files=()
    while IFS= read -r -d '' f; do
        files+=("$(basename "$f")")
    done < <(find "$DATA_DIR" -maxdepth 1 -type f \( -iname '*.tpr' -o -iname '*.gro' -o -iname '*.pdb' \) -print0 | sort -z)

    if (( ${#files[@]} == 0 )); then
        echo "ERROR: no .tpr/.gro/.pdb topology found in $DATA_DIR" >&2
        exit 5
    elif (( ${#files[@]} == 1 )); then
        printf '%s\n' "${files[0]}"
        return
    fi

    echo "ERROR: topology selection is ambiguous in $DATA_DIR:" >&2
    printf '  %s\n' "${files[@]}" >&2
    echo "Set TOP_FILE=<basename> to choose one." >&2
    exit 5
}

XTC="$(choose_xtc)"
TOP="$(choose_topology "$XTC")"

echo "RMSD plateau analysis"
echo "  folder   : $DATA_DIR"
echo "  image    : $IMAGE"
echo "  trajectory: $XTC"
echo "  topology : $TOP"
echo

TTY_ARGS=()
if [[ -t 1 ]]; then
    TTY_ARGS=(-t)
fi

# If the caller did not pass --workers explicitly, allow RMSD_WORKERS
# to control the number of multiprocessing workers. Otherwise the
# Python default is used (up to 8 workers).
EXTRA_ARGS=()
if [[ -n "${RMSD_WORKERS:-}" ]]; then
    has_workers=0
    for arg in "$@"; do
        if [[ "$arg" == "--workers" || "$arg" == --workers=* ]]; then
            has_workers=1
            break
        fi
    done
    if (( has_workers == 0 )); then
        EXTRA_ARGS+=(--workers "$RMSD_WORKERS")
    fi
fi

docker run --rm \
    "${TTY_ARGS[@]}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e MPLBACKEND=Agg \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    -e GMX_MAXBACKUP=-1 \
    -v "$DATA_DIR:/data:rw" \
    "$IMAGE" \
    "/data/$XTC" \
    --top "/data/$TOP" \
    "${EXTRA_ARGS[@]}" \
    "$@"
