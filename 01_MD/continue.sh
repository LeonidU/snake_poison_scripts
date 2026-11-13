#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ============================================================
# SETTINGS
# ============================================================

GMX="${GMX:-/home/luroshlev/software/gromacs-2025.4-cuda/bin/gmx}"

GPU_ID="${GPU_ID:-0}"

# Prefix of the finished 250 ns production run
BASE="${BASE:-rep1_md}"

# Additional simulation time:
# GROMACS uses ps here -> 100000 ps = 100 ns
EXTEND_PS="${EXTEND_PS:-100000}"

# Extended TPR
EXTENDED_TPR="${EXTENDED_TPR:-rep1_md_350ns.tpr}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-28}"
export GMX_MAXBACKUP=-1


# ============================================================
# CHECK INPUT
# ============================================================

command -v "$GMX" >/dev/null 2>&1 || {
    echo "Cannot find GROMACS executable: $GMX" >&2
    exit 1
}

for file in \
    "${BASE}.tpr" \
    "${BASE}.cpt" \
    "${BASE}.xtc" \
    "${BASE}.edr" \
    "${BASE}.log"
do
    if [[ ! -s "$file" ]]; then
        echo "Required file not found or empty: $file" >&2
        exit 1
    fi
done


echo "============================================================"
echo "Continuing production MD"
echo "Current trajectory : ${BASE}.xtc"
echo "Checkpoint         : ${BASE}.cpt"
echo "Original TPR       : ${BASE}.tpr"
echo "Extension          : ${EXTEND_PS} ps"
echo "OpenMP threads     : ${OMP_NUM_THREADS}"
echo "GPU                : ${GPU_ID}"
echo "============================================================"

"$GMX" --version
nvidia-smi || true


# ============================================================
# EXTEND TPR
#
# Original trajectory: 250 ns
# Extension:           100 ns
# Final time:          350 ns
# ============================================================

"$GMX" convert-tpr \
    -s "${BASE}.tpr" \
    -extend "${EXTEND_PS}" \
    -o "${EXTENDED_TPR}"


# ============================================================
# CONTINUE MD
#
# -cpi     load coordinates, velocities, thermostat/barostat
#          state etc. from the checkpoint
#
# -append  append new frames/energies/log entries to the
#          existing rep1_md.xtc / rep1_md.edr / rep1_md.log
#
# Thus rep1_md.xtc will become a continuous 0-350 ns trajectory.
# ============================================================

"$GMX" mdrun \
    -s "${EXTENDED_TPR}" \
    -deffnm "${BASE}" \
    -cpi "${BASE}.cpt" \
    -append \
    -ntmpi 1 \
    -ntomp "${OMP_NUM_THREADS}" \
    -pin on \
    -gpu_id "${GPU_ID}" \
    -cpt 15


echo
echo "============================================================"
echo "Production continuation finished."
echo "Expected final simulation time: 350 ns"
echo "Trajectory: ${BASE}.xtc"
echo "Final structure: ${BASE}.gro"
echo "Checkpoint: ${BASE}.cpt"
echo "============================================================"