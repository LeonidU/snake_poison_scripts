#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ---------------- USER SETTINGS ----------------
INPUT_PDB="${INPUT_PDB:-/home/luroshlev/disintegrins/1zjo_disintergrin/receptor-ligand_model1_non_het.pdb}"
GMX="${GMX:-/home/luroshlev/software/gromacs-2025.4-cuda/bin/gmx}"
FORCE_FIELD="${FORCE_FIELD:-amber03}"
WATER_MODEL="${WATER_MODEL:-tip3p}"
BOX_DISTANCE_NM="${BOX_DISTANCE_NM:-1.2}"
SALT_M="${SALT_M:-0.15}"
GPU_ID="${GPU_ID:-0}"
MAXH_HOURS="${MAXH_HOURS:-36}"  # checkpoint/requeue-friendly segment

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-28}"
export GMX_MAXBACKUP=-1


command -v "$GMX" >/dev/null 2>&1 || {
    echo "Cannot find GROMACS executable: $GMX" >&2
    exit 1
}
[[ -s "$INPUT_PDB" ]] || {
    echo "Input PDB not found: $INPUT_PDB" >&2
    exit 1
}


echo "=== GROMACS ==="
"$GMX" --version
echo "=== GPU ==="
nvidia-smi || true
echo "OpenMP threads: $OMP_NUM_THREADS"

# 1. Protein topology.
# IMPORTANT: pdb2gmx may ask about termini and disulfide bonds.
if [[ ! -s processed.gro || ! -s topol.top ]]; then
    "$GMX" pdb2gmx \
        -f "$INPUT_PDB" \
        -o processed.gro \
        -p topol.top \
        -i posre.itp \
        -ff "$FORCE_FIELD" \
        -water "$WATER_MODEL" \
        -ignh \
        -ter
fi

# 2. Box and solvent.
if [[ ! -s boxed.gro ]]; then
    "$GMX" editconf \
        -f processed.gro \
        -o boxed.gro \
        -bt dodecahedron \
        -d "$BOX_DISTANCE_NM"
fi

if [[ ! -s solv.gro ]]; then
    "$GMX" solvate \
        -cp boxed.gro \
        -cs spc216.gro \
        -o solv.gro \
        -p topol.top
fi

# 3. Neutralize and add physiological salt.
if [[ ! -s ions.gro ]]; then
    "$GMX" grompp \
        -f ions.mdp \
        -c solv.gro \
        -p topol.top \
        -o ions.tpr \
        -maxwarn 2

    # SOL is normally the solvent group. The name-based selection avoids hard-coded numbers.
    printf 'SOL\n' | "$GMX" genion \
        -s ions.tpr \
        -o ions.gro \
        -p topol.top \
        -pname NA \
        -nname CL \
        -neutral \
        -conc "$SALT_M"
fi

# Create explicit Protein / Non-Protein groups used by tc-grps.
if [[ ! -s index.ndx ]]; then
    printf 'q\n' | "$GMX" make_ndx -f ions.gro -o index.ndx
fi

"$GMX" grompp -f 00_em.mdp -c ions.gro -r ions.gro -p topol.top -o em.tpr -maxwarn 1
"$GMX" mdrun -deffnm em  -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -pin on -gpu_id "$GPU_ID"

"$GMX" grompp -f 01_nvt_1ns.mdp   -c em.gro -r em.gro -p topol.top -o rep1_nvt.tpr
"$GMX" mdrun -deffnm rep1_nvt  -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -pin on -gpu_id "$GPU_ID"

"$GMX" grompp -f 02_npt_posres_2ns.mdp   -c rep1_nvt.gro -r em.gro -t rep1_nvt.cpt   -p topol.top -o rep1_npt_posres.tpr
"$GMX" mdrun -deffnm rep1_npt_posres  -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -pin on -gpu_id "$GPU_ID"

"$GMX" grompp -f 03_npt_free_5ns.mdp   -c rep1_npt_posres.gro -t rep1_npt_posres.cpt   -p topol.top -o rep1_npt_free.tpr
"$GMX" mdrun -deffnm rep1_npt_free  -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -pin on -gpu_id "$GPU_ID"

"$GMX" grompp -f 04_md_250ns.mdp   -c rep1_npt_free.gro -t rep1_npt_free.cpt   -p topol.top -o rep1_md.tpr
"$GMX" mdrun -deffnm rep1_md -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -pin on -gpu_id "$GPU_ID"
