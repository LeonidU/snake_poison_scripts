#!/usr/bin/env bash
# Independent MD replicas of a complex on one multi-GPU node (e.g. 8x RTX 4090 on vast.ai).
#
#   prep (pdb2gmx -> box -> solvate -> ions -> EM)  : once, shared by all replicas
#   NVT -> NPT(posres) -> NPT(free) -> MD           : all replicas at once, 1 GPU per replica
#
# Every replica gets its own grompp with its own gen_seed / ld_seed, so the starting
# velocities really differ. Seeds are written to seeds.tsv (for the Methods section).
#
# RUN_MODE=multidir : one MPI job, `gmx_mpi mdrun -multidir rep01 ... repNN` (needs an MPI build)
# RUN_MODE=parallel : N separate thread-MPI mdrun processes, one per GPU (no MPI needed;
#                     a crash in one replica does not kill the others)
#
# The script can be re-run: finished stages are skipped, and interrupted ones (-maxh, a dropped
# host) continue from checkpoints.
# NOTE: pdb2gmx -ter is interactive -> run the first time in a terminal (tmux), not via nohup.

set -Eeuo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# ---------------- USER SETTINGS ----------------
INPUT_PDB="${INPUT_PDB:-rank_1.pdb}"
GMX="${GMX:-/root/gromacs-2025.4/build/bin/gmx}"                   # thread-MPI build: prep, grompp, EM, RUN_MODE=parallel
GMX_MPI="${GMX_MPI:-/root/gromacs-2025.4/build_mpi/bin/gmx_mpi}"   # MPI build: RUN_MODE=multidir
MPIRUN="${MPIRUN:-mpirun}"
MPI_EXTRA="${MPI_EXTRA:---allow-run-as-root --bind-to none}"       # OpenMPI; mdrun pins threads itself
RUN_MODE="${RUN_MODE:-multidir}"                                    # multidir | parallel

FORCE_FIELD="${FORCE_FIELD:-amber03}"
WATER_MODEL="${WATER_MODEL:-tip3p}"
BOX_DISTANCE_NM="${BOX_DISTANCE_NM:-1.2}"
SALT_M="${SALT_M:-0.15}"

N_REP="${N_REP:-8}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((N_REP - 1)))}"   # comma-separated, one GPU per replica
NTOMP="${NTOMP:-$(( $(nproc) / N_REP ))}"           # OpenMP threads per replica
PIN="${PIN:-on}"                                    # set to "off" if the container's cpuset breaks pinning
SEED_BASE="${SEED_BASE:-20261103}"
MAXH_HOURS="${MAXH_HOURS:-36}"                      # mdrun stops cleanly with a checkpoint after this
EM_NTOMP="${EM_NTOMP:-$(( NTOMP < 16 ? NTOMP : 16 ))}"  # EM on 1 GPU needs few threads (GROMACS max is 64/rank)

STAGES=(nvt npt_posres npt_free md)
declare -A MDP=(
    [nvt]=01_nvt_1ns.mdp
    [npt_posres]=02_npt_posres_2ns.mdp
    [npt_free]=03_npt_free_5ns.mdp
    [md]=04_md_250ns.mdp
)
declare -A PREV=([npt_posres]=nvt [npt_free]=npt_posres [md]=npt_free)   # nvt starts from em.gro
declare -A POSRES=([nvt]=1 [npt_posres]=1)                               # stages that need -r em.gro

export GMX_MAXBACKUP=-1
export OMP_NUM_THREADS="$NTOMP"

# ---------------- HELPERS ----------------
log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

rep_dir() { printf 'rep%02d' "$1"; }

rep_dirs() {
    local i
    for ((i = 1; i <= N_REP; i++)); do rep_dir "$i"; echo; done
}

gpu_of() {   # gpu_of <replica index 1..N> -> GPU id
    local ids
    IFS=, read -ra ids <<< "$GPU_IDS"
    echo "${ids[$(( $1 - 1 ))]}"
}

# set_mdp <file> <key> <value> : replace the key (gen_vel == gen-vel) or append it.
# grompp refuses duplicated keys, so a plain append is not enough.
set_mdp() {
    local f=$1 k=$2 v=$3 k_re
    k_re="${k//_/[-_]}"
    if grep -qiE "^[[:space:]]*${k_re}[[:space:]]*=" "$f"; then
        sed -i -E "s/^[[:space:]]*${k_re}[[:space:]]*=.*/${k} = ${v}/I" "$f"
    else
        echo "${k} = ${v}" >> "$f"
    fi
}

stage_done() {   # true if every replica has <stage>.gro
    local d
    for d in $(rep_dirs); do [[ -s "$d/$1.gro" ]] || return 1; done
}

# ---------------- PREFLIGHT ----------------
preflight() {
    command -v "$GMX" >/dev/null 2>&1 \
        || die "Cannot find GROMACS executable: $GMX (thread-MPI build, cmake WITHOUT -DGMX_MPI=ON; needed for prep/grompp/EM even in multidir mode)"
    "$GMX" --version 2>/dev/null | grep -i 'MPI library' | grep -qi thread_mpi \
        || die "$GMX is not a thread-MPI build (point GMX to build/bin/gmx, not gmx_mpi)"
    [[ -s "$INPUT_PDB" ]] || die "Input PDB not found: $INPUT_PDB"
    local s
    for s in "${STAGES[@]}"; do [[ -s "${MDP[$s]}" ]] || die "MDP not found: ${MDP[$s]}"; done
    [[ -s 00_em.mdp && -s ions.mdp ]] || die "00_em.mdp / ions.mdp not found"
    (( NTOMP >= 1 )) || die "NTOMP=$NTOMP: fewer CPU cores ($(nproc)) than replicas ($N_REP)"
    (( NTOMP <= 64 && EM_NTOMP <= 64 )) || die "NTOMP/EM_NTOMP > 64: GROMACS limits OpenMP threads per rank to 64"

    local n_ids n_gpu
    n_ids=$(awk -F, '{print NF}' <<< "$GPU_IDS")
    (( n_ids == N_REP )) || die "GPU_IDS='$GPU_IDS' has $n_ids entries, N_REP=$N_REP"
    n_gpu=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    (( n_gpu >= N_REP )) || die "Need $N_REP GPUs, nvidia-smi sees $n_gpu"

    if [[ "$RUN_MODE" == multidir ]]; then
        command -v "$GMX_MPI" >/dev/null 2>&1 || die "Cannot find MPI build: $GMX_MPI (or use RUN_MODE=parallel)"
        command -v "$MPIRUN" >/dev/null 2>&1 || die "Cannot find $MPIRUN"
        "$GMX_MPI" --version 2>/dev/null | grep -i 'MPI library' | grep -qvi thread_mpi \
            || die "$GMX_MPI is not a real-MPI build (-multidir needs -DGMX_MPI=ON)"
    elif [[ "$RUN_MODE" != parallel ]]; then
        die "RUN_MODE must be 'multidir' or 'parallel', got '$RUN_MODE'"
    fi

    log "GROMACS: $("$GMX" --version 2>/dev/null | grep -m1 'GROMACS version' || echo '?')"
    nvidia-smi -L || true
    log "Mode=$RUN_MODE  replicas=$N_REP  GPUs=$GPU_IDS  threads/replica=$NTOMP  cores=$(nproc)  pin=$PIN"
}

# ---------------- SHARED PREP (once) ----------------
prep_system() {
    if [[ ! -s processed.gro || ! -s topol.top ]]; then
        "$GMX" pdb2gmx -f "$INPUT_PDB" -o processed.gro -p topol.top -i posre.itp \
            -ff "$FORCE_FIELD" -water "$WATER_MODEL" -ignh -ter
    fi
    [[ -s boxed.gro ]] || "$GMX" editconf -f processed.gro -o boxed.gro -bt dodecahedron -d "$BOX_DISTANCE_NM"
    [[ -s solv.gro ]]  || "$GMX" solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p topol.top

    if [[ ! -s ions.gro ]]; then
        "$GMX" grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
        printf 'SOL\n' | "$GMX" genion -s ions.tpr -o ions.gro -p topol.top \
            -pname NA -nname CL -neutral -conc "$SALT_M"
    fi
    [[ -s index.ndx ]] || printf 'q\n' | "$GMX" make_ndx -f ions.gro -o index.ndx

    if [[ ! -s em.gro ]]; then
        log "EM (shared)"
        "$GMX" grompp -f 00_em.mdp -c ions.gro -r ions.gro -p topol.top -n index.ndx -o em.tpr -maxwarn 1
        "$GMX" mdrun -deffnm em -ntmpi 1 -ntomp "$EM_NTOMP" -pin "$PIN" -gpu_id "$(gpu_of 1)"
    fi
}

# ---------------- PER-REPLICA GROMPP ----------------
# Run from the base directory so that topol.top finds posre.itp; outputs go to repNN/.
grompp_stage() {
    local stage=$1 idx i d mdp gseed lseed
    for idx in "${!STAGES[@]}"; do [[ "${STAGES[$idx]}" == "$stage" ]] && break; done

    for ((i = 1; i <= N_REP; i++)); do
        d=$(rep_dir "$i")
        mkdir -p "$d"
        [[ -s "$d/$stage.tpr" ]] && continue

        mdp="$d/$stage.mdp"
        cp "${MDP[$stage]}" "$mdp"
        lseed=$(( SEED_BASE + 1000 * (idx + 1) + i ))
        set_mdp "$mdp" ld_seed "$lseed"     # v-rescale / sd noise must differ between replicas too

        local args=(-f "$mdp" -p topol.top -n index.ndx -o "$d/$stage.tpr" -po "$d/${stage}_mdout.mdp")
        if [[ "$stage" == nvt ]]; then
            gseed=$(( SEED_BASE + i ))
            set_mdp "$mdp" gen_vel yes
            set_mdp "$mdp" gen_seed "$gseed"
            args+=(-c em.gro)
        else
            gseed=-
            set_mdp "$mdp" gen_vel no       # velocities come from the previous checkpoint
            args+=(-c "$d/${PREV[$stage]}.gro" -t "$d/${PREV[$stage]}.cpt")
        fi
        [[ -n "${POSRES[$stage]:-}" ]] && args+=(-r em.gro)

        "$GMX" grompp "${args[@]}" > "$d/${stage}_grompp.log" 2>&1 \
            || { tail -n 30 "$d/${stage}_grompp.log" >&2; die "grompp failed: $d/$stage"; }
        printf '%s\t%s\t%s\t%s\n' "$d" "$stage" "$gseed" "$lseed" >> seeds.tsv
    done
}

# ---------------- MDRUN ----------------
mdrun_flags() {   # mdrun_flags <stage>
    echo -deffnm "$1" -nb gpu -pme gpu -bonded gpu -update gpu \
         -ntomp "$NTOMP" -pin "$PIN" -cpi "$1.cpt" -cpt 15 -maxh "$MAXH_HOURS"
}

run_multidir() {
    local stage=$1
    mkdir -p logs
    # shellcheck disable=SC2046,SC2086  # word splitting of flags and dir list is intended
    "$MPIRUN" -np "$N_REP" $MPI_EXTRA -x OMP_NUM_THREADS -x GMX_MAXBACKUP \
        "$GMX_MPI" mdrun -multidir $(rep_dirs) $(mdrun_flags "$stage") -gpu_id "$GPU_IDS" \
        > "logs/${stage}.out" 2>&1 \
        || { tail -n 40 "logs/${stage}.out" >&2; die "mdrun -multidir failed at stage $stage"; }
}

run_parallel() {
    local stage=$1 i d pids=() fail=0
    for ((i = 1; i <= N_REP; i++)); do
        d=$(rep_dir "$i")
        [[ -s "$d/$stage.gro" ]] && continue
        # shellcheck disable=SC2046  # word splitting of flags is intended
        ( cd "$d" && CUDA_VISIBLE_DEVICES="$(gpu_of "$i")" exec "$GMX" mdrun $(mdrun_flags "$stage") \
              -ntmpi 1 -gpu_id 0 -pinoffset $(( (i - 1) * NTOMP )) -pinstride 1 \
              > "${stage}.mdrun.out" 2>&1 ) &
        pids+=("$!:$d")
    done
    local p
    for p in "${pids[@]}"; do
        wait "${p%%:*}" || { echo "[WARN] ${p#*:}/$stage failed, see ${p#*:}/${stage}.mdrun.out" >&2; fail=1; }
    done
    (( fail == 0 )) || die "some replicas failed at stage $stage (the others are complete)"
}

run_stage() {
    local stage=$1
    if stage_done "$stage"; then log "skip $stage (done in all replicas)"; return; fi
    grompp_stage "$stage"
    log "run $stage x$N_REP ($RUN_MODE)"
    "run_$RUN_MODE" "$stage"
    if ! stage_done "$stage"; then
        log "$stage stopped before the end (-maxh=$MAXH_HOURS h?). Re-run the script to continue from checkpoints."
        exit 0
    fi
}

report() {
    log "Performance (ns/day):"
    local d
    for d in $(rep_dirs); do
        printf '  %s  %s\n' "$d" "$(grep -h '^Performance:' "$d/md.log" 2>/dev/null | awk '{print $2}' | tail -n1)"
    done
}

# ---------------- MAIN ----------------
preflight
prep_system
[[ -s seeds.tsv ]] || printf 'replica\tstage\tgen_seed\tld_seed\n' > seeds.tsv
for stage in "${STAGES[@]}"; do run_stage "$stage"; done
report
log "All $N_REP replicas finished."
