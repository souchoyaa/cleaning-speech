#!/bin/bash
# submit.slurm — sbatch wrapper for the dup_retrieval pipeline (run.sh / torchrun).
#
# One node (4× GH200). run.sh launches torchrun --nproc_per_node=4 INSIDE the
# container; multi-rank stages (manifest, fingerprint, audio_match) use all 4
# ranks, the rank-0-only stages (text_dedup, retention, apply) run on rank 0.
# Needs the language's shar to have >= 4 shards (every real VoxPopuli lang has
# 57-192) so the per-rank manifest barrier completes.
#
# Env (pass via --export):
#   CONFIG      abs path to the dedup YAML (required)
#   REPO_DIR    abs path to the synced repo root (required)
#   START_FROM  pipeline stage to start at      (default manifest)
#   STOP_AFTER  pipeline stage to stop after    (default apply_to_shar)
#
# Usage:
#   sbatch --export=ALL,CONFIG=<abs.yaml>,REPO_DIR=<repo> \
#          dup_retrieval/core/submit.slurm
#
#SBATCH --job-name=dedup
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --mem=460000
#SBATCH --partition=normal
# NOTE: --environment goes on the srun step (not #SBATCH) — the cluster Slurm
# upgrade (June 2026) breaks nested srun inside a containerized batch job
# (image lacks libjson-c.so.5 the Slingshot/pyxis plugins dlopen).
#SBATCH --output=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.err

set -euo pipefail
ulimit -c 0

: "${CONFIG:?CONFIG (abs path to dedup yaml) is required}"
: "${REPO_DIR:?REPO_DIR is required}"
START_FROM="${START_FROM:-manifest}"
STOP_AFTER="${STOP_AFTER:-apply_to_shar}"
DUP="${REPO_DIR}/audio_tokenization/utils/data_selection/dup_retrieval"
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${DUP}/core/run.sh" ]] || { echo "run.sh not found under ${DUP}/core" >&2; exit 1; }

echo "CONFIG=${CONFIG}"
echo "REPO_DIR=${REPO_DIR}"
echo "stages: ${START_FROM} -> ${STOP_AFTER}"

run_phase() {   # $1=start_from $2=stop_after  — one fresh torchrun process
    srun --environment=quality-assesment-gh200 --kill-on-bad-exit=0 bash -c "
        set -euo pipefail
        export REPO_DIR='${REPO_DIR}'
        export PYTHONUNBUFFERED=1
        bash '${DUP}/core/run.sh' '${CONFIG}' '$1' '$2'
    "
}

# TEARDOWN-HANG FIX: the fingerprint stage initialises CUDA, and
# apply_to_shar's fork-based multiprocessing.Pool deadlocks at teardown when
# forked AFTER CUDA init in the same process (the pool's workers finish + write
# output_shar, but the process never exits — observed idling ~3h post-completion
# on en/fr/es/it). spawn/forkserver aren't viable (apply shares millions-of-
# entries dedup/transcript lookup dicts via fork copy-on-write — pickling them
# per worker would OOM). Fix: run apply in a SEPARATE, CUDA-free process. So a
# full run is split into two phases (validated: de's two-job split exited cleanly).
if [ "${START_FROM}" = "manifest" ] && [ "${STOP_AFTER}" = "apply_to_shar" ]; then
    echo "[two-phase] phase 1: manifest -> audio_match"
    run_phase manifest audio_match
    echo "[two-phase] phase 2: quality_ingest -> apply_to_shar (CUDA-free process)"
    run_phase quality_ingest apply_to_shar
else
    run_phase "${START_FROM}" "${STOP_AFTER}"
fi

echo "=== dedup submit.slurm completed (${CONFIG}) ==="
