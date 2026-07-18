#!/bin/bash
# submit.slurm — sbatch wrapper for the standalone Canary driver.
#
# Single SLURM job on one GPU node (4× GH200, 288 CPU). No remote
# vLLM, no WHISPER_API_BASE — Canary + NFA + VAD all run locally on
# the GPUs.
#
# Override knobs (env vars):
#   CONFIG_NAME — YAML in egs/ to use      (default: cv_fr.yaml)
#   OUT_BASE    — root for results + logs  (default: /capstor/scratch/cscs/sgodey/data_selection_runs/quality)
#   LHOTSE_DIR  — editable lhotse fork to prepend (optional; image already ships one)
#
# Usage:
#   bash asr_canary/scripts/launch.sh                                  # delegates to this script
#   sbatch --export=ALL,QME_DIR=<abs>,CONFIG_NAME=cv_fr.yaml asr_canary/scripts/submit.slurm
#
#SBATCH --job-name=pipeline-canary
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mem=460000
#SBATCH --partition=normal
# NOTE: --environment is applied on the `srun` step below, NOT here. Running
# `srun` *inside* a containerized batch job broke after the cluster Slurm
# upgrade (June 2026): the image lacks libjson-c.so.5 that the new
# Slingshot/pyxis SPANK plugins dlopen, so the nested srun aborts instantly.
# Containerizing only the srun step keeps srun on the host (which has the lib).
#SBATCH --output=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.err

set -euo pipefail
ulimit -c 0

CONFIG_NAME="${CONFIG_NAME:-cv_fr.yaml}"

# ---- locate the tree -------------------------------------------------------
if [[ -z "${QME_DIR:-}" ]]; then
    echo "ERROR: QME_DIR is not set. Submit via launch.sh, or pass --export=ALL,QME_DIR=<abs path to quality_metrics_enrichment>" >&2
    exit 1
fi
[[ -d "${QME_DIR}/asr_canary" ]] || {
    echo "ERROR: QME_DIR=${QME_DIR} does not contain asr_canary/." >&2
    exit 1
}
REPO_DIR="$(cd "${QME_DIR}/../../../.." && pwd)"

OUT_BASE="${OUT_BASE:-/capstor/scratch/cscs/${USER}/data_selection_runs/quality}"
export QME_DIR REPO_DIR OUT_BASE

CONFIG="${QME_DIR}/asr_canary/egs/${CONFIG_NAME}"
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }

LOG_DIR="${OUT_BASE}/logs/pipeline_canary"
RUN_NAME_BASE="pipeline_canary"
RUN_TS="$(date +%Y%m%dT%H%M%S)"
JOB_TAG="job${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME_BASE}_${RUN_TS}_${JOB_TAG}"
RUN_DIR="${LOG_DIR}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

export PIPELINE_LOG_DIR="${RUN_DIR}"
export PIPELINE_RUN_NAME="${RUN_NAME}"
ln -sfn "${RUN_NAME}" "${LOG_DIR}/${RUN_NAME_BASE}.latest"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_LOG_HOME="${OUT_BASE}/slurm_raw"
    ln -sfn "${SLURM_LOG_HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${RUN_DIR}/slurm.out"
    ln -sfn "${SLURM_LOG_HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${RUN_DIR}/slurm.err"
    ln -sfn slurm.err "${RUN_DIR}/pipeline.log"
fi

cd "${QME_DIR}"

# Export so the srun bash -c '...' subshell can resolve ${CONFIG} under
# `set -u`. ${QME_DIR} is already exported above.
export CONFIG

echo "QME dir:      ${QME_DIR}"
echo "Repo dir:     ${REPO_DIR}"
echo "Run name:     ${RUN_NAME}"
echo "Run dir:      ${RUN_DIR}"
echo "Config:       ${CONFIG}"
echo "World size:   ${SLURM_NTASKS}"
echo "Output:       see <output_dir>/canary/canary_rank_*.jsonl"

srun --environment=quality-assesment-gh200 --chdir="${QME_DIR}" --kill-on-bad-exit=0 bash -c '
    set -euo pipefail
    cd "${QME_DIR}"
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export LOCAL_RANK="${SLURM_LOCALID}"
    export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE}"
    export GROUP_RANK="${SLURM_NODEID}"

    export TQDM_DISABLE=1

    export PYTHONPATH="${QME_DIR}:${REPO_DIR}:${LHOTSE_DIR:-}:${PYTHONPATH:-}"
    python -m asr_canary.main --config "${CONFIG}"
'

echo "=== asr_canary completed (run=${RUN_NAME}) ==="
