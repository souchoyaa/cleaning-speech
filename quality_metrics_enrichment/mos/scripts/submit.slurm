#!/bin/bash
# submit.slurm — sbatch wrapper for the MOS pipeline (mos.main).
#
# One srun task per GPU. SLURM_* env vars are translated to the names
# torch.distributed expects (RANK, WORLD_SIZE, LOCAL_RANK).
#
# QME_DIR is passed in by the submitter (launch.sh), so the tree can be cloned
# anywhere on the cluster and run as-is. We can't self-locate via BASH_SOURCE
# because sbatch stages this script to /var/spool/<...>/slurm_script.
#
# Uses the same image as the ASR pipeline — quality-assesment-gh200 already
# ships lhotse (Alvorecer721 fork), audiobox-aesthetics, vLLM, NeMo ≥ 2.5,
# and the OMP/MKL env vars are baked in.
#
# Override knobs (env vars):
#   CONFIG_NAME         — YAML in egs/ to use             (default: cv_fr.yaml)
#   OUT_BASE            — root for results + logs         (default: /capstor/scratch/cscs/sgodey/data_selection_runs/quality)
#   LHOTSE_DIR          — editable lhotse fork to prepend (optional; image already ships one)
#   AUDIOBOX_SRC        — editable audiobox source to prepend (optional; image already ships one)
#
# Usage:
#   bash mos/scripts/launch.sh                          # 1 node, 4 GPUs (default)
#   bash mos/scripts/launch.sh --config foo.yaml --time 06:00:00
#
#   # direct sbatch (must pass QME_DIR explicitly):
#   sbatch --export=ALL,QME_DIR=<abs path to quality_metrics_enrichment> \
#          mos/scripts/submit.slurm
#
#SBATCH --job-name=pipeline-mos
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

# ---- locate the tree --------------------------------------------------------
# QME_DIR must be passed in by the submitter (launch.sh exports it via --export).
# We can't self-locate via BASH_SOURCE because sbatch stages this script to
# /var/spool/<...>/slurm_script and runs it from there.
if [[ -z "${QME_DIR:-}" ]]; then
    echo "ERROR: QME_DIR is not set. Submit via launch.sh, or pass --export=ALL,QME_DIR=<abs path to quality_metrics_enrichment>" >&2
    exit 1
fi
[[ -d "${QME_DIR}/mos" ]] || { echo "ERROR: QME_DIR=${QME_DIR} does not contain mos/." >&2; exit 1; }
# REPO_DIR = QME's grandparent×4: <repo>/audio_tokenization/utils/data_selection/quality_metrics_enrichment
REPO_DIR="$(cd "${QME_DIR}/../../../.." && pwd)"

OUT_BASE="${OUT_BASE:-/capstor/scratch/cscs/sgodey/data_selection_runs/quality}"
export QME_DIR REPO_DIR OUT_BASE
# AUDIOBOX_SRC is read by mos/models/audiobox.py at builder time. Export only
# if explicitly set so the image's installed audiobox_aesthetics wins by default.
[[ -n "${AUDIOBOX_SRC:-}" ]] && export AUDIOBOX_SRC

CONFIG="${QME_DIR}/mos/egs/${CONFIG_NAME}"
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }

# Per-job log directory — pipeline.log + symlinked .latest for tail -F.
LOG_DIR="${OUT_BASE}/logs/pipeline_mos"
RUN_NAME_BASE="pipeline_mos"
RUN_TS="$(date +%Y%m%dT%H%M%S)"
JOB_TAG="job${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME_BASE}_${RUN_TS}_${JOB_TAG}"
RUN_DIR="${LOG_DIR}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

export PIPELINE_LOG_DIR="${RUN_DIR}"
export PIPELINE_RUN_NAME="${RUN_NAME}"
ln -sfn "${RUN_NAME}" "${LOG_DIR}/${RUN_NAME_BASE}.latest"

# ---- telemetry / wandb defaults --------------------------------------------
# Compute nodes are typically air-gapped on clariden, so wandb runs offline
# by default. wandb nests its data under ${WANDB_DIR}/wandb/offline-run-*,
# so set WANDB_DIR to the *parent* of the wandb subdir we want.
# Sync from the login node when the job completes:
#     wandb sync ${RUN_DIR}/telemetry/wandb/offline-run-*
# Override WANDB_MODE=online (and provide WANDB_API_KEY) to push live.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${RUN_DIR}/telemetry"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
mkdir -p "${WANDB_DIR}/wandb"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_LOG_HOME="${OUT_BASE}/slurm_raw"
    ln -sfn "${SLURM_LOG_HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${RUN_DIR}/slurm.out"
    ln -sfn "${SLURM_LOG_HOME}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${RUN_DIR}/slurm.err"
fi

cd "${QME_DIR}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
export CONFIG MASTER_ADDR MASTER_PORT

echo "QME dir:      ${QME_DIR}"
echo "Repo dir:     ${REPO_DIR}"
echo "Run name:     ${RUN_NAME}"
echo "Run dir:      ${RUN_DIR}"
echo "Config:       ${CONFIG}"
echo "World size:   ${SLURM_NTASKS}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Telemetry:    JSONL → ${RUN_DIR}/telemetry/rank_*.jsonl"
echo "              wandb (mode=${WANDB_MODE}) → ${WANDB_DIR}"
echo "After job:    wandb sync ${WANDB_DIR}/offline-run-*"

srun --environment=quality-assesment-gh200 --chdir="${QME_DIR}" --kill-on-bad-exit=0 bash -c '
    set -euo pipefail
    cd "${QME_DIR}"
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export LOCAL_RANK="${SLURM_LOCALID}"
    export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE}"
    export GROUP_RANK="${SLURM_NODEID}"

    # Image PYTHONPATH already covers lhotse / audiobox / vllm / nemo.
    # Prepend the user tree so `from common.* / mos.* / audio_tokenization.*`
    # resolve. ${LHOTSE_DIR:-} is appended only if explicitly set (override).
    export PYTHONPATH="${QME_DIR}:${REPO_DIR}:${LHOTSE_DIR:-}:${PYTHONPATH:-}"
    python -m mos.main --config "${CONFIG}"
'

echo "=== mos pipeline completed (run=${RUN_NAME}) ==="
