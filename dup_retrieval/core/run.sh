#!/bin/bash
# ==============================================================================
# dup_retrieval pipeline launcher (mirrors quality_assesment/core/run.sh).
# Drives Stages A → F (and optional G) via the orchestrator, with torchrun
# for the multi-rank stages.  Stage C (text_dedup) runs single-host inside
# rank 0 by default; for multi-node Stage C scaling, see submit_mn.slurm.
# ==============================================================================

set -euo pipefail

# ----- Paths (edit per cluster) -----------------------------------------------
BASE_DIR="${BASE_DIR:-/users/sgodey/home/semester_project}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/sync-project/benchmark-audio-tokenizer-w-dedup}"
LHOTSE_DIR="${LHOTSE_DIR:-${BASE_DIR}/lhotse}"

PIPELINE_DIR="${REPO_DIR}/audio_tokenization/utils/data_selection/dup_retrieval/core"
EGS_DIR="${REPO_DIR}/audio_tokenization/utils/data_selection/dup_retrieval/egs"

# AudioBox isn't used by dup_retrieval directly, but Stage F's quality JSONLs
# may have been produced by quality_assesment which needs it; export anyway
# so notebook helpers don't choke if invoked from this shell.
export AUDIOBOX_SRC="${AUDIOBOX_SRC:-${BASE_DIR}/audiobox-aesthetics/src}"

# Quality JSONLs live next to the originating Shar; cfg's
# retention.quality_search_paths can override.
export PYTHONPATH="${REPO_DIR}:${LHOTSE_DIR}:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

CONFIG="${1:-${EGS_DIR}/full.yaml}"
START_FROM="${2:-manifest}"
# Default through Stage G so a full run emits the clone (augmented cuts.jsonl.gz
# + symlinked audio).  apply_to_shar is a no-op unless enabled in the config.
STOP_AFTER="${3:-apply_to_shar}"

cd "${PIPELINE_DIR}"

# Disable core dumps so a DataLoader crash doesn't fill /var/crash.
ulimit -c 0 || true

echo "======================================================"
echo "dup_retrieval pipeline"
echo "  CONFIG     : ${CONFIG}"
echo "  START_FROM : ${START_FROM}"
echo "  STOP_AFTER : ${STOP_AFTER}"
echo "  PYTHONPATH : ${PYTHONPATH}"
echo "======================================================"

# Single-node (4 GPU): manifest+fingerprint+audio_match use ranks 0..3; C/F use rank 0.
torchrun --standalone --nproc_per_node=4 -m \
    audio_tokenization.utils.data_selection.dup_retrieval.core.pipeline \
    --config "${CONFIG}" \
    --start-from "${START_FROM}" \
    --stop-after "${STOP_AFTER}"

# ----- Multi-node SLURM (uncomment) -------------------------------------------
# srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 --kill-on-bad-exit=0 \
#     python -m audio_tokenization.utils.data_selection.dup_retrieval.core.pipeline \
#     --config "${CONFIG}" \
#     --start-from "${START_FROM}" \
#     --stop-after "${STOP_AFTER}"
