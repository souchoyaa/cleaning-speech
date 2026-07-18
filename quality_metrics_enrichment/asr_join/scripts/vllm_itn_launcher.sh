#!/bin/bash
# Launch google/gemma-4-E4B-it (small multilingual Instruct, ~2B effective,
# ~140 langs) on one Clariden GH200 node with vLLM, DP=4 (4 independent
# replicas, one per GPU). Used as the LLM backend for asr_join's inverse
# text normalization (ITN) pass.
#
# Why E4B: ITN is a simple per-row rewrite task (no reasoning, short
# input/output). Tiny models match the workload — Gemma-4-E4B-it gives
# ~3–5× the throughput of Qwen3-ASR-1.7B for text-only chat completions,
# at no quality cost for this task.
#
# This script wraps the upstream
#   swiss-ai/model-launch/examples/clariden/cli/google/gemma-4-E4B-it-vllm.sh
# while fixing two repeat issues we saw with the Qwen3-ASR launcher:
#   1. ``sml`` shim's shebang only works on login nodes — refuse fast on
#      compute nodes with a clear error.
#   2. The upstream script uses ``src/...vllm.toml`` (relative path) for
#      ``--slurm-environment``, which only resolves when run from a
#      checked-out repo. We use the absolute path inside the uv tool
#      install so it works from any CWD.
#
# Env overrides (so a config flip doesn't require touching this script):
#   VLLM_TIME         walltime HH:MM:SS                      (default 06:00:00)
#   VLLM_PORT         worker port + framework --port         (default 8080)
#   VLLM_MODEL_PATH   local path to model weights            (default below)
#   VLLM_SERVED_NAME  --served-model-name                    (default google/gemma-4-E4B-it-${USER})
#   VLLM_ENV_TOML     --slurm-environment path               (default below)
#   VLLM_FRAMEWORK_EXTRA  extra --framework-args flags       (default "")

set -euo pipefail

# Parse --time / --port CLI flags as alternatives to env vars. Keeps env
# fallback so existing usage doesn't break.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time) VLLM_TIME="$2"; shift 2 ;;
        --port) VLLM_PORT="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# `sml` is a Python shim whose shebang (#!.../python3) only resolves on
# the login node — compute nodes lack /usr/bin/python3.11, so launching
# from inside an interactive `salloc` session fails with
#   "/users/sgodey/.local/bin/sml: cannot execute: required file not found"
# Refuse fast and tell the user where to run from. sml is a job
# submitter, no compute is needed here.
HOST=$(hostname)
case "$HOST" in
    nid*)
        echo "ERROR: cannot run sml from a compute node ($HOST)." >&2
        echo "       Exit this interactive allocation (Ctrl+D) and re-run from a login" >&2
        echo "       node (e.g. clariden-ln001). sml submits a SLURM job; no GPU needed here." >&2
        exit 1
        ;;
esac

VLLM_TIME="${VLLM_TIME:-06:00:00}"
VLLM_PORT="${VLLM_PORT:-8080}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/capstor/store/cscs/swissai/infra01/hf_models/models/google/gemma-4-E4B-it}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-google/gemma-4-E4B-it-${USER}}"
# Absolute path to the vllm.toml shipped inside the swiss-ai-model-launch
# uv tool install — same file the Qwen3-ASR launcher uses. Override with
# VLLM_ENV_TOML=src/... if running from a local checkout of model-launch.
VLLM_ENV_TOML="${VLLM_ENV_TOML:-$HOME/.local/share/uv/tools/swiss-ai-model-launch/lib/python3.11/site-packages/swiss_ai_model_launch/assets/envs/vllm.toml}"
VLLM_FRAMEWORK_EXTRA="${VLLM_FRAMEWORK_EXTRA:-}"

echo "[itn_launcher] model=${VLLM_MODEL_PATH}"
echo "[itn_launcher] served-name=${VLLM_SERVED_NAME}"
echo "[itn_launcher] port=${VLLM_PORT}  walltime=${VLLM_TIME}"
echo "[itn_launcher] After this submits, watch:  squeue -u \$USER"
echo "[itn_launcher] When RUNNING, point asr_join at it via:"
echo "[itn_launcher]   export ITN_API_BASE=\"http://<nid>:${VLLM_PORT}\""
echo "[itn_launcher]   export ITN_MODEL=\"${VLLM_SERVED_NAME}\""

sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-reservation SD-69241-apertus-1-5-0 \
  --slurm-nodes 1 \
  --slurm-time "${VLLM_TIME}" \
  --serving-framework vllm \
  --worker-port "${VLLM_PORT}" \
  --slurm-environment "${VLLM_ENV_TOML}" \
  --framework-args "--model ${VLLM_MODEL_PATH} \
    --served-model-name ${VLLM_SERVED_NAME} \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    ${VLLM_FRAMEWORK_EXTRA}"
