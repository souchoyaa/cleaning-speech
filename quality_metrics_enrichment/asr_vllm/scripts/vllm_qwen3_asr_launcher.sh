#!/bin/bash
# Launch Qwen/Qwen3-ASR-1.7B (1.7B multilingual ASR, 52 langs incl.
# 22 Chinese dialects, built on Qwen3-Omni foundation) on one Clariden
# GH200 node with vLLM, DP=4 TP=1 (4 independent replicas, one per GPU).
# Suitable for high-throughput batch / streaming ASR over many audio clips.
#
# Qwen3-ASR uses the Qwen3ASRForConditionalGeneration architecture, which
# is registered in stock vLLM 0.19+ (no vllm-omni needed). The generic
# ``vllm.toml`` env points at the ci/vllm_cuda13 image (vLLM 0.19.1rc1,
# transformers 5.5.4, torchaudio 2.11) which has the full audio arch set
# and the newer Qwen3ASRConfig schema (with thinker_config). The image
# is missing librosa/audioread (vLLM's audio file loader), so we install
# them at launch via --pre-launch-cmds.
#
# Model weights (downloaded separately):
#   /capstor/store/cscs/swissai/infra01/MLLM/audio_asr/Qwen3-ASR-1.7B/
#
# Env overrides (so a config flip from Voxtral → Qwen3-ASR doesn't
# require touching this script):
#
#   VLLM_TIME         walltime HH:MM:SS                     (default 06:00:00)
#   VLLM_PORT         worker port + framework --port        (default 8080)
#   VLLM_MODEL_PATH   local path to model weights           (default below)
#   VLLM_SERVED_NAME  ``--served-model-name``               (default Qwen/Qwen3-ASR-1.7B-${USER})
#   VLLM_ENV_TOML     ``--slurm-environment`` path          (default below)
#   VLLM_FRAMEWORK_EXTRA  extra ``--framework-args`` flags  (default "")

set -euo pipefail

# Parse --time CLI flag as alternative to VLLM_TIME env. Keeps the env
# fallback so existing scripts/usage don't break.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time) VLLM_TIME="$2"; shift 2 ;;
        --port) VLLM_PORT="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# `sml` is a shim whose shebang (#!.../python3) only resolves on the
# login node — compute nodes lack /usr/bin/python3.11, so launching
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
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/capstor/store/cscs/swissai/infra01/MLLM/audio_asr/Qwen3-ASR-1.7B}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen/Qwen3-ASR-1.7B-${USER}}"
# Absolute path to the vllm.toml shipped inside the `swiss-ai-model-launch`
# uv tool install — same file the legacy Voxtral launcher points at.
# Override with VLLM_ENV_TOML=src/... if you're running from a local
# checkout of the swiss-ai-model-launch source tree.
VLLM_ENV_TOML="${VLLM_ENV_TOML:-$HOME/.local/share/uv/tools/swiss-ai-model-launch/lib/python3.11/site-packages/swiss_ai_model_launch/assets/envs/vllm.toml}"
VLLM_FRAMEWORK_EXTRA="${VLLM_FRAMEWORK_EXTRA:-}"

echo "[qwen3_asr_launcher] model=${VLLM_MODEL_PATH}"
echo "[qwen3_asr_launcher] served-name=${VLLM_SERVED_NAME}"
echo "[qwen3_asr_launcher] port=${VLLM_PORT}  walltime=${VLLM_TIME}"

sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-reservation SD-69241-apertus-1-5-0 \
  --slurm-nodes 1 \
  --slurm-time "${VLLM_TIME}" \
  --serving-framework vllm \
  --worker-port "${VLLM_PORT}" \
  --slurm-environment "${VLLM_ENV_TOML}" \
  --pre-launch-cmds "pip install --quiet 'vllm[audio]' soundfile librosa audioread" \
  --framework-args "--model ${VLLM_MODEL_PATH} \
    --served-model-name ${VLLM_SERVED_NAME} \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --trust-remote-code \
    ${VLLM_FRAMEWORK_EXTRA}"
