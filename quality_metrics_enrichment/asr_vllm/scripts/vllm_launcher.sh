#!/bin/bash
# vllm_launcher.sh — launch a vLLM server via `sml advanced`.
#
# Parameterized version of asr/scripts/voxtral_launcher.sh: you can swap
# the served model without touching this file by setting VLLM_MODEL.
#
# Defaults: Voxtral-Small (matches the legacy fused pipeline). To run
# a different model:
#
#   VLLM_MODEL=Qwen/Qwen2-Audio-7B-Instruct bash vllm_launcher.sh
#   VLLM_MODEL=mistralai/Voxtral-Mini-3B-2507 bash vllm_launcher.sh
#
# Recognised env vars:
#
#   VLLM_MODEL          served model id (HF or local path)
#   VLLM_TIME           SLURM walltime HH:MM:SS  (default 04:00:00)
#   VLLM_WORKERS        sml replicas             (default 2)
#   VLLM_FRAMEWORK_EXTRA extra flags appended to --framework-args verbatim
#                       (e.g. "--limit-mm-per-prompt audio=1")
#
# The chosen model id must also be set in the feeder's YAML under
# ``vllm_feeder.api_model`` (sent as ``model`` in the OpenAI request).

set -euo pipefail

# Refuse to run from a compute node — sml's shim shebang resolves to
# /usr/bin/python3.11, which exists only on login nodes. Fails with
# "/users/sgodey/.local/bin/sml: cannot execute: required file not found"
# inside an interactive salloc otherwise. sml is a job submitter, no
# compute is needed.
HOST=$(hostname)
case "$HOST" in
    nid*)
        echo "ERROR: cannot run sml from a compute node ($HOST)." >&2
        echo "       Exit this interactive allocation (Ctrl+D) and re-run from a login" >&2
        echo "       node (e.g. clariden-ln001)." >&2
        exit 1
        ;;
esac

VLLM_MODEL="${VLLM_MODEL:-mistralai/Voxtral-Small-24B-2507}"
VLLM_TIME="${VLLM_TIME:-04:00:00}"
VLLM_WORKERS="${VLLM_WORKERS:-2}"
VLLM_FRAMEWORK_EXTRA="${VLLM_FRAMEWORK_EXTRA:-}"

# Voxtral needs the mistral tokenizer/config/load format. For other
# models (Qwen-Audio etc.) those flags are wrong — gate them on the
# model id. If you add another model family that needs custom load
# flags, branch here.
EXTRA_LOAD_FLAGS=""
case "${VLLM_MODEL}" in
    *Voxtral*|*voxtral*)
        EXTRA_LOAD_FLAGS="--tokenizer_mode mistral --config_format mistral --load_format mistral"
        ;;
esac

echo "[vllm_launcher] model=${VLLM_MODEL} workers=${VLLM_WORKERS} time=${VLLM_TIME}"

sml advanced \
    --firecrest-system clariden \
    --partition normal \
    --slurm-reservation SD-69241-apertus-1-5 \
    --serving-framework vllm \
    --slurm-workers "${VLLM_WORKERS}" \
    --slurm-nodes-per-worker 1 \
    --slurm-environment ~/.local/share/uv/tools/swiss-ai-model-launch/lib/python3.11/site-packages/swiss_ai_model_launch/assets/envs/vllm.toml \
    --pre-launch-cmds "pip install --quiet 'vllm[audio]' soundfile librosa audioread" \
    --slurm-time "${VLLM_TIME}" \
    --framework-args "--model ${VLLM_MODEL} \
                      --served-model-name ${VLLM_MODEL} \
                      --host 0.0.0.0 \
                      --port 5000 \
                      ${EXTRA_LOAD_FLAGS} \
                      --tensor-parallel-size 1 \
                      --data-parallel-size 4 \
                      --gpu-memory-utilization 0.9 \
                      --max-model-len 8192 \
                      --max-num-seqs 4096 \
                      --max-num-batched-tokens 32768 \
                      ${VLLM_FRAMEWORK_EXTRA}"
