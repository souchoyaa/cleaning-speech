#!/bin/bash
# launch.sh — run the ASR pipeline orchestrator from the interactive node.
#
# Reads pipeline/egs/<CONFIG>.yaml, dispatches each enabled stage to
# its component's existing launch.sh:
#   - parakeet, canary: sbatch + return (fire-and-forget; check squeue)
#   - qwen: synchronous feeder run (blocks until done)
#
# Flags:
#   --stages parakeet,qwen     only run those, ignoring YAML enabled flags
#   --reuse-vllm               force qwen.vllm_mode=mode_b for all datasets
#                              (reuse existing vLLM, never launch new)
#   --shutdown-vllm-on-exit    after all datasets, scancel sml jobs that
#                              weren't running when we started
#
# Usage:
#   bash pipeline/scripts/launch.sh --config datasets_batch.yaml
#   bash pipeline/scripts/launch.sh --config foo.yaml --reuse-vllm
#   bash pipeline/scripts/launch.sh --config foo.yaml --reuse-vllm --shutdown-vllm-on-exit

set -euo pipefail

CONFIG_NAME="cv_fr.yaml"
STAGES=""
REUSE_VLLM=0
SHUTDOWN_VLLM_ON_EXIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --stages) STAGES="$2";      shift 2 ;;
        --reuse-vllm)            REUSE_VLLM=1;            shift ;;
        --shutdown-vllm-on-exit) SHUTDOWN_VLLM_ON_EXIT=1; shift ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${QME_DIR}/../../../.." && pwd)"

CONFIG_PATH="${QME_DIR}/pipeline/egs/${CONFIG_NAME}"
[[ -f "${CONFIG_PATH}" ]] || { echo "ERROR: config not found: ${CONFIG_PATH}" >&2; exit 1; }

cd "${QME_DIR}"
export PYTHONPATH="${QME_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

ARGS=( --config "${CONFIG_PATH}" )
[[ -n "${STAGES}" ]] && ARGS+=( --stages "${STAGES}" )
[[ "${REUSE_VLLM}" -eq 1 ]] && ARGS+=( --reuse-vllm )
[[ "${SHUTDOWN_VLLM_ON_EXIT}" -eq 1 ]] && ARGS+=( --shutdown-vllm-on-exit )

python -m pipeline.main "${ARGS[@]}"
