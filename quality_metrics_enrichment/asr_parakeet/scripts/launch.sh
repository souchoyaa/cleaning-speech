#!/bin/bash
# launch.sh — submit the standalone Parakeet driver.
#
# No vLLM dance, no waiting on another job — Parakeet is fully local
# NeMo. This is just `sbatch` with the right env vars.
#
# Usage:
#   bash asr_parakeet/scripts/launch.sh
#   bash asr_parakeet/scripts/launch.sh --config cv_multilang.yaml
#   bash asr_parakeet/scripts/launch.sh --time 06:00:00
#   bash asr_parakeet/scripts/launch.sh --dry                 # echo sbatch and exit

set -euo pipefail

CONFIG_NAME="cv_fr.yaml"
PIPELINE_TIME=""
DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG_NAME="$2";   shift 2 ;;
        --time)    PIPELINE_TIME="$2"; shift 2 ;;
        --dry)     DRY=1;              shift   ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# This script lives at <QME>/asr_parakeet/scripts/launch.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH="${QME_DIR}/asr_parakeet/egs/${CONFIG_NAME}"
[[ -f "${CONFIG_PATH}" ]] || { echo "ERROR: config not found: ${CONFIG_PATH}" >&2; exit 1; }

export CONFIG_NAME QME_DIR

SBATCH_ARGS=( --parsable --export=ALL )
[[ -n "${PIPELINE_TIME}" ]] && SBATCH_ARGS+=( --time="${PIPELINE_TIME}" )
SBATCH_ARGS+=( "${SCRIPT_DIR}/submit.slurm" )

if [[ "${DRY}" -eq 1 ]]; then
    echo "Would run:  sbatch ${SBATCH_ARGS[*]}"
    echo "Env:        QME_DIR=${QME_DIR} CONFIG_NAME=${CONFIG_NAME}"
    exit 0
fi

JOBID=$(sbatch "${SBATCH_ARGS[@]}")

echo
echo "Submitted: ${JOBID}"
echo
echo "Watch:"
echo "  squeue -u \$USER -j ${JOBID}"
echo "  tail -F /capstor/scratch/cscs/sgodey/data_selection_runs/quality/logs/pipeline_parakeet/pipeline_parakeet.latest/pipeline.log"
