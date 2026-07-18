#!/bin/bash
# launch.sh — submit the MOS pipeline (mos.main).
#
# Resolves QME_DIR from this script's location (sbatch can't, since it stages
# the batch script to /var/spool/<...>/slurm_script) and exports it so
# submit.slurm picks it up.
#
# Usage:
#   bash mos/scripts/launch.sh
#   bash mos/scripts/launch.sh --config cv_fr.yaml
#   bash mos/scripts/launch.sh --time 06:00:00         # override walltime

set -euo pipefail

CONFIG_NAME="cv_fr.yaml"
PIPELINE_TIME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_NAME="$2";   shift 2 ;;
        --time)   PIPELINE_TIME="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# This script lives at <QME>/mos/scripts/launch.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Submitting MOS pipeline (${CONFIG_NAME}) ..."

SBATCH_ARGS=(
    --parsable
    --export=ALL,CONFIG_NAME="${CONFIG_NAME}",QME_DIR="${QME_DIR}"
)
[[ -n "${PIPELINE_TIME}" ]] && SBATCH_ARGS+=( --time="${PIPELINE_TIME}" )
SBATCH_ARGS+=( "${SCRIPT_DIR}/submit.slurm" )

JOB=$(sbatch "${SBATCH_ARGS[@]}")

echo
echo "Submitted: mos = ${JOB}"
echo
echo "Watch:"
echo "  squeue -u \$USER"
echo "  tail -F /capstor/scratch/cscs/sgodey/data_selection_runs/quality/logs/pipeline_mos/pipeline_mos.latest/pipeline.log"
