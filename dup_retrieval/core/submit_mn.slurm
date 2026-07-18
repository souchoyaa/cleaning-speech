#!/bin/bash
# submit_mn.slurm — MULTI-NODE dedup (manifest→audio_match). Each SLURM task is
# one pipeline rank (pipeline.py reads SLURM_PROCID/NTASKS; fingerprint picks its
# GPU by SLURM_LOCALID). Speeds the rank-parallel stages (manifest, fingerprint,
# audio_match) ~linearly with nodes. text_dedup still runs on rank 0 (use
# backend: dask_cudf to spread it across that node's GPUs). --gpu-bind=none so
# every task sees all 4 local GPUs and selects by local rank (matches the
# single-node torchrun behaviour).
#
# Env (--export): CONFIG, REPO_DIR; optional START_FROM/STOP_AFTER.
# Nodes: set with sbatch --nodes=N (default 3 here).
#SBATCH --job-name=dedup-mn
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=04:00:00
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.err

set -euo pipefail
: "${CONFIG:?CONFIG required}"
: "${REPO_DIR:?REPO_DIR required}"
START_FROM="${START_FROM:-manifest}"
STOP_AFTER="${STOP_AFTER:-audio_match}"
echo "MULTI-NODE dedup: nodes=${SLURM_NNODES} ntasks=${SLURM_NTASKS} ${START_FROM}->${STOP_AFTER}"

srun --ntasks-per-node=4 --gpu-bind=none --environment=quality-assesment-gh200 --kill-on-bad-exit=0 bash -c "
    export PYTHONPATH='${REPO_DIR}':\${PYTHONPATH:-}
    export PYTHONUNBUFFERED=1
    python -m audio_tokenization.utils.data_selection.dup_retrieval.core.pipeline \
        --config '${CONFIG}' --start-from '${START_FROM}' --stop-after '${STOP_AFTER}'
"
echo "=== multi-node dedup finished ==="
