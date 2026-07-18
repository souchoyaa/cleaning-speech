#!/bin/bash
# run_eval.slurm — run one eval python command in the container as an sbatch job,
# so it has a job-id other jobs can --dependency=afterok on.
# Env (via --export): REPO_DIR (abs repo root), EVAL_CMD (the python command).
#SBATCH --job-name=eval-py
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.err
set -euo pipefail
: "${REPO_DIR:?REPO_DIR required}"
: "${EVAL_CMD:?EVAL_CMD required}"
srun --environment=quality-assesment-gh200 bash -c "export PYTHONPATH='${REPO_DIR}'; ${EVAL_CMD}"
