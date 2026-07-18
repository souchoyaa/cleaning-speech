"""MOS pipeline entry point.

Usage::

    sbatch mos/scripts/submit.slurm

    # Local single-rank debug:
    cd <quality_metrics_enrichment>
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \\
        python -m mos.main --config mos/egs/cv_fr.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import yaml

from common.logging import setup_logging
from common.runtime import resolve_run_context
from common.timing import RunMetrics

from .worker import MosAssessmentWorker, STAGES

logger = logging.getLogger("pipeline.mos.main")


def main(cfg: dict) -> dict:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    torch.cuda.set_device(local_rank)

    pipeline_name = "mos"
    run_name, log_dir = resolve_run_context(pipeline_name)
    setup_logging(rank, world_size, log_dir, run_name)
    logger.info("Starting MOS pipeline (local_rank=%d)", local_rank)

    metrics = RunMetrics(
        rank=rank, world_size=world_size,
        pipeline=pipeline_name, run_name=run_name,
        stages=STAGES,
    )

    worker = MosAssessmentWorker(
        cfg=cfg, rank=rank, world_size=world_size, local_rank=local_rank,
        decode_num_workers=int(cfg.get("decode_num_workers", 4)),
        metrics=metrics,
    )

    output_dir = cfg.get("output_dir", "./results_mos")
    if cfg.get("unique_output_subdir", False):
        # Per-run subdir keyed off the launcher's log dir — lets a sweep
        # re-launch with different params without clobbering or skipping cuts.
        tag = (
            Path(os.environ["PIPELINE_LOG_DIR"]).name
            if os.environ.get("PIPELINE_LOG_DIR")
            else f"run_{time.strftime('%Y%m%dT%H%M%S')}_pid{os.getpid()}"
        )
        output_dir = str(Path(output_dir) / tag)
        logger.info("unique_output_subdir → writing to %s", output_dir)

    stats = worker.run(output_dir=output_dir)
    logger.info("Process completed successfully.")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
