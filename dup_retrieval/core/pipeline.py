"""End-to-end orchestrator for the dup_retrieval pipeline.

Runs Stages A → F (and optionally G) in order.  Per-stage modules can also be
invoked standalone (each has its own ``main()``) — this orchestrator is for
single-host single-shot use.

Multi-rank stages (A, D, E) require torchrun / SLURM to populate RANK +
WORLD_SIZE; running this script with --start-from manifest from a single host
processes rank 0 only by default.

Use ``--start-from`` and ``--stop-after`` to slice the pipeline:

    python pipeline.py --config egs/full.yaml --start-from text_dedup --stop-after audio_match
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STAGES = [
    "manifest",
    "text_dedup",
    "audio_fingerprint",
    "audio_match",
    "quality_ingest",
    "retention",
    "apply_to_shar",
]

# Filesystem-based rank barrier timeout (seconds).  Bumped beyond what any
# realistic stage takes to prevent spurious aborts on slow nodes.
_BARRIER_TIMEOUT_SECS = 7200


def _wait_for_all_ranks(parts_dir: Path, world_size: int,
                         pattern_template: str,
                         timeout: float = _BARRIER_TIMEOUT_SECS) -> None:
    """Block until ``pattern_template.format(rank=r)`` exists in *parts_dir*
    for every r in [0, world_size).

    Used as a lightweight cross-process barrier on rank 0 before running
    finalize work that depends on all ranks' outputs.  Avoids requiring
    torch.distributed initialization for stages that don't need GPU-side
    collectives.
    """
    expected = [parts_dir / pattern_template.format(rank=r)
                for r in range(world_size)]
    deadline = time.time() + timeout
    while True:
        missing = [p.name for p in expected if not p.exists()]
        if not missing:
            logger.info("Barrier passed: all %d rank outputs present in %s",
                        world_size, parts_dir.name)
            return
        if time.time() > deadline:
            raise TimeoutError(
                f"Barrier timeout after {timeout}s waiting for {missing[:3]}"
                f"{'...' if len(missing) > 3 else ''} in {parts_dir}.")
        time.sleep(2.0)


def _wait_for_success(stage_dir: Path,
                       timeout: float = _BARRIER_TIMEOUT_SECS) -> None:
    """Block until ``stage_dir/_SUCCESS`` exists.  Used by every rank as a
    pre-stage barrier so a rank doesn't race ahead of an upstream stage that
    only runs on rank 0."""
    success = stage_dir / "_SUCCESS"
    deadline = time.time() + timeout
    while not success.exists():
        if time.time() > deadline:
            raise TimeoutError(
                f"Barrier timeout after {timeout}s waiting for {success}.")
        time.sleep(2.0)


# Keys are orchestrator stage names; values are upstream OUTPUT DIR names
# (canonical, see run_layout.STAGE) whose _SUCCESS must exist first.
_UPSTREAM_DIRS = {
    "text_dedup":       ["manifest"],
    "audio_fingerprint": ["text_dedup"],
    "audio_match":       ["text_dedup", "fingerprint"],
    "retention":         ["audio_match", "text_dedup"],
    "apply_to_shar":     ["retention"],
}


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _run_stage(name: str, cfg: dict, rank: int, world_size: int) -> None:
    logger.info("=== Stage %s starting (rank %d/%d) ===", name, rank, world_size)
    output_dir = Path(cfg["output_dir"])

    # Pre-stage barrier: every rank waits for upstream _SUCCESS markers.
    # Necessary because single-host stages (B, C, F, G) only do work on
    # rank 0; without this, ranks 1..N would race ahead and try to read
    # outputs that haven't been written yet.
    for upstream_subdir in _UPSTREAM_DIRS.get(name, []):
        _wait_for_success(output_dir / upstream_subdir)
    if name == "manifest":
        from . import manifest
        manifest.run_rank(cfg, rank=rank, world_size=world_size)
        if rank == 0:
            _wait_for_all_ranks(output_dir / "manifest", world_size,
                                 "part_{rank:04d}.parquet")
            manifest.assert_unique_and_mark_success(cfg, rank=rank)
    elif name == "text_dedup":
        from . import text_dedup
        if rank == 0:
            text_dedup.run(cfg)
    elif name == "audio_fingerprint":
        from . import audio_fingerprint
        audio_fingerprint.run_rank(cfg, rank=rank, world_size=world_size)
        if rank == 0:
            _wait_for_all_ranks(output_dir / "fingerprint", world_size,
                                 "part_{rank:04d}.parquet")
            audio_fingerprint.finalize(cfg)
    elif name == "audio_match":
        from . import audio_match
        audio_match.run_rank(cfg, rank=rank, world=world_size)
        if rank == 0:
            _wait_for_all_ranks(output_dir / "audio_match", world_size,
                                 "part_{rank:04d}.done")
            audio_match.merge(cfg, world_size)
    elif name == "quality_ingest":
        from . import quality_ingest
        if rank == 0:
            quality_ingest.run(cfg)
    elif name == "retention":
        from . import retention
        if rank == 0:
            retention.run(cfg)
    elif name == "apply_to_shar":
        from . import apply_to_shar
        if rank == 0:
            apply_to_shar.run(cfg)
    else:
        raise ValueError(f"Unknown stage: {name}")
    logger.info("=== Stage %s done (rank %d) ===", name, rank)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Dup retrieval orchestrator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-from", default="manifest", choices=STAGES)
    parser.add_argument("--stop-after", default="apply_to_shar", choices=STAGES)
    parser.add_argument("--rank", type=int,
                        default=int(os.environ.get("RANK",
                                    os.environ.get("SLURM_PROCID", 0))))
    parser.add_argument("--world-size", type=int,
                        default=int(os.environ.get("WORLD_SIZE",
                                    os.environ.get("SLURM_NTASKS", 1))))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [rank %(processName)s] %(name)s: %(message)s",
    )
    cfg = _load_cfg(args.config)
    start_idx = STAGES.index(args.start_from)
    stop_idx  = STAGES.index(args.stop_after)
    if stop_idx < start_idx:
        raise ValueError("--stop-after must be at or after --start-from")
    for name in STAGES[start_idx:stop_idx + 1]:
        _run_stage(name, cfg, rank=args.rank, world_size=args.world_size)

    # Roll up a single clean index of the run's outputs (rank 0 only).
    if args.rank == 0:
        from . import run_layout
        mp = run_layout.write_run_manifest(cfg["output_dir"])
        logger.info("Wrote run manifest: %s", mp)


if __name__ == "__main__":
    main()
