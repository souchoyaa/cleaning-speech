"""Two-stream logger setup for pipeline runs.

``setup_logging`` splits records into two files:

- ``<run_name>.log``     — anything on the ``pipeline.*`` logger tree.
- ``<run_name>.lib.log`` — everything else (NeMo, vLLM, lhotse, …) plus the
  process's redirected stdout/stderr.

Rank 0 also mirrors the pipeline stream to the original stderr so
``tail -F`` on the slurm console keeps working.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_PIPELINE_PREFIX = "pipeline."

_NOISY_LIB_LOGGERS = (
    "nemo", "nemo_logger", "vllm", "transformers",
    "torch._dynamo", "torch._inductor",
    "audiobox_aesthetics", "lhotse",
)


class _RankFilter(logging.Filter):
    def __init__(self, rank: int, world_size: int) -> None:
        super().__init__()
        self._rank = rank
        self._world = world_size

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self._rank
        record.world = self._world
        return True


class _PipelineOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(_PIPELINE_PREFIX)


class _LibraryOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(_PIPELINE_PREFIX)


_LOGGING_DONE = False


def setup_logging(
    rank: int,
    world_size: int,
    log_dir: Optional[str | Path],
    run_name: str,
    *,
    level_rank0: int = logging.INFO,
    level_other: int = logging.WARNING,
) -> logging.Logger:
    global _LOGGING_DONE
    if _LOGGING_DONE:
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    rank_filter = _RankFilter(rank, world_size)

    fmt_pipeline = logging.Formatter(
        fmt="%(asctime)s %(levelname).1s [r%(rank)d/%(world)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_library = logging.Formatter(
        fmt="%(asctime)s %(levelname).1s [r%(rank)d/%(world)d] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pipeline_level = level_rank0 if rank == 0 else level_other
    saved_stderr = sys.__stderr__

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = log_dir / f"{run_name}.log"
        library_path = log_dir / f"{run_name}.lib.log"

        pipeline_fh = RotatingFileHandler(
            pipeline_path, maxBytes=20_000_000, backupCount=3, encoding="utf-8",
        )
        pipeline_fh.setLevel(pipeline_level)
        pipeline_fh.setFormatter(fmt_pipeline)
        pipeline_fh.addFilter(_PipelineOnlyFilter())
        pipeline_fh.addFilter(rank_filter)
        root.addHandler(pipeline_fh)

        # Single open stream for non-pipeline records AND library prints, so
        # rotation can't invalidate the redirected sys.stdout reference.
        try:
            lib_stream = open(library_path, mode="a", buffering=1, encoding="utf-8")
        except OSError:
            lib_stream = None

        if lib_stream is not None:
            library_sh = logging.StreamHandler(stream=lib_stream)
            library_sh.setLevel(logging.INFO)
            library_sh.setFormatter(fmt_library)
            library_sh.addFilter(_LibraryOnlyFilter())
            library_sh.addFilter(rank_filter)
            root.addHandler(library_sh)
            sys.stdout = lib_stream
            sys.stderr = lib_stream

    if rank == 0:
        sh = logging.StreamHandler(stream=saved_stderr)
        sh.setLevel(pipeline_level)
        sh.setFormatter(fmt_pipeline)
        sh.addFilter(_PipelineOnlyFilter())
        sh.addFilter(rank_filter)
        root.addHandler(sh)

    for name in _NOISY_LIB_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _LOGGING_DONE = True

    pipeline_logger = logging.getLogger(f"pipeline.{run_name}")
    pipeline_logger.info(
        "Logging initialised — pipeline=<%s> dir=<%s>", run_name,
        str(log_dir) if log_dir else "stderr-only",
    )
    return pipeline_logger
