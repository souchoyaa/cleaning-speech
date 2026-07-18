"""Run-context environment helpers.

The slurm wrappers export ``PIPELINE_LOG_DIR`` and ``PIPELINE_RUN_NAME`` so
all ranks of one job land in the same per-run log directory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional


def resolve_run_context(pipeline_name: str) -> tuple[str, Optional[Path]]:
    """Read ``PIPELINE_RUN_NAME`` / ``PIPELINE_LOG_DIR`` set by the launcher."""
    run_name = os.environ.get("PIPELINE_RUN_NAME") or (
        f"{pipeline_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    )
    log_dir_env = os.environ.get("PIPELINE_LOG_DIR")
    return run_name, (Path(log_dir_env) if log_dir_env else None)
