"""Per-rank torch.profiler factory + JSONL resume helper."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def make_profiler(cfg: dict, rank: int, output_dir: str):
    """Build a perfetto-trace torch profiler, or return None if disabled."""
    if not bool(cfg.get("use_profiler", False)):
        return None
    n_active = int(cfg.get("profile_batches", 0))
    if n_active <= 0:
        return None

    warmup = int(cfg.get("profile_warmup_batches", 2))
    trace_dir = Path(cfg.get("profile_dir", output_dir)) / "profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = trace_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    handler = torch.profiler.tensorboard_trace_handler(
        str(trace_dir), worker_name=f"rank{rank:04d}",
    )
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=warmup, warmup=1, active=n_active, repeat=1),
        on_trace_ready=handler,
        record_shapes=False,
        with_stack=False,
        profile_memory=True,
    )


def load_seen_cut_ids(jsonl_path: Path) -> set:
    """Resume helper: rebuild the set of cut_ids already present in a JSONL."""
    seen: set = set()
    if not jsonl_path.exists():
        return seen
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cid = json.loads(line).get("cut_id")
            except json.JSONDecodeError:
                continue
            if cid:
                seen.add(cid)
    logger.info("Resuming: %d cut_ids already in %s", len(seen), jsonl_path.name)
    return seen
