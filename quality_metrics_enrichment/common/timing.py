"""Per-stage wall-clock timing + run-level metrics.

``StageTimer`` is a context manager that adds its elapsed wall time to a
``RunMetrics``. ``RunMetrics`` accumulates per-stage seconds, sample counts,
and emits periodic / final progress lines.
"""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Optional


class StageTimer(AbstractContextManager):
    """Wall-clock timer that adds its duration to a RunMetrics."""

    __slots__ = ("_metrics", "_stage", "_profiler_ctx", "_t0")

    def __init__(
        self,
        metrics: "RunMetrics",
        stage: str,
        *,
        profiler_rf: Optional[AbstractContextManager] = None,
    ) -> None:
        self._metrics = metrics
        self._stage = stage
        self._profiler_ctx = profiler_rf
        self._t0 = 0.0

    def __enter__(self) -> "StageTimer":
        if self._profiler_ctx is not None:
            self._profiler_ctx.__enter__()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._metrics.tick(self._stage, time.perf_counter() - self._t0)
        if self._profiler_ctx is not None:
            self._profiler_ctx.__exit__(exc_type, exc, tb)


@dataclass
class RunMetrics:
    rank: int
    world_size: int
    pipeline: str
    run_name: str
    t_start: float = 0.0

    stages: tuple[str, ...] = ()

    batches: int = 0
    samples: int = 0
    samples_skipped: int = 0
    samples_failed: int = 0
    samples_filtered: int = 0
    audio_secs: float = 0.0

    stage_seconds: dict[str, float] = field(default_factory=dict)
    stage_calls: dict[str, int] = field(default_factory=dict)

    last_log_time: float = 0.0
    last_log_audio_secs: float = 0.0

    def __post_init__(self) -> None:
        now = time.perf_counter()
        if not self.t_start:
            self.t_start = now
        self.last_log_time = now
        for s in self.stages:
            self.stage_seconds.setdefault(s, 0.0)
            self.stage_calls.setdefault(s, 0)

    def tick(self, stage: str, secs: float) -> None:
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + secs
        self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1

    def add_batch(
        self, n: int, audio_secs: float = 0.0, *, status: str = "ok",
    ) -> None:
        if status == "ok":
            self.batches += 1
            self.samples += n
            self.audio_secs += audio_secs
        elif status == "skipped":
            self.samples_skipped += n
        elif status == "failed":
            self.samples_failed += n
        elif status == "filtered":
            self.samples_filtered += n
        else:
            raise ValueError(f"unknown status: {status!r}")

    def maybe_periodic_log(
        self, logger: logging.Logger, *, every_n: int = 50,
    ) -> None:
        if self.batches == 0 or self.batches % every_n != 0:
            return
        now = time.perf_counter()
        interval_elapsed = max(now - self.last_log_time, 1e-6)
        interval_audio_secs = self.audio_secs - self.last_log_audio_secs
        thr_rank = interval_audio_secs / interval_elapsed
        thr_total = thr_rank * self.world_size
        logger.info(
            "batch %d | %d samples | %.1f audio-h/wall-h (rank) ~%.1f (world×%d)",
            self.batches, self.samples, thr_rank, thr_total, self.world_size,
        )
        self.last_log_time = now
        self.last_log_audio_secs = self.audio_secs

    def final_summary(self, logger: logging.Logger) -> None:
        import torch
        torch.cuda.synchronize()

        wall = max(time.perf_counter() - self.t_start, 1e-6)
        thr = self.audio_secs / wall

        seen = set(self.stages)
        unknown = sorted(
            (s for s in self.stage_seconds if s not in seen),
            key=lambda s: self.stage_seconds[s],
            reverse=True,
        )
        ordered = [s for s in self.stages if s in self.stage_seconds] + unknown

        stage_total = sum(self.stage_seconds.values())
        other = max(wall - stage_total, 0.0)

        max_name = max((len(s) for s in ordered), default=0)
        max_name = max(max_name, len("other"))
        rows = [
            f"    {s:<{max_name}}  {(self.stage_seconds.get(s, 0.0) / wall) * 100.0:5.1f} %  {self.stage_seconds.get(s, 0.0):8.1f} s"
            for s in ordered
        ]
        rows.append(
            f"    {'other':<{max_name}}  {(other / wall) * 100.0:5.1f} %  {other:8.1f} s"
        )

        logger.info(
            "Pipeline finished — %s\n"
            "  wall time   : %8.1f s\n"
            "  audio       : %8.2f h\n"
            "  throughput  : %8.2f audio-h / wall-h\n"
            "  samples     : %d  (skipped %d, failed %d, filtered %d)\n"
            "  stage time share:\n%s",
            self.pipeline, wall, self.audio_secs / 3600.0, thr,
            self.samples, self.samples_skipped, self.samples_failed,
            self.samples_filtered,
            "\n".join(rows),
        )

    def as_stats(self) -> dict:
        return {
            "rank": self.rank,
            "batches_processed": self.batches,
            "samples_processed": self.samples,
            "samples_skipped": self.samples_skipped,
            "samples_failed": self.samples_failed,
            "samples_filtered": self.samples_filtered,
            "audio_hours": self.audio_secs / 3600.0,
        }
