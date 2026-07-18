"""Async vLLM feeder (runs on the interactive node, no GPU).

Pipeline: shar (direct tar reader, N IO threads) -> bounded asyncio.Queue ->
N async consumers calling ``backend.transcribe()`` with retry+backoff -> single
async writer (JSONL + sidecar). vLLM does its own server-side batching.

Resume: the per-rank JSONL is scanned at startup; written cut_ids are skipped and
their cumulative duration seeds ``stats.resume_offset_s``.

Failure semantics: permanent errors (4xx, bad_json, no_audio_blob) are written to
JSONL **and** sidecar so reruns skip; transient errors (exhausted retries) write
nothing, so reruns retry the cut.

Run::
    python -m asr_vllm.main --config egs/cv_fr_voxtral.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import yaml

from common.run_history import end_run, guard_cfg_hash, start_run

from .backends import build_backend
from .backends.base import TransientError
from .shar_reader import CutItem, iter_shar_cuts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — keep next to the YAML keys they back so they're easy to find.
# ---------------------------------------------------------------------------

_DEFAULT_CONCURRENCY = 128
_DEFAULT_TIMEOUT_S = 240.0
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_RETRY_BACKOFF_S = 0.5
_DEFAULT_SHARD_READERS = 4
_DEFAULT_READER_QUEUE = 512
_DEFAULT_WORK_QUEUE = 256
_DEFAULT_LOG_EVERY = 1000
_DEFAULT_HEARTBEAT_S = 5.0  # terminal "I'm alive" cadence regardless of throughput
# Circuit-breaker: abort the run if this many consecutive transient errors hit.
# Catches "vLLM died mid-run" — at default backoff (~0.5..8s × 4 retries × 50 cuts)
# the breaker trips in ~3-5 minutes once the server actually goes down, vs.
# silently burning through the remaining wall time. Override via YAML
# vllm_feeder.circuit_break_threshold or set 0/negative to disable.
_DEFAULT_CIRCUIT_BREAK_THRESHOLD = 50


# ---------------------------------------------------------------------------
# JSONL helpers — orjson is in the cluster image; we assert it's available
# so we don't carry a two-path write loop. If a future env lacks orjson,
# install it (it's a hard dep, not a soft optimization).
# ---------------------------------------------------------------------------

import orjson  # noqa: E402

def _json_dumps(obj: dict) -> bytes:
    return orjson.dumps(obj)


def _load_resume_state(jsonl_path: Path) -> tuple[set[str], float]:
    """Scan the per-rank JSONL on resume — returns (seen_cut_ids, audio_committed_s).

    JSONL is the source of truth for resume: it has every committed row
    with both ``cut_id`` and ``duration``. One pass gives us:
      - the set used by the producer to skip already-done work.
      - the total audio-seconds already committed in earlier runs, used
        as ``stats.resume_offset_s`` so the heartbeat's ``audio=`` line
        stays cumulative across restarts (instead of resetting to 0h).

    Tolerates SIGKILL-mid-write trailing garbage by stopping at the
    first JSON parse error — committed rows up to that point are kept.
    """
    if not jsonl_path.exists():
        return set(), 0.0
    seen: set[str] = set()
    total_s = 0.0
    with jsonl_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                row = orjson.loads(raw)
            except Exception:
                logger.warning("Malformed JSONL line in %s — stopping resume scan.", jsonl_path)
                break
            cid = row.get("cut_id")
            if cid:
                seen.add(cid)
                total_s += float(row.get("duration") or 0.0)
    return seen, total_s


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _fmt_dur(seconds: float) -> str:
    """Human duration: 1.5h / 45m / 30s."""
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    if seconds >= 60:
        return f"{seconds/60:.0f}m"
    return f"{seconds:.0f}s"


class _CircuitBreaker:
    """Aborts the run after N consecutive transient failures.

    Designed to catch "vLLM crashed/walltime'd mid-run". When the server
    dies, every cut hits exhausted-retry and returns None; without this,
    we'd burn through the remaining wall time writing no transcripts.
    ``threshold <= 0`` disables the breaker.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = int(threshold)
        self.consecutive_failures = 0
        self.tripped = False

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> bool:
        """Increment; return True iff the breaker just tripped."""
        if self.threshold <= 0:
            return False
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold and not self.tripped:
            self.tripped = True
            return True
        return False


class _Stats:
    def __init__(self, *, resume_offset_s: float = 0.0) -> None:
        self.t0 = time.monotonic()
        self.n_in = 0           # cuts pulled from shar reader (post-resume-skip)
        self.n_ok = 0
        self.n_perm_err = 0     # committed errors
        self.n_transient = 0    # NOT committed (rerun will retry)
        self.last_log_t = self.t0
        self.last_log_done = 0
        # Audio-time tracking, split into two parts so the display can
        # be cumulative (resume + this_run) while RTFx stays per-run:
        #   processed_seconds — committed in THIS run; drives RTFx.
        #   resume_offset_s   — already committed in earlier runs; seeded
        #                       at startup from the JSONL scan.
        self.processed_seconds = 0.0
        self.resume_offset_s = float(resume_offset_s)
        self.total_seconds = 0.0      # remaining-at-scan-time (set from env)

    def snapshot(self, *, advance_window: bool = True) -> str:
        """One-line status. ``advance_window=False`` for ad-hoc snapshots
        (heartbeat) that shouldn't reset the recent-rate window."""
        now = time.monotonic()
        wall = max(now - self.t0, 1e-6)
        done = self.n_ok + self.n_perm_err
        recent_dt = max(now - self.last_log_t, 1e-6)
        recent_rps = (done - self.last_log_done) / recent_dt
        if advance_window:
            self.last_log_t = now
            self.last_log_done = done

        rtfx = self.processed_seconds / wall      # this-run server-side rate
        cumulative_s = self.processed_seconds + self.resume_offset_s
        proc = _fmt_dur(cumulative_s)
        if self.total_seconds > 0:
            full_dataset_s = self.total_seconds + self.resume_offset_s
            remaining_s   = max(self.total_seconds - self.processed_seconds, 0.0)
            eta = _fmt_dur(remaining_s / rtfx) if rtfx > 0 and remaining_s > 0 else "--"
            audio_str = f"audio={proc}/{_fmt_dur(full_dataset_s)}  ETA={eta}"
        else:
            audio_str = f"audio={proc}  ETA=--"

        return (
            f"{audio_str}  RTFx={rtfx:.1f}×  "
            f"ok={self.n_ok} perm_err={self.n_perm_err} "
            f"wall={_fmt_dur(wall)} rate={recent_rps:.1f}/s"
        )


# ---------------------------------------------------------------------------
# The feeder
# ---------------------------------------------------------------------------


async def _run_feeder(cfg: dict) -> dict:
    feeder_cfg = cfg.get("vllm_feeder") or {}
    # Env overrides — same precedence as launch.sh: env > YAML.
    #   VLLM_API_BASE: comma-list of vLLM URLs (set by launch.sh, or by hand)
    #   VLLM_MODEL:    served-model-name (set by launch.sh after ${USER}
    #                  expansion, so the request's "model" field matches what
    #                  the server registered as --served-model-name).
    # If you run ``python -m asr_vllm.main`` directly without launch.sh,
    # either pre-expand ${USER} in the YAML or export VLLM_MODEL by hand.
    env_base = os.environ.get("VLLM_API_BASE")
    if env_base:
        feeder_cfg = {**feeder_cfg, "api_base": env_base}
    env_model = os.environ.get("VLLM_MODEL")
    if env_model:
        feeder_cfg = {**feeder_cfg, "api_model": env_model}

    backend = build_backend(feeder_cfg)
    slot = backend.slot_name

    # Layout: ``<output_dir>/asr_plain_output/<slot>/<rank-files>``.
    output_dir = Path(cfg["output_dir"])
    out_subdir = output_dir / "quality_asr" / slot
    out_subdir.mkdir(parents=True, exist_ok=True)
    # Refuse to append to an output dir whose previous run used a
    # different cfg (e.g. swapped vLLM model, changed prompt template).
    # Set RESUME_ANYWAY=1 to override.
    guard_cfg_hash(
        out_subdir, cfg,
        allow_overwrite=bool(os.environ.get("RESUME_ANYWAY")),
    )
    # Single-process feeder — rank 0 is the only rank. Keep the
    # ``_NNNN`` naming so the offline joiner can use the same glob it
    # uses for parakeet/canary outputs.
    jsonl_path = out_subdir / f"{slot}_rank_0000.jsonl"
    seen, resume_offset_s = _load_resume_state(jsonl_path)
    logger.info(
        "feeder boot: slot=%s output=%s resume=%d cuts (%s already committed)",
        slot, jsonl_path, len(seen), _fmt_dur(resume_offset_s),
    )

    # Concurrency: prefer per-replica × num_replicas so scaling the
    # vLLM allocation from 2 → 4 workers doubles in-flight requests
    # automatically (matches the convention in the legacy fused
    # coordinator at coordinator.py:124-127). Falls back to absolute
    # api_concurrency when per-replica isn't set.
    api_base_for_count = (feeder_cfg.get("api_base") or "")
    n_replicas = max(1, len([b for b in api_base_for_count.split(",") if b.strip()]))
    per_replica = feeder_cfg.get("api_concurrency_per_replica")
    if per_replica is not None:
        concurrency = int(per_replica) * n_replicas
        logger.info(
            "concurrency: %d = %d per_replica × %d replicas",
            concurrency, int(per_replica), n_replicas,
        )
    else:
        concurrency = int(feeder_cfg.get("api_concurrency", _DEFAULT_CONCURRENCY))
        logger.info(
            "concurrency: %d (absolute; %d replicas → %.1f per replica)",
            concurrency, n_replicas, concurrency / n_replicas,
        )
    timeout = float(feeder_cfg.get("api_timeout", _DEFAULT_TIMEOUT_S))
    max_retries = int(feeder_cfg.get("api_max_retries", _DEFAULT_MAX_RETRIES))
    log_every = int(feeder_cfg.get("log_every", _DEFAULT_LOG_EVERY))
    heartbeat_s = float(feeder_cfg.get("heartbeat_seconds", _DEFAULT_HEARTBEAT_S))
    work_queue_size = int(feeder_cfg.get("work_queue_size", _DEFAULT_WORK_QUEUE))
    circuit = _CircuitBreaker(
        int(feeder_cfg.get("circuit_break_threshold", _DEFAULT_CIRCUIT_BREAK_THRESHOLD))
    )

    loader_cfg = cfg.get("loader") or {}
    reader_workers = int(loader_cfg.get("num_workers", _DEFAULT_SHARD_READERS))
    reader_qsize = int(loader_cfg.get("queue_size", _DEFAULT_READER_QUEUE))
    min_dur = loader_cfg.get("min_duration")
    max_dur = loader_cfg.get("max_duration")

    # asyncio.Queue: producer pushes CutItems; consumers pull. Bounded so
    # we don't pile up megabytes of FLAC blobs while consumers wait on HTTP.
    work_q: "asyncio.Queue[Optional[CutItem]]" = asyncio.Queue(maxsize=work_queue_size)
    # All-results funnel for the single writer task. Bounded by concurrency
    # × 2 so consumers can hand off without lock contention.
    write_q: "asyncio.Queue[Optional[dict]]" = asyncio.Queue(maxsize=concurrency * 2)

    stop_event = asyncio.Event()
    stats = _Stats(resume_offset_s=resume_offset_s)

    # Total dataset duration — pre-computed by launch.sh's parallel scan
    # and passed via env var. If absent (user ran ``python -m
    # asr_vllm.main`` directly without launch.sh), we just skip ETA and
    # only show cumulative hours processed.
    total_s_env = os.environ.get("ASR_VLLM_TOTAL_SECONDS")
    if total_s_env:
        try:
            stats.total_seconds = float(total_s_env)
            stats.scan_complete = True
            stats.scan_progress = 1.0
            logger.info(
                "Using pre-scanned total: %.1fh of unseen audio (from ASR_VLLM_TOTAL_SECONDS).",
                stats.total_seconds / 3600.0,
            )
        except ValueError:
            logger.warning("ASR_VLLM_TOTAL_SECONDS=%r invalid — ETA disabled.", total_s_env)

    logger.info(
        "─" * 72 + "\n"
        "  asr_vllm feeder\n"
        "  backend:     %s\n"
        "  api_base:    %s\n"
        "  api_model:   %s\n"
        "  shar_dir:    %s\n"
        "  output:      %s\n"
        "  concurrency: %d  (per_replica=%s × replicas=%d)\n"
        "  readers:     %d shards × queue=%d\n"
        "  resume:      %d cuts (%s already committed)\n"
        "  heartbeat:   every %.1fs\n"
        + "─" * 72,
        slot, feeder_cfg.get("api_base"), feeder_cfg.get("api_model"),
        cfg["shar_dir"], jsonl_path,
        concurrency, str(per_replica), n_replicas,
        reader_workers, reader_qsize,
        len(seen), _fmt_dur(resume_offset_s), heartbeat_s,
    )

    # ----------------------------------------------------------------
    # Producer — pulls from the (threaded) shar reader into work_q.
    # ----------------------------------------------------------------
    async def producer() -> None:
        loop = asyncio.get_running_loop()

        def _iter_blocking():
            return iter_shar_cuts(
                cfg["shar_dir"],
                num_workers=reader_workers,
                queue_size=reader_qsize,
                min_duration=min_dur,
                max_duration=max_dur,
            )

        gen = await loop.run_in_executor(None, _iter_blocking)
        stopped = False
        try:
            while not stop_event.is_set():
                item = await loop.run_in_executor(None, _next_or_none, gen)
                if item is None:
                    break
                if item.cut_id in seen:
                    continue
                stats.n_in += 1
                # Cooperative put: 1s timeout so we wake on stop_event when
                # work_q is full (consumers stop draining on Ctrl+C).
                while True:
                    try:
                        await asyncio.wait_for(work_q.put(item), timeout=1.0)
                        break
                    except asyncio.TimeoutError:
                        if stop_event.is_set():
                            stopped = True
                            break
                if stopped:
                    break
        finally:
            # Normal end-of-shar: signal EOF so consumers exit cleanly.
            # On stop_event-driven shutdown: consumers wake on their own
            # stop check, no EOF needed (and pushing into a full work_q
            # would deadlock here).
            if not stop_event.is_set():
                for _ in range(concurrency):
                    await work_q.put(None)
            logger.info("producer done: pushed %d cuts", stats.n_in)

    def _next_or_none(it):
        try:
            return next(it)
        except StopIteration:
            return None

    # ----------------------------------------------------------------
    # Consumers — one HTTP call each, with retry+backoff.
    # ----------------------------------------------------------------
    async def consumer(client: httpx.AsyncClient, worker_id: int) -> None:
        # Failure semantics:
        #   - TransientError exhausted → _transcribe_with_retry returns None.
        #     We log, count it, skip the cut. Sidecar stays untouched so
        #     a rerun picks it up. Single-cut failures don't kill the run.
        #   - Any other exception (backend bug, unexpected httpx error,
        #     OOM, etc.) → propagates up. Consumer task fails, asyncio.gather
        #     surfaces the exception, the whole feeder crashes. Big problems
        #     stop the run so they get noticed and fixed.
        while True:
            # Cooperative get: 1s timeout so we wake on stop_event when
            # work_q is empty (producer hits EOF or also stopped). Polling
            # overhead is negligible — only fires when actually blocked.
            try:
                item = await asyncio.wait_for(work_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if stop_event.is_set():
                    return
                continue
            if item is None:
                return
            result = await _transcribe_with_retry(
                client, backend, item, max_retries=max_retries,
            )
            if result is None:
                stats.n_transient += 1
                if circuit.record_failure():
                    # vLLM almost certainly dead — stop the bleed. We
                    # raise so asyncio.gather(consumers) surfaces it and
                    # the main loop exits with a non-zero status instead
                    # of silently chewing through the remaining shar.
                    stop_event.set()
                    raise RuntimeError(
                        f"circuit breaker tripped: {circuit.threshold} consecutive "
                        f"transient failures (vLLM likely down). Aborting."
                    )
                continue
            circuit.record_success()
            await write_q.put({
                "cut_id":   item.cut_id,
                "duration": round(item.duration, 3),
                "ref_text": item.ref_text,
                "speaker":  item.speaker,
                "language_hint": item.language,
                **result,
            })

    # ----------------------------------------------------------------
    # Writer — single task owns the file handles; no lock needed.
    # ----------------------------------------------------------------
    async def writer() -> None:
        flush_every = int(feeder_cfg.get("flush_every", 500))
        n_since_flush = 0
        f_out = jsonl_path.open("ab")
        try:
            n_writers_done = 0
            while True:
                row = await write_q.get()
                if row is None:
                    n_writers_done += 1
                    if n_writers_done >= concurrency:
                        break
                    continue
                # Every row reaching the writer is committable. Transient
                # failures were filtered upstream in the consumer; what
                # remains is either a successful transcript or a permanent
                # error (4xx / bad_json / no_audio_blob) that we record so
                # reruns don't retry.
                f_out.write(_json_dumps(row)); f_out.write(b"\n")
                if row.get("error") is None:
                    stats.n_ok += 1
                else:
                    stats.n_perm_err += 1
                # Audio duration drives RTFx + ETA. Count permanent-error
                # rows too — they consumed wall time even if no transcript.
                stats.processed_seconds += float(row.get("duration") or 0.0)
                n_since_flush += 1
                if n_since_flush >= flush_every:
                    f_out.flush()
                    n_since_flush = 0
                done = stats.n_ok + stats.n_perm_err
                if done % log_every == 0:
                    logger.info("WRITER  %s", stats.snapshot())
        finally:
            f_out.flush()
            f_out.close()

    # ----------------------------------------------------------------
    # Heartbeat — wall-clock cadence so "I'm alive" lands in the terminal
    # even when throughput is low (e.g. vLLM still warming up, or one of
    # 4 replicas is slow). Independent of log_every (per-N-cut cadence).
    # ----------------------------------------------------------------
    async def heartbeat() -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=heartbeat_s)
                return  # stop_event fired
            except asyncio.TimeoutError:
                pass
            # advance_window=False so heartbeat doesn't clobber log_every's rate window.
            logger.info("HEARTBEAT  %s", stats.snapshot(advance_window=False))

    # ----------------------------------------------------------------
    # Spin up everything.
    # ----------------------------------------------------------------
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout_cfg = httpx.Timeout(timeout, connect=min(30.0, timeout))
    async with httpx.AsyncClient(timeout=timeout_cfg, limits=limits) as client:
        prod_task = asyncio.create_task(producer(), name="producer")
        cons_tasks = [
            asyncio.create_task(consumer(client, i), name=f"consumer_{i}")
            for i in range(concurrency)
        ]
        writer_task = asyncio.create_task(writer(), name="writer")
        hb_task = asyncio.create_task(heartbeat(), name="heartbeat")

        # SIGTERM → set stop_event; consumers finish in-flight HTTP and exit.
        def _on_signal(signum: int) -> None:
            logger.warning("Received signal %d — shutting down.", signum)
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _on_signal, sig)
            except (NotImplementedError, RuntimeError):
                pass

        try:
            await prod_task
            await asyncio.gather(*cons_tasks)
        finally:
            # Tell the writer "no more rows".
            for _ in range(concurrency):
                await write_q.put(None)
            await writer_task
            # Stop the heartbeat last (after final stats are accurate).
            stop_event.set()
            try:
                await asyncio.wait_for(hb_task, timeout=heartbeat_s + 1.0)
            except asyncio.TimeoutError:
                hb_task.cancel()

    logger.info("feeder done: %s", stats.snapshot())
    return {
        "n_in":         stats.n_in,
        "n_ok":         stats.n_ok,
        "n_perm_err":   stats.n_perm_err,
        "n_transient":  stats.n_transient,
        "wall_seconds": time.monotonic() - stats.t0,
        "output":       str(jsonl_path),
    }


async def _transcribe_with_retry(
    client: httpx.AsyncClient,
    backend,
    item: CutItem,
    *,
    max_retries: int,
) -> Optional[dict]:
    """Retry on TransientError with exponential backoff.

    Returns:
      - the backend's dict (success OR permanent error like 4xx/bad_json)
      - ``None`` if all retries exhausted on transient errors (caller skips
        this cut; sidecar stays untouched; rerun will retry).

    Any other exception (backend code bug, unhandled httpx error, OOM …)
    propagates. We do NOT swallow them — one cut crashing the run is
    better than silently mis-committing thousands.
    """
    last_err: Optional[TransientError] = None
    for attempt in range(max_retries + 1):
        try:
            return await backend.transcribe(
                client, item.flac_bytes, item.language,
                cut_id=item.cut_id, ext=item.ext,
            )
        except TransientError as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(_DEFAULT_RETRY_BACKOFF_S * (2 ** attempt))
    logger.warning(
        "transient failure exhausted for cut_id=%s (will retry on rerun): %s",
        item.cut_id, last_err,
    )
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    # httpx logs one INFO line per request — at 128 concurrent that drowns
    # the heartbeat. Lift it to WARNING so only failures surface.
    # Override with LOG_LEVEL_HTTPX=DEBUG for HTTP debugging.
    httpx_level = os.environ.get("LOG_LEVEL_HTTPX", "WARNING").upper()
    logging.getLogger("httpx").setLevel(httpx_level)
    logging.getLogger("httpcore").setLevel(httpx_level)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM feeder")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    _setup_logging()
    cfg = _load_config(args.config)
    if not cfg.get("shar_dir"):
        raise ValueError("config.shar_dir is required")
    if not cfg.get("output_dir"):
        raise ValueError("config.output_dir is required")

    ctx = start_run(
        stage="qwen", yaml_path=args.config,
        rank=0, world_size=1,
        output_dir=cfg.get("output_dir"),
    )
    try:
        stats = asyncio.run(_run_feeder(cfg))
    except KeyboardInterrupt:
        end_run(ctx, status="terminated", reason="KeyboardInterrupt")
        raise
    except Exception as e:
        import traceback
        end_run(
            ctx, status="crashed",
            reason=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc()[-1500:],
        )
        raise

    logger.info("EXIT_STATS %s", json.dumps(stats))
    extras = {k: stats[k] for k in (
        "n_ok", "n_in", "n_perm_err", "n_transient", "wall_seconds", "output",
    ) if k in stats}
    end_run(ctx, status="ok", **extras)

    # Non-zero if every request failed permanently — likely misconfigured.
    if stats["n_in"] > 0 and stats["n_ok"] == 0 and stats["n_perm_err"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
