"""Standalone Parakeet driver — per-rank GPU loop with optional Frame-VAD.

Threading model: loader (Shar + bucketing) -> main thread (VAD -> Parakeet,
sequential CUDA) -> bounded queue -> writer thread (JSONL + sidecar); a heartbeat
thread logs liveness every N s. SIGTERM/SIGINT sets ``stop_event``: the GPU loop
breaks at the next batch boundary and the writer drains before exit.

Multilang: outer loop over (lang, split) pairs; loader, Parakeet and VAD models
are reloaded per language to bound host RSS.

Resume is JSONL-based: a startup pass yields (seen_cut_ids, audio_committed_s).
Failure semantics: NeMo handles one-cut transient errors (OOM split-retry,
silent-clip fallback); fatal CUDA / model-load errors crash the rank.

Run::
    sbatch asr_parakeet/scripts/submit.slurm
    # local: RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python -m asr_parakeet.main --config <cfg>
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# tqdm silencer — NeMo's `model.transcribe()` shows a "Transcribing: N/N"
# progress bar even with verbose=False (known issue: some AED/multitask
# decode paths build their own tqdm internally and ignore the kwarg).
# Replacing tqdm.tqdm with a subclass that pins disable=True kills every
# NeMo progress bar in one shot. Must happen BEFORE any NeMo import so
# the NeMo modules pick up our patched class at their first `from tqdm
# import tqdm`. Triggered only when TQDM_DISABLE=1 (set in submit.slurm).
# ---------------------------------------------------------------------------
import tqdm as _tqdm_mod  # noqa: E402
if os.environ.get("TQDM_DISABLE"):
    _orig_tqdm_cls = _tqdm_mod.tqdm

    class _DisabledTqdm(_orig_tqdm_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    _tqdm_mod.tqdm = _DisabledTqdm  # type: ignore[misc]

import numpy as np
import orjson
import torch
import yaml

from common.loader import SharAudioLoader
from common.run_history import end_run, guard_cfg_hash, start_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — keep next to the YAML keys they back so they're easy to find.
# ---------------------------------------------------------------------------

_DEFAULT_HEARTBEAT_S = 5.0     # wall-clock cadence — the only snapshot logger


# Module-level worker cache: key (kind, cfg_hash) → loaded worker. Lets
# the multilang loop SKIP reloading Parakeet/VAD between languages when
# their cfg sub-blocks are identical (Parakeet TDT v3 is a single
# multilingual model — same cfg across all langs, so we load once and
# reuse). Saves ~30s × n_langs of NeMo load per rank in multilang runs.
_WORKER_CACHE: dict = {}


def _cache_get_or_build(kind: str, cfg_block: dict, build_fn):
    """Return a cached worker for ``(kind, hash(cfg_block))``, or build
    and cache one. ``build_fn(cfg_block)`` is called only on cache miss."""
    import orjson as _oj
    key = (kind, _oj.dumps(cfg_block or {}, option=_oj.OPT_SORT_KEYS).decode())
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        logger.info("worker cache hit: %s", kind)
        return cached
    worker = build_fn(cfg_block)
    _WORKER_CACHE[key] = worker
    return worker
_DEFAULT_FLUSH_EVERY = 100     # flush JSONL+sidecar to disk every N rows
_DEFAULT_WRITER_QUEUE = 1024   # bounded write_q to limit RAM if disk slows
_DEFAULT_DECODE_NUM_WORKERS = 16
_DEFAULT_PREFETCH_FACTOR = 4
_DEFAULT_DATALOADER_TIMEOUT = 300


# ---------------------------------------------------------------------------
# JSONL I/O — orjson is a hard dep (asserted at import). Single write path,
# matches asr_vllm so the offline ``asr_join`` reads both with one parser.
# ---------------------------------------------------------------------------


def _json_dumps(obj: dict) -> bytes:
    return orjson.dumps(obj)


def _load_resume_state(jsonl_path: Path) -> tuple[set[str], float]:
    """JSONL-based resume: returns (seen_cut_ids, audio_committed_s).

    Each row carries cut_id+duration; one pass gives both. Stops at the
    first parse error so a SIGKILL-mid-write trailing garbage tail is
    tolerated — committed rows up to that point are kept.
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
                logger.warning(
                    "Malformed JSONL line in %s — stopping resume scan.", jsonl_path,
                )
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
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    if seconds >= 60:
        return f"{seconds/60:.0f}m"
    return f"{seconds:.0f}s"


class _Stats:
    """Tracks throughput for the snapshot line.

    ``processed_seconds`` is THIS run only (drives RTFx). ``resume_offset_s``
    is what was already committed by previous runs — added to processed in
    the display so ``audio=`` stays continuous across restarts.
    """

    def __init__(self, *, resume_offset_s: float = 0.0) -> None:
        self.t0 = time.monotonic()
        self.n_in = 0          # cuts pulled from loader (post-resume-skip)
        self.n_ok = 0          # parakeet hyp written
        self.n_silent = 0      # short-circuited by VAD (empty hyp written)
        self.last_log_t = self.t0
        self.last_log_done = 0
        self.processed_seconds = 0.0
        self.resume_offset_s = float(resume_offset_s)

    def snapshot(self, *, advance_window: bool = True) -> str:
        now = time.monotonic()
        wall = max(now - self.t0, 1e-6)
        done = self.n_ok + self.n_silent
        recent_dt = max(now - self.last_log_t, 1e-6)
        recent_rps = (done - self.last_log_done) / recent_dt
        if advance_window:
            self.last_log_t = now
            self.last_log_done = done
        rtfx = self.processed_seconds / wall
        cumulative_s = self.processed_seconds + self.resume_offset_s
        return (
            f"audio={_fmt_dur(cumulative_s)}  "
            f"RTFx={rtfx:.1f}×  "
            f"ok={self.n_ok} silent={self.n_silent} "
            f"wall={_fmt_dur(wall)} rate={recent_rps:.1f}/s"
        )


# ---------------------------------------------------------------------------
# Per-rank inference loop
# ---------------------------------------------------------------------------


def _run_one(
    cfg: dict,
    rank: int,
    world_size: int,
    local_rank: int,
) -> dict:
    """Run Parakeet on one (lang, split) for this rank — returns stats dict.

    Three threads: main (GPU loop), writer (JSONL+sidecar), heartbeat.
    Same SIGTERM/snapshot model as asr_vllm.
    """
    from .workers import (
        ParakeetWorker, VadWorker,
        VAD_DEFAULT_MIN_SILENCE_MS, VAD_DEFAULT_MIN_SPEECH_MS, VAD_DEFAULT_THRESHOLD,
    )
    # NeMo writes ``[NeMo W ...]`` warnings straight to stderr via its own
    # logger object (not Python ``logging``), so taming
    # logging.getLogger("nemo") doesn't catch them. Silence here, after the
    # workers import has triggered NeMo load (else the import isn't found).
    from nemo.utils import logging as _nemo_logging
    _nemo_logging.setLevel("ERROR")

    device = torch.device(f"cuda:{local_rank}")

    # ----- output paths -------------------------------------------------
    # Layout: ``<output_dir>/asr_plain_output/parakeet/<rank-files>``. The
    # ``asr_plain_output/`` layer separates raw per-slot transcripts from
    # the merged ``asr_agg/rover/`` output written later by ``asr_join``.
    output_dir = Path(cfg["output_dir"])
    out_subdir = output_dir / "quality_asr" / "parakeet"
    out_subdir.mkdir(parents=True, exist_ok=True)
    # Refuse to append to an output dir whose previous run used a
    # different cfg (different model / different micro_batch / different
    # vad threshold...). Set RESUME_ANYWAY=1 to override.
    if rank == 0:
        guard_cfg_hash(
            out_subdir, cfg,
            allow_overwrite=bool(os.environ.get("RESUME_ANYWAY")),
        )
    jsonl_path = out_subdir / f"parakeet_rank_{rank:04d}.jsonl"
    seen, resume_offset_s = _load_resume_state(jsonl_path)

    # ----- VAD knobs ----------------------------------------------------
    vad_cfg = cfg.get("vad") or {}
    vad_enabled = bool(vad_cfg.get("enabled", True))
    vad_threshold = float(vad_cfg.get("threshold", VAD_DEFAULT_THRESHOLD))
    vad_min_speech_ms = int(vad_cfg.get("min_speech_ms", VAD_DEFAULT_MIN_SPEECH_MS))
    vad_min_silence_ms = int(vad_cfg.get("min_silence_ms", VAD_DEFAULT_MIN_SILENCE_MS))
    vad_skip_below = vad_cfg.get("skip_if_below")  # None or float

    # ----- model loads (cached across multilang iterations) -------------
    parakeet = _cache_get_or_build(
        "parakeet", cfg.get("parakeet") or {},
        lambda c: ParakeetWorker(c, device),
    )
    vad: Optional["VadWorker"] = None
    if vad_enabled:
        vad = _cache_get_or_build("vad", vad_cfg, lambda c: VadWorker(c, device))

    # ----- loader -------------------------------------------------------
    loader_cfg = dict(cfg.get("loader") or {})
    loader_cfg.setdefault("shar_dir", cfg["shar_dir"])
    loader_cfg.setdefault("target_sample_rate", cfg.get("target_sample_rate", 16000))
    loader = SharAudioLoader(
        loader_cfg,
        rank=rank, world_size=world_size,
        num_workers=int(loader_cfg.get("decode_num_workers", _DEFAULT_DECODE_NUM_WORKERS)),
        prefetch_factor=int(loader_cfg.get("prefetch_factor", _DEFAULT_PREFETCH_FACTOR)),
        dataloader_timeout=int(loader_cfg.get("dataloader_timeout", _DEFAULT_DATALOADER_TIMEOUT)),
    )

    # ----- writer + heartbeat knobs -------------------------------------
    runner_cfg = cfg.get("runner") or {}
    heartbeat_s = float(runner_cfg.get("heartbeat_seconds", _DEFAULT_HEARTBEAT_S))
    flush_every = int(runner_cfg.get("flush_every", _DEFAULT_FLUSH_EVERY))
    writer_qsize = int(runner_cfg.get("writer_queue_size", _DEFAULT_WRITER_QUEUE))

    write_q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=writer_qsize)
    stop_event = threading.Event()
    stats = _Stats(resume_offset_s=resume_offset_s)

    # ----------------------------------------------------------------
    # Writer thread — single owner of the file handles, no lock needed.
    # ----------------------------------------------------------------
    def _writer_run() -> None:
        n_since_flush = 0
        f_out = jsonl_path.open("ab")
        try:
            while True:
                row = write_q.get()
                if row is None:
                    break
                f_out.write(_json_dumps(row)); f_out.write(b"\n")
                if row.get("vad_short_circuit"):
                    stats.n_silent += 1
                else:
                    stats.n_ok += 1
                # Audio drives RTFx. Count short-circuited cuts too — they
                # still consumed wall time (VAD pass + bookkeeping).
                stats.processed_seconds += float(row.get("duration") or 0.0)
                n_since_flush += 1
                if n_since_flush >= flush_every:
                    f_out.flush()
                    n_since_flush = 0
        finally:
            f_out.flush()
            f_out.close()

    # ----------------------------------------------------------------
    # Cross-rank aggregate (rank 0 only) — reads sibling per-rank JSONLs
    # incrementally (cached file positions) and sums committed cuts +
    # audio seconds across all 4 ranks. One extra ``HEARTBEAT_AGG`` line
    # per heartbeat. Scoped to this _run_one call so positions reset
    # cleanly between language iterations.
    #
    # IMPORTANT: at startup we pre-seed each file's position to current
    # EOF and capture the at-startup totals separately, so the heartbeat
    # delta (rate / RTFx) reflects ONLY work done in this run, while the
    # cumulative display (total_ok / total_audio) includes resume rows.
    # Without this, an existing JSONL of e.g. 380k rows yielded
    # agg_rate=380000/wall_s_so_far on the very first heartbeat.
    # ----------------------------------------------------------------
    agg_positions: dict = {}
    agg_delta = {"n": 0, "s": 0.0}
    agg_start = {"n": 0, "s": 0.0}

    def _scan_jsonl(jp: Path, start_pos: int) -> tuple[int, int, float]:
        """Read [start_pos, last_newline+1) of jp; return (new_pos, n, sum_s)."""
        try:
            with jp.open("rb") as f:
                f.seek(start_pos)
                buf = f.read()
        except FileNotFoundError:
            return start_pos, 0, 0.0
        last_nl = buf.rfind(b"\n")
        if last_nl < 0:
            return start_pos, 0, 0.0
        n = 0
        s = 0.0
        for raw in buf[:last_nl + 1].splitlines():
            if not raw.strip():
                continue
            try:
                row = orjson.loads(raw)
            except Exception:
                continue
            n += 1
            s += float(row.get("duration") or 0.0)
        return start_pos + last_nl + 1, n, s

    def _seed_aggregate_baseline() -> None:
        """Rank-0 only: at startup, count all sibling JSONL content and
        park each file's position at current EOF. Subsequent heartbeats
        only see rows written AFTER this snapshot."""
        for jp in sorted(out_subdir.glob("parakeet_rank_*.jsonl")):
            new_pos, n, s = _scan_jsonl(jp, 0)
            agg_positions[str(jp)] = new_pos
            agg_start["n"] += n
            agg_start["s"] += s

    def _read_aggregate_delta() -> tuple[int, float]:
        """Tail every sibling JSONL since last call; return THIS-RUN delta."""
        for jp in sorted(out_subdir.glob("parakeet_rank_*.jsonl")):
            key = str(jp)
            pos = agg_positions.get(key, 0)
            new_pos, n, s = _scan_jsonl(jp, pos)
            agg_positions[key] = new_pos
            agg_delta["n"] += n
            agg_delta["s"] += s
        return agg_delta["n"], agg_delta["s"]

    if rank == 0:
        _seed_aggregate_baseline()
        logger.info(
            "rank-0 aggregate baseline: %d cuts (%s) already committed "
            "across all ranks before this run; deltas only from here on.",
            agg_start["n"], _fmt_dur(agg_start["s"]),
        )

    # ----------------------------------------------------------------
    # Heartbeat thread — wall-clock cadence, only snapshot logger.
    # Rank 0 emits an extra ``HEARTBEAT_AGG`` line summing cuts + audio
    # across all ranks. Layout matches per-rank semantics: cumulative
    # totals include the resume baseline, rate/RTFx use this-run delta.
    # ----------------------------------------------------------------
    def _heartbeat_run() -> None:
        while not stop_event.is_set():
            if stop_event.wait(timeout=heartbeat_s):
                return
            logger.info("HEARTBEAT  %s", stats.snapshot(advance_window=True))
            if rank == 0:
                d_n, d_s = _read_aggregate_delta()
                wall = max(time.monotonic() - stats.t0, 1e-6)
                cum_n = agg_start["n"] + d_n
                cum_s = agg_start["s"] + d_s
                logger.info(
                    "HEARTBEAT_AGG  total_ok=%d  total_audio=%s  "
                    "this_run=+%d (+%s)  agg_rate=%.1f/s  RTFx=%.0f×",
                    cum_n, _fmt_dur(cum_s), d_n, _fmt_dur(d_s),
                    d_n / wall, d_s / wall,
                )

    # SIGTERM/SIGINT → set stop_event, finish in-flight batch, exit.
    def _on_signal(signum, _frame):
        if not stop_event.is_set():
            logger.warning(
                "Received %s — finishing current batch and exiting.",
                signal.Signals(signum).name,
            )
        stop_event.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    writer_thread = threading.Thread(
        target=_writer_run, name="parakeet_writer", daemon=False,
    )
    writer_thread.start()
    hb_thread = threading.Thread(
        target=_heartbeat_run, name="parakeet_heartbeat", daemon=True,
    )
    hb_thread.start()

    # Banner: rank 0 prints the full multi-line config; the other ranks
    # emit a one-line "starting" so you can confirm they came up without
    # cluttering the log with N copies of the same config block.
    if rank == 0:
        logger.info(
            "─" * 72 + "\n"
            "  asr_parakeet world_size=%d\n"
            "  shar_dir:    %s\n"
            "  output:      %s\n"
            "  parakeet:    %s\n"
            "  vad:         %s (threshold=%.2f)\n"
            "  resume(r0):  %d cuts (%s already committed)\n"
            "  heartbeat:   every %.1fs\n"
            + "─" * 72,
            world_size,
            cfg["shar_dir"], output_dir,
            (cfg.get("parakeet") or {}).get("model", "default"),
            "enabled" if vad_enabled else "disabled", vad_threshold,
            len(seen), _fmt_dur(resume_offset_s), heartbeat_s,
        )
    else:
        logger.info(
            "starting on %s — resume=%d cuts (%s already committed)",
            device, len(seen), _fmt_dur(resume_offset_s),
        )

    # ----------------------------------------------------------------
    # GPU loop — VAD → Parakeet → write_q.
    # ----------------------------------------------------------------
    try:
        for batch in loader:
            if stop_event.is_set():
                logger.warning("stop_event set — breaking loader iteration.")
                break

            # Skip cuts already committed by an earlier run (resume).
            new_idx = [
                i for i, cid in enumerate(batch.cut_ids) if cid not in seen
            ]
            if not new_idx:
                continue
            stats.n_in += len(new_idx)

            audio_np = batch.audio.numpy()
            lengths_np = batch.lengths.numpy()
            sub_lengths = [int(lengths_np[i]) for i in new_idx]
            audio_list = [
                audio_np[i, :n].astype(np.float32, copy=False)
                for i, n in zip(new_idx, sub_lengths)
            ]
            cuts = [batch.cuts[i] for i in new_idx]
            cut_ids = [batch.cut_ids[i] for i in new_idx]
            durations = [float(c.duration) for c in cuts]

            # VAD batch pass first — its result decides per-cut short-circuit.
            if vad is not None:
                vad_records = vad.run(
                    audio_list, batch.sr,
                    threshold=vad_threshold,
                    min_speech_ms=vad_min_speech_ms,
                    min_silence_ms=vad_min_silence_ms,
                )
            else:
                vad_records = [None] * len(audio_list)

            short_set: set[int] = set()
            run_idx: list[int] = []
            for i, vr in enumerate(vad_records):
                if (
                    vad_skip_below is not None
                    and vr is not None
                    and float(vr.get("speech_pct", 0.0)) < float(vad_skip_below)
                ):
                    short_set.add(i)
                else:
                    run_idx.append(i)

            # Parakeet: sort longest-first so OOM (if any) trips early; the
            # ParakeetWorker itself OOM-split-retries internally.
            order = sorted(run_idx, key=lambda i: -sub_lengths[i])
            ordered_audio = [audio_list[i] for i in order]
            parakeet_hyps = (
                parakeet.transcribe(ordered_audio) if ordered_audio else []
            )
            order_to_pos = {orig_i: pos for pos, orig_i in enumerate(order)}

            # Submit one row per cut (in source order).
            for i in range(len(new_idx)):
                sup = cuts[i].supervisions[0] if cuts[i].supervisions else None
                row: dict = {
                    "cut_id":   cut_ids[i],
                    "duration": round(durations[i], 3),
                    "ref_text": (sup.text or "") if sup else "",
                    "speaker":  (sup.speaker or "") if sup else "",
                    "language_hint": (sup.language or "") if sup else "",
                    "vad":      vad_records[i],
                }
                if i in short_set:
                    row["parakeet"] = {
                        "text": "", "avg_logp": 0.0, "word_timestamps": [],
                    }
                    row["vad_short_circuit"] = True
                else:
                    row["parakeet"] = parakeet_hyps[order_to_pos[i]]
                # Bounded put: 1s timeout so a stuck disk eventually shows
                # up as a stop_event check rather than silent backlog.
                while True:
                    try:
                        write_q.put(row, timeout=1.0)
                        break
                    except queue.Full:
                        if stop_event.is_set():
                            break
                seen.add(cut_ids[i])
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — shutting down cleanly.")
        stop_event.set()
    finally:
        # Tell the writer "no more rows" and wait for the JSONL to flush.
        write_q.put(None)
        writer_thread.join(timeout=120.0)
        if writer_thread.is_alive():
            logger.warning("Writer thread did not exit within 120s.")
        # Stop the heartbeat last — final stats are then accurate.
        stop_event.set()
        hb_thread.join(timeout=heartbeat_s + 1.0)

    logger.info("rank done: %s", stats.snapshot())
    return {
        "rank":         rank,
        "n_in":         stats.n_in,
        "n_ok":         stats.n_ok,
        "n_silent":     stats.n_silent,
        "wall_seconds": time.monotonic() - stats.t0,
        "output":       str(jsonl_path),
    }


# ---------------------------------------------------------------------------
# Multilang loop — same pattern as asr/main.py:_resolve_language_subdirs +
# _build_language_cfg, minus voxtral/canary/aligner overrides.
# ---------------------------------------------------------------------------


def _resolve_language_subdirs(
    root: Path,
    languages_override,
    subdir_template: str,
):
    """Pair each language to its Lhotse Shar subdir.

    Returns ``[(lang, shar_dir), ...]``. One bag per language — splits
    (validation / other / invalidated) are NOT iterated; the user's
    per-language ``shar_dir`` should be either a flat union shar or a
    unified directory containing every split's shards.
    """
    base_pairs = []
    if languages_override:
        for lang in languages_override:
            sub = root / subdir_template.format(lang=lang)
            if not sub.is_dir():
                raise FileNotFoundError(
                    f"language_split_dir: expected '{sub}' for lang={lang!r}",
                )
            base_pairs.append((str(lang), sub))
    else:
        if "{lang}" not in subdir_template:
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    base_pairs.append((child.name, child))
        else:
            prefix, _, suffix = subdir_template.partition("{lang}")
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                name = child.name
                if (
                    name.startswith(prefix)
                    and name.endswith(suffix)
                    and len(name) > len(prefix) + len(suffix)
                ):
                    end = -len(suffix) if suffix else None
                    lang = name[len(prefix):end]
                    base_pairs.append((lang, child))

        if not base_pairs:
            raise FileNotFoundError(
                f"language_split_dir: no subdirs of {root} match template "
                f"{subdir_template!r}",
            )

    return base_pairs


def _build_language_cfg(
    base_cfg: dict,
    lang: str,
    lang_shar_dir: Path,
    base_output_dir: Path,
) -> dict:
    """Per-language deep copy of the base cfg.

    Substitutions:
      - ``shar_dir`` → ``lang_shar_dir``
      - ``output_dir`` → ``<base>/<lang>``  (the ``parakeet/`` subdir is
        appended by ``_run_one``, giving ``<base>/<lang>/parakeet/...``).

    Parakeet-TDT v3 is a single multilingual model — no per-lang model
    substitution needed (compare to ``asr_canary/main.py`` which also
    rewrites ``canary.language`` and ``alignment.model``).
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["shar_dir"] = str(lang_shar_dir)
    cfg["output_dir"] = str(base_output_dir / lang)
    return cfg


_END_EXTRA_KEYS = (
    "n_ok", "n_in", "n_silent", "wall_seconds", "output",
)


def _tracked_run(
    cfg: dict,
    rank: int,
    world_size: int,
    local_rank: int,
    *,
    yaml_path: str,
    language: Optional[str] = None,
) -> dict:
    """Wrap ``_run_one`` with run_history start/end events.

    Emits ``status=ok`` on normal return (even partial — SIGTERM exits
    cleanly through ``stop_event``, returning a low-n_committed stats
    dict). Emits ``status=terminated`` on KeyboardInterrupt and
    ``status=crashed`` on any other exception, with a truncated
    traceback in ``reason``.
    """
    ctx = start_run(
        stage="parakeet", yaml_path=yaml_path,
        rank=rank, world_size=world_size,
        output_dir=cfg.get("output_dir"),
    )
    try:
        stats = _run_one(cfg, rank, world_size, local_rank)
    except KeyboardInterrupt:
        end_run(ctx, status="terminated", reason="KeyboardInterrupt", language=language)
        raise
    except Exception as e:
        import traceback
        end_run(
            ctx, status="crashed",
            reason=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc()[-1500:],
            language=language,
        )
        raise
    extras = {k: stats[k] for k in _END_EXTRA_KEYS if k in stats}
    end_run(ctx, status="ok", language=language, **extras)
    return stats


def _slurm_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    return rank, world_size, local_rank


def _setup_logging(rank: int, world_size: int) -> None:
    """Per-rank stderr handler + optional file handler under PIPELINE_LOG_DIR.

    Same shape as asr/main.py — the SLURM .err file (which srun merges all
    ranks into) plus an optional rank-tagged file under the run dir.
    """
    log_dir_env = os.environ.get("PIPELINE_LOG_DIR")
    fmt = f"%(asctime)s [r{rank}/{world_size}] %(levelname)s %(name)s: %(message)s"
    handlers: list = [logging.StreamHandler(sys.stderr)]
    if log_dir_env:
        log_dir = Path(log_dir_env)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / f"asr_parakeet_rank{rank:04d}.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)

    # Tame chatty libraries — they drown the heartbeat at high throughput.
    for name in ("nemo", "nemo_logger", "transformers", "lhotse",
                 "urllib3.connectionpool"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser(description="standalone Parakeet driver")
    ap.add_argument(
        "--config", required=True,
        help="YAML config (asr_parakeet/egs/*.yaml shape).",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())

    rank, world_size, local_rank = _slurm_env()
    _setup_logging(rank, world_size)
    torch.cuda.set_device(local_rank)
    logger.info(
        "asr_parakeet starting: rank %d/%d on GPU %d (%s)",
        rank, world_size, local_rank, torch.cuda.get_device_name(local_rank),
    )

    yaml_path = str(cfg_path)

    lsd = cfg.get("language_split_dir")
    if not lsd:
        # Single-shar mode — same path as the multilang loop's body.
        stats = _tracked_run(cfg, rank, world_size, local_rank, yaml_path=yaml_path)
        logger.info("asr_parakeet done: %s", stats)
        return 0

    root = Path(lsd)
    if not root.is_dir():
        print(f"language_split_dir not a directory: {root}", file=sys.stderr)
        return 1
    pairs = _resolve_language_subdirs(
        root,
        cfg.get("languages"),
        cfg.get("language_subdir_template", "{lang}"),
    )
    base_output_dir = Path(cfg.get("output_dir", "./results_asr_parakeet"))
    logger.info(
        "language_split_dir=%s — running %d language(s) sequentially: %s",
        root, len(pairs), [lang for lang, _ in pairs],
    )

    all_stats = []
    for lang, lang_shar_dir in pairs:
        logger.info("=== BEGIN %s shar=%s ===", lang, lang_shar_dir)
        lang_cfg = _build_language_cfg(cfg, lang, lang_shar_dir, base_output_dir)
        stats = _tracked_run(
            lang_cfg, rank, world_size, local_rank,
            yaml_path=yaml_path, language=lang,
        )
        stats["language"] = lang
        all_stats.append(stats)
        logger.info("=== END %s stats=%s ===", lang, stats)

        # NOTE: we deliberately do NOT gc.collect() / torch.cuda.empty_cache()
        # between iterations — workers are cached in _WORKER_CACHE and reused.
        # The DataLoader from SharAudioLoader IS released by Python's normal
        # refcount when _run_one returns; that's enough to bound RSS.

    logger.info("asr_parakeet done — %d run(s): %s", len(all_stats), all_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
