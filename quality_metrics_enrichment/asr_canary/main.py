"""Standalone Canary driver — per-rank GPU loop with inline NFA alignment.

Threading model (same as asr_parakeet): loader (Shar + bucketing) -> main thread
(VAD -> Canary -> optional NFA aligner, sequential CUDA) -> bounded queue ->
writer thread (JSONL + sidecar); a heartbeat thread logs liveness every N s.

Inline alignment reuses the audio tensors already on the GPU, so the offline
join pass needn't re-decode FLAC to recover word boundaries — at the cost of
running the few-GB NFA model sequentially after Canary.

Multilang: outer (lang, split) loop. ``canary.language`` is pinned per call
(Canary AED is prompted) and the per-language NFA checkpoint ``alignment.model``
is substituted; set ``alignment.enabled: false`` where no checkpoint exists.

Resume is JSONL-based: a startup pass yields (seen_cut_ids, audio_committed_s).
Failure semantics: workers self-recover from per-batch OOM and silent-clip
errors; an aligner crash drops only that batch's word_timestamps (text kept);
fatal CUDA / model-load errors crash the rank.

Run::
    sbatch asr_canary/scripts/submit.slurm
    # local: RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python -m asr_canary.main --config <cfg>
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
# tqdm silencer — see asr_parakeet/main.py for the rationale. NeMo's
# AED multitask path (Canary) ignores verbose=False in several internal
# tqdm() calls; replacing tqdm.tqdm with a disable=True subclass kills
# them. Must happen BEFORE any NeMo import. Triggered by TQDM_DISABLE=1.
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

_DEFAULT_HEARTBEAT_S = 5.0      # wall-clock cadence — the only snapshot logger


# Module-level worker cache. Same pattern as asr_parakeet:
#   - canary block: stable across multilang iterations (language is a
#     per-call kwarg to .transcribe, not a load-time param) → cached.
#   - vad block: also stable → cached.
#   - alignment block: model path includes {language}, so a new lang
#     means a new model. We evict the previous "aligner" cache entry
#     before loading the new one to free GPU memory.
_WORKER_CACHE: dict = {}


def _cache_get_or_build(kind: str, cfg_block: dict, build_fn, *, evict_kind: bool = False):
    """Return a cached worker for ``(kind, hash(cfg_block))``, building
    on miss. ``evict_kind=True`` drops any other entry of the same kind
    first — used for the aligner whose model differs per language."""
    import orjson as _oj
    key = (kind, _oj.dumps(cfg_block or {}, option=_oj.OPT_SORT_KEYS).decode())
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        logger.info("worker cache hit: %s", kind)
        return cached
    if evict_kind:
        for k in list(_WORKER_CACHE):
            if k[0] == kind:
                del _WORKER_CACHE[k]
        gc.collect()
        torch.cuda.empty_cache()
    worker = build_fn(cfg_block)
    _WORKER_CACHE[key] = worker
    return worker
_DEFAULT_FLUSH_EVERY = 100      # flush JSONL+sidecar every N rows
_DEFAULT_WRITER_QUEUE = 1024    # bounded write_q to limit RAM if disk slows
_DEFAULT_DECODE_NUM_WORKERS = 16
_DEFAULT_PREFETCH_FACTOR = 4
_DEFAULT_DATALOADER_TIMEOUT = 300


# ---------------------------------------------------------------------------
# JSONL I/O — orjson hard dep, single write path. Matches asr_parakeet so
# the offline asr_join reads both with one parser.
# ---------------------------------------------------------------------------


def _json_dumps(obj: dict) -> bytes:
    return orjson.dumps(obj)


def _load_resume_state(jsonl_path: Path) -> tuple[set[str], float]:
    """JSONL-based resume — same shape as asr_parakeet._load_resume_state.

    Returns (seen_cut_ids, audio_committed_s). Stops at the first parse
    error so a SIGKILL-mid-write tail is tolerated.
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
    def __init__(self, *, resume_offset_s: float = 0.0) -> None:
        self.t0 = time.monotonic()
        self.n_in = 0
        self.n_ok = 0
        self.n_silent = 0       # short-circuited by VAD
        self.n_aligned = 0      # rows whose word_timestamps come from NFA
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
            f"ok={self.n_ok} silent={self.n_silent} aligned={self.n_aligned} "
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
    """Run Canary (+ optional NFA) on one (lang, split) for this rank."""
    from .workers import (
        AlignmentWorker, CanaryWorker, VadWorker,
        VAD_DEFAULT_MIN_SILENCE_MS, VAD_DEFAULT_MIN_SPEECH_MS, VAD_DEFAULT_THRESHOLD,
    )
    # NeMo writes ``[NeMo W ...]`` warnings straight to stderr via its own
    # logger (not Python ``logging``), so taming logging.getLogger("nemo")
    # doesn't catch them. Silence here, after the workers import has
    # triggered NeMo load (else the module isn't found yet).
    from nemo.utils import logging as _nemo_logging
    _nemo_logging.setLevel("ERROR")

    device = torch.device(f"cuda:{local_rank}")

    # ----- output paths -------------------------------------------------
    # Layout: ``<output_dir>/asr_plain_output/canary/<rank-files>``.
    output_dir = Path(cfg["output_dir"])
    out_subdir = output_dir / "quality_asr" / "canary"
    out_subdir.mkdir(parents=True, exist_ok=True)
    # Refuse to append to an output dir whose previous run used a
    # different cfg (e.g. swapped Canary model, changed alignment).
    # Set RESUME_ANYWAY=1 to override.
    if rank == 0:
        guard_cfg_hash(
            out_subdir, cfg,
            allow_overwrite=bool(os.environ.get("RESUME_ANYWAY")),
        )
    jsonl_path = out_subdir / f"canary_rank_{rank:04d}.jsonl"
    seen, resume_offset_s = _load_resume_state(jsonl_path)

    # ----- VAD knobs ----------------------------------------------------
    vad_cfg = cfg.get("vad") or {}
    vad_enabled = bool(vad_cfg.get("enabled", True))
    vad_threshold = float(vad_cfg.get("threshold", VAD_DEFAULT_THRESHOLD))
    vad_min_speech_ms = int(vad_cfg.get("min_speech_ms", VAD_DEFAULT_MIN_SPEECH_MS))
    vad_min_silence_ms = int(vad_cfg.get("min_silence_ms", VAD_DEFAULT_MIN_SILENCE_MS))
    vad_skip_below = vad_cfg.get("skip_if_below")

    # ----- alignment knobs ----------------------------------------------
    align_cfg = cfg.get("alignment") or {}
    alignment_enabled = bool(align_cfg.get("enabled", False))

    # ----- model loads (cached across multilang iterations) -------------
    canary = _cache_get_or_build(
        "canary", cfg.get("canary") or {},
        lambda c: CanaryWorker(c, device),
    )
    vad: Optional["VadWorker"] = None
    if vad_enabled:
        vad = _cache_get_or_build("vad", vad_cfg, lambda c: VadWorker(c, device))
    aligner: Optional["AlignmentWorker"] = None
    if alignment_enabled:
        # evict_kind=True so switching language drops the old aligner
        # (different stt_<lang>_* checkpoint) before loading the new one.
        aligner = _cache_get_or_build(
            "aligner", align_cfg,
            lambda c: AlignmentWorker(c, device),
            evict_kind=True,
        )

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
    # Writer thread.
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
                if row.get("timestamp_source") == "nfa":
                    stats.n_aligned += 1
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
    # audio seconds across all ranks. One extra ``HEARTBEAT_AGG`` line
    # per heartbeat. Scoped to this _run_one call so positions reset
    # between (lang, split) iterations.
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
        for jp in sorted(out_subdir.glob("canary_rank_*.jsonl")):
            new_pos, n, s = _scan_jsonl(jp, 0)
            agg_positions[str(jp)] = new_pos
            agg_start["n"] += n
            agg_start["s"] += s

    def _read_aggregate_delta() -> tuple[int, float]:
        """Tail every sibling JSONL since last call; return THIS-RUN delta."""
        for jp in sorted(out_subdir.glob("canary_rank_*.jsonl")):
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
        target=_writer_run, name="canary_writer", daemon=False,
    )
    writer_thread.start()
    hb_thread = threading.Thread(
        target=_heartbeat_run, name="canary_heartbeat", daemon=True,
    )
    hb_thread.start()

    # Banner: rank 0 prints the full multi-line config; the other ranks
    # emit a one-line "starting" so you can confirm they came up without
    # cluttering the log with N copies of the same config block.
    if rank == 0:
        logger.info(
            "─" * 72 + "\n"
            "  asr_canary world_size=%d\n"
            "  shar_dir:    %s\n"
            "  output:      %s\n"
            "  canary:      %s  lang_cfg=%s\n"
            "  alignment:   %s\n"
            "  vad:         %s (threshold=%.2f)\n"
            "  resume(r0):  %d cuts (%s already committed)\n"
            "  heartbeat:   every %.1fs\n"
            + "─" * 72,
            world_size,
            cfg["shar_dir"], output_dir,
            (cfg.get("canary") or {}).get("model", "default"),
            (cfg.get("canary") or {}).get("language", "<per-cut>"),
            align_cfg.get("model") if alignment_enabled else "disabled",
            "enabled" if vad_enabled else "disabled", vad_threshold,
            len(seen), _fmt_dur(resume_offset_s), heartbeat_s,
        )
    else:
        logger.info(
            "starting on %s — resume=%d cuts (%s already committed)",
            device, len(seen), _fmt_dur(resume_offset_s),
        )

    # ----------------------------------------------------------------
    # GPU loop — VAD → Canary → optional NFA → write_q.
    # ----------------------------------------------------------------
    try:
        for batch in loader:
            if stop_event.is_set():
                logger.warning("stop_event set — breaking loader iteration.")
                break

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

            # Resolve per-cut language: Canary AED needs source/target_lang
            # as scalar prompt slots, so the worker buckets by language
            # internally — we just hand it the list.
            canary_langs = canary.resolve_languages(cuts)

            # VAD first — its result decides per-cut short-circuit.
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

            # Canary: sort longest-first so OOM (if any) trips early.
            order = sorted(run_idx, key=lambda i: -sub_lengths[i])
            ordered_audio = [audio_list[i] for i in order]
            ordered_langs = [canary_langs[i] for i in order]
            canary_hyps = (
                canary.transcribe(ordered_audio, ordered_langs)
                if ordered_audio else []
            )

            # Optional NFA — same audio buffer, no re-decode.
            aligned_words: Optional[list] = None
            if aligner is not None and ordered_audio:
                target_texts = [(h.get("text") or "").strip() for h in canary_hyps]
                try:
                    aligned_words = aligner.align(ordered_audio, target_texts)
                except Exception:
                    logger.exception(
                        "Aligner crashed on whole batch — keeping canary text, "
                        "emptying word_timestamps.",
                    )
                    aligned_words = [[] for _ in canary_hyps]
                # Overwrite Canary's CTC-aux word_timestamps with NFA ones.
                for h, words in zip(canary_hyps, aligned_words):
                    h["word_timestamps"] = words

            order_to_pos = {orig_i: pos for pos, orig_i in enumerate(order)}

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
                    row["canary"] = {
                        "text": "", "avg_logp": 0.0, "word_timestamps": [],
                    }
                    if canary_langs[i] is not None:
                        row["canary"]["language"] = canary_langs[i]
                    row["vad_short_circuit"] = True
                else:
                    row["canary"] = canary_hyps[order_to_pos[i]]
                    # Stamp where the word_timestamps came from so the join
                    # pass can prefer NFA when ROVER backs off to canary.
                    row["timestamp_source"] = "nfa" if aligner is not None else "ctc_aux"
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
        write_q.put(None)
        writer_thread.join(timeout=120.0)
        if writer_thread.is_alive():
            logger.warning("Writer thread did not exit within 120s.")
        stop_event.set()
        hb_thread.join(timeout=heartbeat_s + 1.0)

    logger.info("rank done: %s", stats.snapshot())
    return {
        "rank":         rank,
        "n_in":         stats.n_in,
        "n_ok":         stats.n_ok,
        "n_silent":     stats.n_silent,
        "n_aligned":    stats.n_aligned,
        "wall_seconds": time.monotonic() - stats.t0,
        "output":       str(jsonl_path),
    }


# ---------------------------------------------------------------------------
# Multilang loop — same shape as asr/main.py and asr_parakeet/main.py.
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
      - ``output_dir`` → ``<base>/<lang>``  (``canary/`` is appended by
        ``_run_one`` → ``<base>/<lang>/canary/canary_rank_*.jsonl``).
      - ``canary.language`` → ``lang`` if unset, or ``{lang}`` substituted
        in templated values (literal values pass through untouched).
      - ``alignment.model`` → ``{lang}`` substituted (each NFA checkpoint
        is monolingual: ``stt_fr_*``, ``stt_de_*``, …).
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["shar_dir"] = str(lang_shar_dir)
    cfg["output_dir"] = str(base_output_dir / lang)

    can = cfg.setdefault("canary", {})
    cur = can.get("language")
    if cur is None:
        can["language"] = lang
    elif isinstance(cur, str) and "{lang}" in cur:
        can["language"] = cur.format(lang=lang)

    align = cfg.get("alignment")
    if align and align.get("enabled"):
        amodel = align.get("model")
        if isinstance(amodel, str) and "{lang}" in amodel:
            align["model"] = amodel.format(lang=lang)

    return cfg


_END_EXTRA_KEYS = (
    "n_ok", "n_in", "n_silent", "n_aligned", "wall_seconds", "output",
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
    """Wrap ``_run_one`` with run_history start/end events. See
    ``asr_parakeet/main.py:_tracked_run`` for the rationale."""
    ctx = start_run(
        stage="canary", yaml_path=yaml_path,
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
    log_dir_env = os.environ.get("PIPELINE_LOG_DIR")
    fmt = f"%(asctime)s [r{rank}/{world_size}] %(levelname)s %(name)s: %(message)s"
    handlers: list = [logging.StreamHandler(sys.stderr)]
    if log_dir_env:
        log_dir = Path(log_dir_env)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / f"asr_canary_rank{rank:04d}.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)

    for name in ("nemo", "nemo_logger", "transformers", "lhotse",
                 "urllib3.connectionpool"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser(description="standalone Canary driver")
    ap.add_argument(
        "--config", required=True,
        help="YAML config (asr_canary/egs/*.yaml shape).",
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
        "asr_canary starting: rank %d/%d on GPU %d (%s)",
        rank, world_size, local_rank, torch.cuda.get_device_name(local_rank),
    )

    yaml_path = str(cfg_path)

    lsd = cfg.get("language_split_dir")
    if not lsd:
        stats = _tracked_run(cfg, rank, world_size, local_rank, yaml_path=yaml_path)
        logger.info("asr_canary done: %s", stats)
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
    base_output_dir = Path(cfg.get("output_dir", "./results_asr_canary"))
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

        # NOTE: we deliberately do NOT gc.collect() / cuda.empty_cache()
        # here — Canary + VAD live in _WORKER_CACHE across iterations,
        # and the aligner is evicted explicitly inside _cache_get_or_build
        # when the language changes (model path differs per lang).

    logger.info("asr_canary done — %d run(s): %s", len(all_stats), all_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
