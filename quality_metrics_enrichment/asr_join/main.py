"""asr_join — offline merge + ROVER + (optional) LLM-based ITN.

Two-phase per corpus (single-shar or one language in multilang):
  Phase 1 (sync): merge per-rank JSONLs across slots -> ROVER per cut -> buffer.
  Phase 2 (async, optional): for each row with non-empty rover.text, call the
    vLLM chat API for ITN (an LLM covers any served language, beyond NeMo's ~14).

Output: ``<merge_output_dir>[/<lang>]/rover/merged.jsonl``

Run::
    python -m asr_join.main --config asr_join/egs/cv_fr.yaml
    bash asr_join/scripts/launch.sh --config cv_fr.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import orjson
import yaml

from common.run_history import end_run, guard_cfg_hash, start_run

from . import join, rover_offline
from .itn import ITNClient, ITNTransportError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults for the async ITN pass — overridable via YAML rover.itn.*
# ---------------------------------------------------------------------------

_DEFAULT_ITN_CONCURRENCY = 32
_DEFAULT_ITN_TIMEOUT_S = 60.0
_DEFAULT_ITN_CHUNK_SIZE = 1000  # in-flight task batch size (memory bound)
_DEFAULT_ITN_PROGRESS_EVERY = 5000
# Circuit-breaker: abort if a chunk has more than this fraction of LLM
# failures (passthrough due to HTTP/parse error). Catches "ITN vLLM died";
# at 1000-row chunks, >80% failures means hundreds of dead requests in a
# row — server is gone. 0 disables, 1.0 means never trip.
_DEFAULT_ITN_CIRCUIT_BREAK_FRACTION = 0.8


def _run_phase1_merge_rover(
    input_root: Path,
    slots_cfg: dict,
    rover_cfg: dict,
    expected_language: Optional[str],
) -> tuple:
    """Sync phase: load + merge + ROVER. Returns (rows, stats_partial).

    All three slots are loaded from the same ``input_root`` (the dataset
    directory) — the per-slot subdir is ``asr_plain_output/<dir_name>/``
    by convention, handled in ``join.load_slot``.
    """
    t0 = time.monotonic()

    slot_rows: dict = {}
    for slot_name, slot_def in slots_cfg.items():
        dir_name = slot_def.get("dir_name", slot_name)
        payload_at = slot_def.get("payload_at", slot_name)
        slot_rows[slot_name] = join.load_slot(input_root, dir_name, payload_at, slot_name)

    rows: list = []
    n_filtered = 0
    n_partial = 0
    n_slots_total = len(slots_cfg)
    n_filt_by_reason: dict = {}

    # ─── OPTIONAL / REMOVABLE: dual-emit (deferred "ASR transcript → dedup" idea) ───
    # Off by default. Enabled via `--emit-dedup-enhanced` (or rover.emit_dual: true).
    # When on, each row also gets, in ONE join pass (no ASR re-run later):
    #   rover.text_dedup    = reference-INDEPENDENT majority consensus  (future dedup key)
    #   rover.text_enhanced = reference-ANCHORED consensus (ref as extra voter; improvement)
    # The normal rover.text / primary_fallbacks (the ROVER quality metric) are unchanged.
    # If the dedup change is NOT made before hand-in, delete: this block, the
    # `_dual_consensus` helper, and the `--emit-dedup-enhanced` flag in main().
    # See memory: project_asr_canonical_dedup_idea.
    emit_dual = bool(rover_cfg.get("emit_dual", False))

    def _dual_consensus(m, inc_ref):
        """Recompute the ROVER consensus text for a given include_ref_text on a
        shallow copy (apply() only writes top-level keys, reads hyps read-only)."""
        mc = {**m}
        rover_offline.apply(
            mc,
            primary=rover_cfg.get("primary", "qwen"),
            voting=rover_cfg.get("voting", "majority"),
            timestamp_source=rover_cfg.get("timestamp_source"),
            repetition_ngram=int(rover_cfg.get("repetition_ngram", 15)),
            repetition_max_count=int(rover_cfg.get("repetition_max_count", 5)),
            expected_language=expected_language,
            include_ref_text=inc_ref,
        )
        return (mc.get("rover") or {}).get("text", "")
    # ───────────────────────────────────────────────────────────────────────────────

    # ─── Enhanced (improved) transcript: ref-anchored, weighted, deterministic ───
    # Adds rover.text_enhanced = the original transcript corrected by ASR
    # consensus (ref is the anchor/default; >=2 ASRs override it).  Distinct
    # subset-sum weights make the per-word arg-max unique -> deterministic for
    # any voter count.  ITN (if on) refines this; Stage G writes it to the shar.
    enh_cfg = rover_cfg.get("enhanced") or {}
    enh_enabled = bool(enh_cfg.get("enabled", False))
    enh_weights = dict(enh_cfg.get("weights") or {})
    if enh_enabled:
        enh_weights.setdefault("ref", 1.5)
        rover_offline.assert_deterministic_weights(enh_weights)
        logger.info("enhanced consensus ON — ref-anchored weighted voting, "
                    "weights=%s", enh_weights)

    for merged in join.merge_slots(slot_rows):
        if len(merged["hypotheses"]) < n_slots_total:
            n_partial += 1
        rover_offline.apply(
            merged,
            primary=rover_cfg.get("primary", "qwen"),
            voting=rover_cfg.get("voting", "majority"),
            timestamp_source=rover_cfg.get("timestamp_source"),
            repetition_ngram=int(rover_cfg.get("repetition_ngram", 15)),
            repetition_max_count=int(rover_cfg.get("repetition_max_count", 5)),
            expected_language=expected_language,
            include_ref_text=bool(rover_cfg.get("include_ref_text", False)),
        )
        # ─── OPTIONAL / REMOVABLE (dual-emit) ───
        if emit_dual and merged.get("rover"):
            main_ref = bool(rover_cfg.get("include_ref_text", False))
            merged["rover"]["text_dedup"] = (
                merged["rover"]["text"] if not main_ref
                else _dual_consensus(merged, False))
            merged["rover"]["text_enhanced"] = (
                merged["rover"]["text"] if main_ref
                else _dual_consensus(merged, True))
        # ─────────────────────────────────────────
        # Enhanced (improved) transcript — ref-anchored weighted consensus.
        if enh_enabled and merged.get("rover"):
            enh = rover_offline.consensus_enhanced(
                merged,
                weights=enh_weights,
                timestamp_source=rover_cfg.get("timestamp_source"),
                repetition_ngram=int(rover_cfg.get("repetition_ngram", 15)),
                repetition_max_count=int(rover_cfg.get("repetition_max_count", 5)),
                expected_language=expected_language,
            )
            r = merged["rover"]
            # Fall back to the plain ASR consensus when there is no original
            # transcript to anchor on (ref_text empty -> enhanced text empty).
            r["text_enhanced"] = enh.get("text") or r.get("text", "")
            r["word_timestamps_enhanced"] = enh.get("word_timestamps") or []
            r["ambiguous_words_enhanced"] = enh.get("ambiguous_words") or []
        # Unified id: stamp the dataset tag so retention's quality join keys on
        # (dataset, cut_id) and can read merged.jsonl directly (no bridge copy).
        if rover_cfg.get("dataset"):
            merged["dataset"] = rover_cfg["dataset"]
        rows.append(merged)
        reason = merged.get("filtered_reason")
        if reason:
            n_filtered += 1
            n_filt_by_reason[reason] = n_filt_by_reason.get(reason, 0) + 1

    wall = time.monotonic() - t0
    logger.info(
        "phase 1 (merge+rover): %d cuts in %.1fs — %d filtered (%s), "
        "%d partial (missing 1+ slot)",
        len(rows), wall, n_filtered, n_filt_by_reason, n_partial,
    )
    return rows, {
        "n_total":          len(rows),
        "n_filtered":       n_filtered,
        "n_filt_by_reason": n_filt_by_reason,
        "n_partial":        n_partial,
        "phase1_seconds":   wall,
    }


async def _run_phase2_itn(
    rows: list,
    itn_cfg: dict,
    expected_language: Optional[str],
) -> float:
    """Async phase: parallel LLM ITN on each row's rover.text.

    Mutates rows in place — adds ``rover.text_itn``. Returns wall time.
    Skips rows where rover.text is empty (all_failed cuts). Failures fall
    back to verbatim inside ITNClient.normalize.
    """
    t0 = time.monotonic()
    client_cfg = ITNClient(itn_cfg)

    concurrency = int(itn_cfg.get("concurrency", _DEFAULT_ITN_CONCURRENCY))
    timeout_s = float(itn_cfg.get("timeout_seconds", _DEFAULT_ITN_TIMEOUT_S))
    chunk_size = int(itn_cfg.get("chunk_size", _DEFAULT_ITN_CHUNK_SIZE))
    progress_every = int(itn_cfg.get("progress_every", _DEFAULT_ITN_PROGRESS_EVERY))
    circuit_break_fraction = float(
        itn_cfg.get("circuit_break_fraction", _DEFAULT_ITN_CIRCUIT_BREAK_FRACTION),
    )
    # Which rover field ITN reads.  "text_enhanced" routes ITN onto the
    # ref-anchored improved transcript (-> text_enhanced_itn) and uses its
    # ambiguous_words_enhanced; the default keeps legacy behavior on the plain
    # consensus (-> text_itn).  Falls back to "text" when the field is absent.
    input_field = str(itn_cfg.get("input_field", "text"))
    output_field = input_field + "_itn"
    ambig_field = ("ambiguous_words_enhanced" if input_field == "text_enhanced"
                   else "ambiguous_words")

    def _src(r: dict) -> str:
        rv = r.get("rover") or {}
        return (rv.get(input_field) or rv.get("text") or "")

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency,
    )
    timeout = httpx.Timeout(timeout_s, connect=min(15.0, timeout_s))

    n_processed = 0
    n_skipped_empty = 0

    n_failed_total = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def process_row(row: dict) -> bool:
            """Return True iff this row's ITN failed (transport error).
            Length-sanity passthrough is NOT counted — that's a per-row
            quirk, not a server fault."""
            text = _src(row)
            if not text.strip():
                # all_failed / empty rover output: <field>_itn = "" (consistent
                # with the source field); skip the HTTP round-trip.
                row["rover"][output_field] = ""
                return False
            lc = row.get("language_consistency") or {}
            lang = (
                lc.get("expected")
                or row.get("language_hint")
                or expected_language
            )
            # Pass through ambiguous-word positions from the ROVER pass so
            # the LLM can resolve word-level ties contextually in the same
            # call that does ITN (single round-trip).
            ambig = (row.get("rover") or {}).get(ambig_field) or []
            async with sem:
                try:
                    row["rover"][output_field] = await client_cfg.normalize(
                        client, text, lang, ambiguous_words=ambig,
                    )
                    return False
                except ITNTransportError as e:
                    # Verbatim passthrough so downstream sees a value; but
                    # the failure is counted for the circuit breaker.
                    row["rover"][output_field] = text
                    logger.warning(
                        "ITN: transport error lang=%s (%s) — passthrough.", lang, e,
                    )
                    return True

        # Chunked execution: bounded number of in-flight tasks so we don't
        # explode RAM with 400k coroutines or saturate the event loop. After
        # each chunk, check the circuit breaker — if too many rows failed,
        # abort the whole ITN pass loudly (likely vLLM down).
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            chunk_failures = await asyncio.gather(*(process_row(r) for r in chunk))
            n_chunk_failed = sum(chunk_failures)
            n_failed_total += n_chunk_failed
            n_processed += len(chunk)
            n_skipped_empty += sum(
                1 for r in chunk if not _src(r).strip()
            )

            # Circuit breaker: a chunk where >= circuit_break_fraction of
            # rows hit transport errors means the ITN vLLM has almost
            # certainly died. Stop wasting compute.
            if 0.0 < circuit_break_fraction <= 1.0:
                # Only count non-skip rows in the denominator (empty rover.text
                # rows don't make HTTP calls).
                n_called = sum(1 for r in chunk if _src(r).strip())
                if n_called > 0:
                    frac_failed = n_chunk_failed / n_called
                    if frac_failed >= circuit_break_fraction:
                        raise RuntimeError(
                            f"ITN circuit breaker tripped: chunk "
                            f"[{start}:{start + len(chunk)}] had "
                            f"{n_chunk_failed}/{n_called} transport failures "
                            f"({frac_failed:.0%} >= {circuit_break_fraction:.0%}). "
                            f"ITN vLLM likely down. Aborting."
                        )

            if n_processed % progress_every < chunk_size:
                logger.info(
                    "ITN: %d/%d rows processed (%.0f%%, %d failed total)",
                    n_processed, len(rows),
                    100.0 * n_processed / max(len(rows), 1), n_failed_total,
                )

    wall = time.monotonic() - t0
    logger.info(
        "phase 2 (LLM ITN): %d rows in %.1fs (%.1f rows/s, %d empty skipped)",
        n_processed, wall, n_processed / max(wall, 1e-6), n_skipped_empty,
    )
    return wall


def _write_rows(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row)); f.write(b"\n")


def _run_one(
    input_root: Path,
    slots_cfg: dict,
    merge_output_dir: Path,
    rover_cfg: dict,
    expected_language: Optional[str] = None,
) -> dict:
    """Merge + rover + (optional) LLM ITN for one corpus. Returns stats.

    Layout: ``<merge_output_dir>/asr_agg/rover/merged.jsonl``. The
    ``asr_agg/`` layer pairs with the per-slot ``asr_plain_output/``
    written by the three ASR runners — both live under the same dataset
    root so the join can read inputs and write outputs in the same tree.
    """
    # Refuse to append into a rover/ output dir whose previous run used a
    # different ROVER/ITN cfg. RESUME_ANYWAY=1 to override.
    rover_dir = merge_output_dir / "quality_asr" / "rover"
    rover_dir.mkdir(parents=True, exist_ok=True)
    # Hash only the rover_cfg + slot definitions (those determine the
    # output schema); the input_roots / expected_language are operational.
    guard_cfg_hash(
        rover_dir,
        {"rover": rover_cfg, "slots": slots_cfg},
        allow_overwrite=bool(os.environ.get("RESUME_ANYWAY")),
    )
    rows, stats = _run_phase1_merge_rover(
        input_root, slots_cfg, rover_cfg, expected_language,
    )

    itn_cfg = rover_cfg.get("itn") or {}
    if itn_cfg.get("enabled"):
        phase2_wall = asyncio.run(_run_phase2_itn(rows, itn_cfg, expected_language))
        stats["phase2_seconds"] = phase2_wall
    else:
        stats["phase2_seconds"] = 0.0

    out_path = merge_output_dir / "quality_asr" / "rover" / "merged.jsonl"
    _write_rows(rows, out_path)
    logger.info("wrote %d rows to %s", len(rows), out_path)
    stats["output"] = str(out_path)
    stats["wall_seconds"] = stats["phase1_seconds"] + stats["phase2_seconds"]
    return stats


_END_EXTRA_KEYS = (
    "n_total", "n_filtered", "n_partial",
    "phase1_seconds", "phase2_seconds", "wall_seconds", "output",
)


def _tracked_run_one(
    input_root: Path,
    slots_cfg: dict,
    merge_output_dir: Path,
    rover_cfg: dict,
    expected_language: Optional[str] = None,
    *,
    yaml_path: str,
    language: Optional[str] = None,
) -> dict:
    """Wrap ``_run_one`` with run_history start/end events."""
    ctx = start_run(
        stage="asr_join", yaml_path=yaml_path,
        rank=0, world_size=1,
        output_dir=str(merge_output_dir),
    )
    try:
        stats = _run_one(
            input_root, slots_cfg, merge_output_dir,
            rover_cfg, expected_language,
        )
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


def _resolve_languages(input_root: Path, explicit) -> list:
    """If ``languages`` is set in the YAML, use it. Else auto-discover by
    listing the immediate subdirs of ``input_root``."""
    if explicit:
        return list(explicit)
    langs = sorted(p.name for p in input_root.iterdir() if p.is_dir())
    logger.info("auto-discovered %d languages from %s: %s", len(langs), input_root, langs)
    return langs


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # httpx logs one INFO line per request — at 32 concurrent + 400k rows
    # that drowns the heartbeat. Lift to WARNING so only failures surface.
    logging.getLogger("httpx").setLevel("WARNING")
    logging.getLogger("httpcore").setLevel("WARNING")


def main() -> int:
    ap = argparse.ArgumentParser(description="asr_join — offline merge + ROVER + LLM ITN")
    ap.add_argument(
        "--config", required=True,
        help="YAML config (asr_join/egs/*.yaml shape).",
    )
    # OPTIONAL / REMOVABLE (dual-emit, deferred ASR->dedup idea). Off by default.
    # Adds rover.text_dedup (ref-independent, future dedup key) + rover.text_enhanced
    # (ref-anchored improvement) in the same pass. Delete this flag if unused.
    ap.add_argument(
        "--emit-dedup-enhanced", action="store_true",
        help="Also emit rover.text_dedup + rover.text_enhanced (no ASR re-run "
             "needed later for the ASR-transcript-as-dedup-key change).",
    )
    args = ap.parse_args()

    _setup_logging()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())

    # Single input_root — all 3 slots live under
    # ``<input_root>/asr_plain_output/<slot>/``. Output lands at
    # ``<merge_output_dir>/asr_agg/rover/merged.jsonl``.
    # ``input_root`` and ``merge_output_dir`` are usually the SAME path
    # (the per-dataset directory), so a single line in YAML works.
    input_root = Path(cfg["input_root"])
    slots_cfg = cfg["slots"]
    merge_output_dir = Path(cfg.get("merge_output_dir", cfg["input_root"]))
    rover_cfg = cfg.get("rover") or {}
    # Unified id: pass the dataset tag down so each merged row is stamped (lets
    # retention read rover/merged.jsonl directly, no asr_moe bridge copy).
    if cfg.get("dataset"):
        rover_cfg["dataset"] = cfg["dataset"]
    # OPTIONAL / REMOVABLE (dual-emit): CLI flag overrides the config.
    if args.emit_dedup_enhanced:
        rover_cfg["emit_dual"] = True

    # Multilang mode is signalled by ``language_split_dir`` OR ``languages``.
    multilang = bool(cfg.get("language_split_dir")) or bool(cfg.get("languages"))

    yaml_path = str(cfg_path)

    if not multilang:
        stats = _tracked_run_one(
            input_root, slots_cfg, merge_output_dir,
            rover_cfg, cfg.get("expected_language"),
            yaml_path=yaml_path,
        )
        logger.info("done: %s", stats)
        return 0

    languages = _resolve_languages(input_root, cfg.get("languages"))
    all_stats = []
    for lang in languages:
        logger.info("=== BEGIN %s ===", lang)
        lang_input_root = input_root / lang
        lang_out = merge_output_dir / lang
        try:
            stats = _tracked_run_one(
                lang_input_root, slots_cfg, lang_out,
                rover_cfg, expected_language=lang,
                yaml_path=yaml_path, language=lang,
            )
        except FileNotFoundError as e:
            logger.error("skipping %s: %s", lang, e)
            continue
        stats["language"] = lang
        all_stats.append(stats)
        logger.info("=== END %s stats=%s ===", lang, stats)

    logger.info("multilang join done — %d run(s): %s", len(all_stats), all_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
