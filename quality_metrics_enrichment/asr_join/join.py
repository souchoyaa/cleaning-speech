"""Load per-slot JSONLs into dicts keyed by cut_id, merge across slots.

In-memory merge (not streaming): for our scale (~400k cuts × 3 slots ≈ 1.2M
rows × ~500 bytes = ~600 MB at peak) RAM is fine and the code stays simple.
Streaming/k-way merge would require pre-sorting each JSONL by cut_id, which
the upstream writers don't guarantee.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import orjson

logger = logging.getLogger(__name__)


# Top-level fields that asr_vllm puts directly on the row (vs nested under
# the slot name like parakeet/canary do). Used when payload_at=None.
_VLLM_TOPLEVEL_FIELDS = ("text", "avg_logp", "language", "error")

# Cross-slot row metadata we propagate to the merged row. Same keys are
# emitted by all three runners.
_META_FIELDS = ("duration", "ref_text", "speaker", "language_hint", "vad")


def load_slot(
    root: Path,
    dir_name: str,
    payload_at: Optional[str],
    slot_name: str,
) -> dict:
    """Read all rank files for one slot under
    ``root/asr_plain_output/dir_name/``.

    Returns ``{cut_id: {meta..., "_hyp": hyp_dict}}`` where ``hyp_dict``
    is the slot's hypothesis (text, avg_logp, possibly word_timestamps /
    language / error).

    - ``payload_at`` is the row key under which the hypothesis lives
      (e.g. "parakeet" → row["parakeet"]). When None, top-level fields
      (``text``, ``avg_logp``, ``language``, ``error``) are lifted into
      a synthesized hyp dict — this is how ``asr_vllm`` writes its rows.
    - Tolerates malformed lines (skipped with a count); a torn last line
      (SIGKILL mid-write) just gets dropped.

    The ``asr_plain_output/`` layer is hardcoded to match the layout the
    three ASR runners write into (see asr_parakeet/main.py et al.).
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Slot input root missing: {root}")
    slot_dir = root / "quality_asr" / dir_name
    if not slot_dir.is_dir():
        raise FileNotFoundError(f"Slot dir not found: {slot_dir}")

    out: dict = {}
    n_files = 0
    n_rows = 0
    n_skipped = 0
    pattern = f"{dir_name}_rank_*.jsonl"
    for jp in sorted(slot_dir.glob(pattern)):
        n_files += 1
        with jp.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    row = orjson.loads(raw)
                except Exception:
                    n_skipped += 1
                    continue
                cid = row.get("cut_id")
                if not cid:
                    n_skipped += 1
                    continue
                if payload_at:
                    hyp = row.get(payload_at) or {}
                else:
                    hyp = {k: row[k] for k in _VLLM_TOPLEVEL_FIELDS if k in row}
                meta = {k: row.get(k) for k in _META_FIELDS}
                out[cid] = {**meta, "_hyp": hyp}
                n_rows += 1
    if n_files == 0:
        raise FileNotFoundError(
            f"No rank files matched {slot_dir}/{pattern}",
        )
    logger.info(
        "load_slot %s: %d files → %d unique cuts (%d malformed/no-id)",
        slot_name, n_files, len(out), n_skipped,
    )
    return out


def merge_slots(slot_rows: dict) -> Iterator[dict]:
    """Yield one merged row per cut_id across all slots.

    Output row::

        {cut_id, duration, ref_text, speaker, language_hint, vad,
         hypotheses: {slot_name: hyp_dict, ...}}

    Metadata (``duration`` / ``vad`` / ``ref_text`` / etc.) is taken from
    whichever slot reports a non-null value first — they should all agree,
    so first-non-null is a safe and fast policy.

    A cut present in only some slots gets ``hypotheses[missing_slot]``
    omitted (downstream rover handles that — votes from 2 slots become
    a 2-of-2 majority instead of 2-of-3).
    """
    all_cut_ids: set = set()
    for rows in slot_rows.values():
        all_cut_ids.update(rows.keys())

    for cid in sorted(all_cut_ids):
        merged: dict = {"cut_id": cid, "hypotheses": {}}
        for slot_name, rows in slot_rows.items():
            r = rows.get(cid)
            if r is None:
                continue
            for k in _META_FIELDS:
                if merged.get(k) is None and r.get(k) is not None:
                    merged[k] = r[k]
            merged["hypotheses"][slot_name] = r["_hyp"]
        yield merged
