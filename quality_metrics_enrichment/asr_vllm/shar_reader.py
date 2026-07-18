"""Minimal Lhotse Shar reader for the vLLM feeder.

Skips Lhotse's DataLoader + DynamicBucketingSampler entirely: vLLM does
its own scheduling server-side, so all the feeder needs is raw FLAC
bytes + light per-cut metadata.

Format (from ``utils/prepare_data/common.py:build_shar_index_from_parts``
and ``lhotse.shar.readers.tar``):

* ``shar_index.json`` at the root has ``fields.cuts`` (list of
  ``cuts.NNNNNN.jsonl.gz`` paths) and ``fields.recording`` (list of
  ``recording.NNNNNN.tar`` paths). Both lists are aligned by index.
* Each ``cuts.*.jsonl.gz`` is a one-cut-per-line manifest (with
  ``id``, ``duration``, ``supervisions``, ``custom``, ...).
* Each ``recording.*.tar`` stores **pairs** of entries in the same
  order as the manifest: a data file (e.g. ``<cut_id>.flac``) followed
  by a JSON metadata file. We need only the data side.

This module reads cuts.jsonl and the recording tar in lockstep per
shard, yielding ``(meta_dict, flac_bytes, ext)`` tuples. Shards are
read concurrently by a small thread pool — Lustre/Capstor likes a few
in flight to overlap IO latency.
"""

from __future__ import annotations

import gzip
import json
import logging
import queue
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


_DEFAULT_INDEX_FILENAME = "shar_index.json"
_END_SENTINEL = object()


@dataclass
class ShardPair:
    """One ``(cuts.jsonl.gz, recording.tar)`` pair from shar_index.json."""

    cuts_path: Path
    recording_path: Path
    shard_idx: int  # for logging / determinism


@dataclass
class CutItem:
    """One cut emitted by the reader."""

    cut_id: str
    flac_bytes: bytes
    duration: float
    language: Optional[str]
    ref_text: Optional[str]
    speaker: Optional[str]
    ext: str  # "flac", "wav", "ogg", ...
    shard_idx: int


def _load_shard_pairs(shar_dir: Path, *, index_filename: str) -> list[ShardPair]:
    """Read shar_index.json and pair up cuts/recording shards."""
    index_path = shar_dir / index_filename
    payload = json.loads(index_path.read_text())
    fields = payload.get("fields", {})
    cuts_files = sorted(fields.get("cuts") or [])
    rec_files = sorted(fields.get("recording") or [])
    if not cuts_files:
        raise FileNotFoundError(f"No 'cuts' field in {index_path}")
    if not rec_files:
        raise FileNotFoundError(f"No 'recording' field in {index_path}")
    if len(cuts_files) != len(rec_files):
        raise ValueError(
            f"Mismatched shard counts in {index_path}: "
            f"cuts={len(cuts_files)} recording={len(rec_files)}"
        )
    pairs: list[ShardPair] = []
    for i, (c, r) in enumerate(zip(cuts_files, rec_files)):
        pairs.append(ShardPair(
            cuts_path=shar_dir / c,
            recording_path=shar_dir / r,
            shard_idx=i,
        ))
    return pairs


def _split_supervision(meta: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (language, ref_text, speaker) with the same precedence used elsewhere.

    Mirrors the convention in ``asr/workers/canary.py:resolve_languages``:
    cfg default > cut.custom['language'] > supervisions[0].language.
    Here we don't have a cfg default (the feeder applies its own), so:
    cut.custom['language'] > supervisions[0].language.
    """
    supervisions = meta.get("supervisions") or []
    sup = supervisions[0] if supervisions else {}
    custom = meta.get("custom") or {}
    language = (
        custom.get("language")
        or sup.get("language")
        or None
    )
    ref_text = sup.get("text")
    speaker = sup.get("speaker") or meta.get("speaker")
    return language, ref_text, speaker


def _stream_shard(
    pair: ShardPair,
    *,
    min_duration: Optional[float],
    max_duration: Optional[float],
) -> Iterator[CutItem]:
    """Yield CutItems for one shard, reading cuts.jsonl.gz + tar in lockstep.

    Tar layout per Lhotse Shar: pairs of (data, metadata) entries in the
    same order as the cuts manifest. We only consume the data side and
    discard the metadata entry.

    Streaming-mode tarfile (``r|*``) cannot seek backwards: ``extractfile``
    must be called BEFORE the iterator advances past the entry. So the
    order per cut is: get data tarinfo → read its bytes → advance past
    metadata tarinfo. Doing the two ``next()`` calls back-to-back before
    extracting raises ``StreamError: seeking backwards is not allowed``.
    """
    with gzip.open(pair.cuts_path, "rt", encoding="utf-8") as cf:
        with tarfile.open(pair.recording_path, mode="r|*") as tar:
            tar_iter = iter(tar)
            for line in cf:
                meta = json.loads(line)
                cut_id = meta["id"]
                duration = float(meta.get("duration", 0.0))

                # Always advance the tar iter in lockstep with the manifest,
                # even when we're going to skip this cut by duration filter.
                data_info = next(tar_iter, None)
                if data_info is None:
                    raise RuntimeError(
                        f"Tar truncated for {pair.recording_path}: "
                        f"missing data entry at cut_id={cut_id}",
                    )

                # Decide whether to keep this cut BEFORE reading bytes —
                # for filtered cuts we skip the costly extractfile.
                keep = (
                    (min_duration is None or duration >= min_duration)
                    and (max_duration is None or duration <= max_duration)
                )

                flac_bytes: bytes = b""
                ext = "nodata"
                if keep:
                    if data_info.path.endswith(".nodata"):
                        # No audio for this cut — emit empty blob so the
                        # writer can mark it failed downstream.
                        pass
                    else:
                        # Read NOW while the stream is positioned at data_info.
                        f = tar.extractfile(data_info)
                        flac_bytes = f.read() if f is not None else b""
                        ext = Path(data_info.path).suffix.lstrip(".") or "bin"

                # Advance past the metadata entry (we don't need it).
                meta_info = next(tar_iter, None)
                if meta_info is None:
                    raise RuntimeError(
                        f"Tar truncated for {pair.recording_path}: "
                        f"missing metadata entry at cut_id={cut_id}",
                    )

                if not keep:
                    continue

                lang, ref_text, speaker = _split_supervision(meta)
                yield CutItem(
                    cut_id=cut_id,
                    flac_bytes=flac_bytes,
                    duration=duration,
                    language=lang,
                    ref_text=ref_text,
                    speaker=speaker,
                    ext=ext,
                    shard_idx=pair.shard_idx,
                )


# ---------------------------------------------------------------------------
# Public iterator
# ---------------------------------------------------------------------------


def _scan_one_shard(
    p: ShardPair,
    *,
    skip_cut_ids: Optional[set[str]],
    min_duration: Optional[float],
    max_duration: Optional[float],
) -> float:
    """Sum filtered durations from a single ``cuts.*.jsonl.gz``."""
    total = 0.0
    try:
        with gzip.open(p.cuts_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    meta = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if skip_cut_ids is not None and meta.get("id") in skip_cut_ids:
                    continue
                d = float(meta.get("duration", 0.0))
                if min_duration is not None and d < min_duration:
                    continue
                if max_duration is not None and d > max_duration:
                    continue
                total += d
    except Exception:
        logger.exception("scan_one_shard: failed on %s — continuing.", p.cuts_path)
    return total


def scan_total_duration(
    shar_dir: str | Path,
    *,
    skip_cut_ids: Optional[set[str]] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    on_progress=None,
    num_workers: int = 16,
    index_filename: str = _DEFAULT_INDEX_FILENAME,
) -> float:
    """Sum durations across all cuts.*.jsonl.gz manifests, in parallel.

    Reads only the gzipped manifests — no tar audio touched. Gzip decode
    + JSON parse releases the GIL during the decompression syscalls, so
    a ThreadPoolExecutor gives near-linear speedup on Capstor up to
    ~16 workers (then disk bandwidth caps it).

    Filtering:
      - ``skip_cut_ids``: cuts already in the resume sidecar are
        excluded (so the total reflects ``remaining`` work for ETA).
      - ``min_duration`` / ``max_duration``: same filter the feeder
        applies, so the total matches what will actually be processed.

    Progress: ``on_progress(shards_done, shards_total, partial_seconds)``
    fires after each shard finishes. Order is by completion (not by
    shard index) since shards run in parallel.
    """
    shar_dir = Path(shar_dir)
    pairs = _load_shard_pairs(shar_dir, index_filename=index_filename)
    if not pairs:
        return 0.0

    n_workers = max(1, min(int(num_workers), len(pairs)))
    total_s = 0.0
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="scan") as exe:
        futures = [
            exe.submit(
                _scan_one_shard, p,
                skip_cut_ids=skip_cut_ids,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            for p in pairs
        ]
        for fut in as_completed(futures):
            total_s += fut.result()
            done += 1
            if on_progress is not None:
                try:
                    on_progress(done, len(pairs), total_s)
                except Exception:
                    pass
    return total_s


def iter_shar_cuts(
    shar_dir: str | Path,
    *,
    num_workers: int = 4,
    queue_size: int = 256,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    index_filename: str = _DEFAULT_INDEX_FILENAME,
) -> Iterator[CutItem]:
    """Iterate cuts from a Shar with N concurrent shard readers.

    Ordering: items from any one shard are emitted in manifest order;
    items across shards are interleaved (whichever worker finishes its
    next read first wins the queue slot). For a feeder this is fine —
    the join is offline on cut_id.

    Backpressure: each worker blocks on ``queue.put`` when the queue is
    full, so memory is bounded to ~``queue_size`` cuts in flight.

    Concurrency: ``num_workers`` is the count of shards being read at
    once. One shard at a time per worker (sequential within a shard).
    """
    shar_dir = Path(shar_dir)
    pairs = _load_shard_pairs(shar_dir, index_filename=index_filename)
    if not pairs:
        return

    n_workers = max(1, min(num_workers, len(pairs)))
    out_q: "queue.Queue" = queue.Queue(maxsize=queue_size)
    stop = threading.Event()

    def _worker(pair_slice: list[ShardPair]) -> None:
        try:
            for p in pair_slice:
                if stop.is_set():
                    return
                try:
                    for item in _stream_shard(
                        p,
                        min_duration=min_duration,
                        max_duration=max_duration,
                    ):
                        if stop.is_set():
                            return
                        out_q.put(item)
                except Exception as e:
                    logger.exception(
                        "Shard reader failed: shard_idx=%d cuts=%s recording=%s err=%s",
                        p.shard_idx, p.cuts_path, p.recording_path, e,
                    )
        finally:
            out_q.put(_END_SENTINEL)

    # Round-robin shards across workers.
    slices: list[list[ShardPair]] = [[] for _ in range(n_workers)]
    for i, p in enumerate(pairs):
        slices[i % n_workers].append(p)

    threads = [
        threading.Thread(target=_worker, args=(s,), name=f"shar_reader_{i}",
                         daemon=True)
        for i, s in enumerate(slices)
    ]
    for t in threads:
        t.start()

    n_done = 0
    try:
        while n_done < n_workers:
            item = out_q.get()
            if item is _END_SENTINEL:
                n_done += 1
                continue
            yield item
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10.0)
