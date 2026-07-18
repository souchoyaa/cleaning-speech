"""Stage D loader — random-access audio reads for the candidate (dataset, cut_id) set.

The Stage-A manifest carries a tar locator (tar_path/tar_offset/tar_size) per
cut; ``RandomAccessSharReader`` seeks directly to each candidate's audio member
so Stage D decodes only the cuts Stage C flagged.  The whitelist is a
``set[(dataset, cut_id)]`` composite key.
"""

import io
import logging
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AudioBatch with composite key (dataset, cut_id) tagged on every sample
# ---------------------------------------------------------------------------


@dataclass
class CandidateAudioBatch:
    """One batch of candidate cuts, tagged with the source dataset.

    Attributes
    ----------
    dataset:
        Logical dataset tag for every sample in this batch (single-shar
        loader, so this is constant across the batch).
    cut_ids:
        Lhotse cut ids — one per sample.
    audio:
        ``(B, T_max)`` float32 CPU tensor — padded waveforms.
    lengths:
        ``(B,)`` int64 CPU tensor — actual sample counts.
    sr:
        Sampling rate (target_sample_rate from cfg).
    """

    dataset: str
    cut_ids: List[str]
    audio: torch.Tensor
    lengths: torch.Tensor
    sr: int


# ---------------------------------------------------------------------------
# Whitelist helpers
# ---------------------------------------------------------------------------


def load_whitelist(text_clusters_parquet: Path) -> Set[Tuple[str, str]]:
    """Read the (dataset, cut_id) pairs from clusters.parquet."""
    import pyarrow.parquet as pq
    t = pq.read_table(str(text_clusters_parquet), columns=["dataset", "cut_id"])
    ds = t.column("dataset").to_pylist()
    cid = t.column("cut_id").to_pylist()
    return set(zip(ds, cid))


def whitelist_for_dataset(whitelist: Set[Tuple[str, str]], dataset: str) -> Set[str]:
    """Return the set of cut_ids in ``whitelist`` belonging to ``dataset``."""
    return {cid for (ds, cid) in whitelist if ds == dataset}






# ---------------------------------------------------------------------------
# Random-access path (Stage-A locator → direct tar seek+read)
# ---------------------------------------------------------------------------
#
# The Stage-A manifest carries the tar locator (tar_path/tar_offset/tar_size),
# so Stage D seeks directly to each candidate's audio member — reading only
# ~candidate_fraction of the bytes instead of streaming every tar end-to-end.
#
# All tar access here is STRICTLY READ-ONLY (open "rb", seek, read).  Nothing in
# this module writes to, truncates, or repacks a shar tar.


# (dataset) -> {cut_id: (rel_tar_path, tar_offset, tar_size, audio_format, num_samples)}
LocatorByDataset = Dict[str, Dict[str, Tuple[str, int, int, Optional[str], Optional[int]]]]


def load_locator(manifest_dir: Path,
                 whitelist: Set[Tuple[str, str]]) -> LocatorByDataset:
    """Read the Stage-A locator columns for the candidate cuts only.

    One projected scan of ``manifest/part_*.parquet``.  Returns an empty dict if
    the manifest predates the locator (no ``tar_offset`` column) — Stage D then
    errors and asks for a manifest re-run.  Cuts with a null offset (text-only /
    no audio member) are omitted — they cannot be fingerprinted.
    """
    import pyarrow.parquet as pq

    manifest_dir = Path(manifest_dir)
    parts = sorted(manifest_dir.glob("part_*.parquet"))
    if not parts:
        return {}
    names = set(pq.ParquetFile(str(parts[0])).schema_arrow.names)
    if "tar_offset" not in names:
        logger.warning("Manifest at %s has no locator columns — re-run the "
                       "manifest stage.", manifest_dir)
        return {}

    want: Dict[str, Set[str]] = {}
    for ds, cid in whitelist:
        want.setdefault(ds, set()).add(cid)

    out: LocatorByDataset = {}
    cols = ["dataset", "cut_id", "tar_path", "tar_offset",
            "tar_size", "audio_format", "num_samples"]
    for p in parts:
        t = pq.read_table(p, columns=cols)
        ds_a  = t.column("dataset").to_pylist()
        cid_a = t.column("cut_id").to_pylist()
        tp_a  = t.column("tar_path").to_pylist()
        off_a = t.column("tar_offset").to_pylist()
        sz_a  = t.column("tar_size").to_pylist()
        fmt_a = t.column("audio_format").to_pylist()
        ns_a  = t.column("num_samples").to_pylist()
        for i in range(len(cid_a)):
            ds = ds_a[i]
            s = want.get(ds)
            if not s or cid_a[i] not in s or off_a[i] is None:
                continue
            out.setdefault(ds, {})[cid_a[i]] = (
                tp_a[i], int(off_a[i]), int(sz_a[i]), fmt_a[i],
                int(ns_a[i]) if ns_a[i] is not None else None,
            )
    return out


def _decode_resample(data: bytes, fmt: Optional[str],
                     target_sr: int) -> Optional[torch.Tensor]:
    """Decode encoded audio bytes → mono float32 tensor at ``target_sr``.

    soundfile (libsndfile) handles flac/wav/ogg from an in-memory buffer;
    torchaudio is the fallback (e.g. mp3/opus).  Returns None on decode failure.
    """
    wav = None
    sr = None
    try:
        import soundfile as sf
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        try:
            import torchaudio
            t, sr = torchaudio.load(io.BytesIO(data),
                                    format=fmt if fmt else None)
            wav = t.numpy()
        except Exception as e:  # pragma: no cover - depends on codecs present
            logger.warning("Random-access decode failed (fmt=%s): %s", fmt, e)
            return None

    import numpy as np
    wav = np.asarray(wav)
    if wav.ndim == 2:                       # (T, C) or (C, T) -> mono
        ch_axis = 0 if wav.shape[0] < wav.shape[1] else 1
        wav = wav.mean(axis=ch_axis)
    wav_t = torch.from_numpy(np.ascontiguousarray(wav)).float()
    if sr is not None and int(sr) != int(target_sr):
        import torchaudio
        wav_t = torchaudio.functional.resample(wav_t, int(sr), int(target_sr))
    return wav_t


def _plan_tar_reads(
    locator: Dict[str, Tuple[str, int, int, Optional[str], Optional[int]]],
    dataset: str,
    skip: Set[Tuple[str, str]],
    rank: int,
    world_size: int,
) -> List[Tuple[str, List[Tuple[str, int, int, Optional[str], Optional[int]]]]]:
    """Plan this rank's reads: ``[(rel_tar, [members …]), …]``.

    Members ``(cut_id, offset, size, fmt, num_samples)`` are grouped by tar;
    tars are assigned round-robin across ranks (locality — each rank touches a
    small set of whole files); within a tar members are ordered by duration
    (minimal batch padding) then offset (sequential-ish reads in the file).
    Pure / no I/O, so it is unit-testable without torch or real audio.
    """
    by_tar: Dict[str, list] = {}
    for cid, (rel, off, sz, fmt, ns) in locator.items():
        if (dataset, cid) in skip:
            continue
        by_tar.setdefault(rel, []).append((cid, off, sz, fmt, ns))
    my_tars = sorted(by_tar)[rank::max(1, world_size)]
    return [
        (rel, sorted(by_tar[rel], key=lambda m: (m[4] if m[4] is not None else 0, m[1])))
        for rel in my_tars
    ]


class RandomAccessSharReader:
    """Yield ``CandidateAudioBatch`` by seeking directly to candidate members.

    Per assigned tar it reads only the candidate
    members (sorted by duration for low padding, then by offset), decodes them
    (thread pool — libsndfile releases the GIL), optionally truncates, and packs
    duration-bounded padded batches.

    Rank split: whole tars are assigned round-robin across ranks, keeping each
    rank's reads within a small set of files.
    """

    def __init__(
        self,
        cfg: dict,
        dataset: str,
        shar_dir: str,
        locator: Dict[str, Tuple[str, int, int, Optional[str], Optional[int]]],
        rank: int = 0,
        world_size: int = 1,
        num_workers: Optional[int] = None,
        skip_keys: Optional[Set[Tuple[str, str]]] = None,
    ):
        self._cfg = cfg
        self._dataset = dataset
        self._shar_dir = Path(shar_dir)
        self._loc = locator
        self._rank = rank
        self._world = max(1, world_size)
        self._skip = skip_keys if skip_keys is not None else set()

        self._target_sr = int(cfg.get("target_sample_rate", 16000))
        max_batch_dur = float(cfg.get("max_batch_duration", 300.0))
        self._budget = max(1, int(max_batch_dur * self._target_sr))
        max_cuts = cfg.get("max_batch_cuts")
        self._max_cuts = int(max_cuts) if max_cuts is not None else None
        trunc = cfg.get("truncate_secs")
        self._trunc_samples = int(float(trunc) * self._target_sr) if trunc is not None else None

        if num_workers is None:
            cpu = os.cpu_count() or 1
            gpu = max(torch.cuda.device_count(), 1)
            num_workers = min(4, max(1, cpu // gpu))
        self._num_workers = max(1, int(num_workers))

    def __iter__(self) -> Iterator[CandidateAudioBatch]:
        plan = _plan_tar_reads(self._loc, self._dataset, self._skip,
                               self._rank, self._world)
        pool = ThreadPoolExecutor(self._num_workers) if self._num_workers > 1 else None
        try:
            buf_audio: List[torch.Tensor] = []
            buf_len: List[int] = []
            buf_cid: List[str] = []
            cur = 0

            def _emit() -> Optional[CandidateAudioBatch]:
                nonlocal buf_audio, buf_len, buf_cid, cur
                if not buf_audio:
                    return None
                T = max(buf_len)
                padded = torch.zeros(len(buf_audio), T, dtype=torch.float32)
                for i, a in enumerate(buf_audio):
                    padded[i, : a.shape[0]] = a
                try:
                    padded = padded.pin_memory()
                except RuntimeError:
                    pass  # no CUDA / pinning unavailable
                batch = CandidateAudioBatch(
                    dataset=self._dataset,
                    cut_ids=list(buf_cid),
                    audio=padded,
                    lengths=torch.tensor(buf_len, dtype=torch.int64),
                    sr=self._target_sr,
                )
                buf_audio, buf_len, buf_cid, cur = [], [], [], 0
                return batch

            for rel, members in plan:
                abs_tar = self._shar_dir / rel
                raws: List[Tuple[str, bytes, Optional[str]]] = []
                with open(abs_tar, "rb") as fh:        # READ-ONLY
                    for cid, off, sz, fmt, _ns in members:
                        fh.seek(off)
                        raws.append((cid, fh.read(sz), fmt))

                tsr = self._target_sr
                if pool is not None:
                    decoded = list(pool.map(
                        lambda r: (r[0], _decode_resample(r[1], r[2], tsr)), raws))
                else:
                    decoded = [(cid, _decode_resample(data, fmt, tsr))
                               for cid, data, fmt in raws]

                for cid, wav in decoded:
                    if wav is None or wav.numel() == 0:
                        continue
                    if self._trunc_samples is not None and wav.shape[0] > self._trunc_samples:
                        wav = wav[: self._trunc_samples]
                    n = int(wav.shape[0])
                    over_budget = cur + n > self._budget
                    over_cuts = self._max_cuts is not None and len(buf_audio) >= self._max_cuts
                    if buf_audio and (over_budget or over_cuts):
                        b = _emit()
                        if b is not None:
                            yield b
                    buf_audio.append(wav)
                    buf_len.append(n)
                    buf_cid.append(cid)
                    cur += n

            b = _emit()
            if b is not None:
                yield b
        finally:
            if pool is not None:
                pool.shutdown()
