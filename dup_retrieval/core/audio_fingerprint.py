"""Stage D — audio fingerprint (Wang 2003 / Moshi Appendix B).

For every cut in the candidate set (Stage C output), compute a constellation-
based fingerprint:

  1. Mel spectrogram on GPU at 40 Hz frame rate, 64 bins in 200-3000 Hz.
  2. Constellation map = AND of three filters (energy, time-local-max,
     freq-argmax-per-time).
  3. Quantize keypoint timestamps to 12.5 Hz so the (m=4, M=20) window spans
     ~3.2 s as in the paper.
  4. For each keypoint, find forward + backward partners in [m, M) and emit
     the triplet hash s = (fb, fk, ff, Δtb, Δtf) packed into 26 bits / int64.

Resume: per-rank ``.cutids`` sidecar listing already-processed (dataset,cut_id).

Inputs  : <output_dir>/text_dedup/clusters.parquet   (the whitelist)
          shar_dirs                                        (audio source)
Outputs : <output_dir>/fingerprint/part_{rank:04d}.parquet
              schema: (dataset, cut_id, hash, t_hash, n_hashes_total)
                      one row per signature; sentinel row (hash=NULL, t_hash=-1)
                      for cuts with zero keypoints.
          <output_dir>/fingerprint/part_{rank:04d}.parquet.cutids
          <output_dir>/fingerprint/_SUCCESS  (rank-0 finalize)
"""

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SUCCESS_MARKER = "_SUCCESS"


# ---------------------------------------------------------------------------
# Hash packing
# ---------------------------------------------------------------------------

# Bit layout (26 bits total):
#   bits  0..3  : Δtf  (4 bits, 0..15)   — tf - tk in 12.5 Hz frames
#   bits  4..7  : Δtb  (4 bits, 0..15)   — tk - tb in 12.5 Hz frames
#   bits  8..13 : ff   (6 bits, 0..63)   — forward keypoint freq band
#   bits 14..19 : fk   (6 bits, 0..63)   — center keypoint freq band
#   bits 20..25 : fb   (6 bits, 0..63)   — backward keypoint freq band
#
# Stored as int64 in parquet (room for future expansion).

def _pack_hash(fb: int, fk: int, ff: int, dtb: int, dtf: int) -> int:
    return (
        (fb  & 0x3F) << 20 |
        (fk  & 0x3F) << 14 |
        (ff  & 0x3F) <<  8 |
        (dtb & 0x0F) <<  4 |
        (dtf & 0x0F)
    )


# ---------------------------------------------------------------------------
# Constellation extraction
# ---------------------------------------------------------------------------


def _build_mel(target_sr: int, n_mels: int, f_min: float, f_max: float,
               frame_rate: float, device: torch.device):
    """``torchaudio.transforms.MelSpectrogram`` configured for the paper's
    40 Hz frame rate.

    hop_length = sr / frame_rate.  For 16 kHz / 40 Hz, hop = 400.
    n_fft = 2 * hop_length keeps overlap = 50%.
    """
    import torchaudio
    hop = int(round(target_sr / frame_rate))
    n_fft = max(int(2 ** np.ceil(np.log2(2 * hop))), 256)
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        f_min=float(f_min),
        f_max=float(f_max),
        power=2.0,
    ).to(device)


def constellation_map(
    mel: torch.Tensor,         # (B, F, T) on GPU
    lengths_frames: torch.Tensor,  # (B,) int64 valid mel-frame counts
    energy_factor: float,
    time_window: int,
    freq_top_k: int = 1,
    peak_mode: str = "global",
    local_norm_time: int = 41,
    silence_rel_floor: float = 0.02,
) -> torch.Tensor:
    """Return a boolean (B, F, T) constellation mask via the three filters.

    Padded frames (beyond lengths_frames[i]) are forced False so they don't
    pollute keypoints.  ``freq_top_k`` keeps the k loudest bands per time frame
    (k=1 = the single max band, the original behavior); k>1 densifies the
    constellation (#9) so short/degraded clips clear Stage-E's min_keypoints floor.

    ``peak_mode``:
      * ``global`` (default) — energy gate vs the whole-clip mean (best for
        FULL-clip dedup; the original behavior).
      * ``local`` — Shazam-style **crop-invariant** gate: energy vs a TIME-LOCAL
        per-band mean (window ``local_norm_time`` frames), so a time crop selects
        the same peaks as the parent (the global mean shifts under cropping; a
        local mean does not).  A relative floor ``silence_rel_floor`` × the clip's
        loudest peak suppresses silence/noise peaks (precision).  Targets
        partial_crop / heavy degradation while keeping full-clip dedup.
    """
    B, Fb, T = mel.shape

    # Valid (non-padded) time mask per clip — reused for the energy mean below
    # and the final keypoint masking.
    if T > 0:
        idx = torch.arange(T, device=mel.device)                 # (T,)
        valid_t = idx[None, :] < lengths_frames[:, None]         # (B, T) bool
    else:
        valid_t = None

    # 1) Energy filter.
    if peak_mode == "local":
        # Crop-invariant (Shazam-style): compare each cell to a TIME-LOCAL per-band
        # mean over a window, not the whole-clip mean.  A time crop barely changes
        # the local mean in the overlapping region, so the same audio selects the
        # same peaks cropped or not (the global mean shifts with the crop).  The
        # relative floor (× the clip's loudest peak) suppresses silence/noise peaks.
        ntW = int(local_norm_time)
        if ntW % 2 == 0:
            ntW += 1                                             # odd kernel -> length-preserving
        local_ref = F.avg_pool2d(
            mel.unsqueeze(1), kernel_size=(1, ntW), stride=1,
            padding=(0, ntW // 2), count_include_pad=False).squeeze(1)
        if valid_t is not None:
            clip_max = (mel * valid_t[:, None, :]).amax(dim=(-2, -1), keepdim=True)
        else:
            clip_max = mel.amax(dim=(-2, -1), keepdim=True)
        energy_mask = (mel > energy_factor * local_ref) & (mel > silence_rel_floor * clip_max)
    else:
        # global (default): per-clip mean over VALID frames (batch-composition
        # independent; for an unpadded clip identical to the plain mean).
        if valid_t is not None:
            valid_cells = (valid_t.sum(dim=1).clamp(min=1) * Fb).to(mel.dtype)  # (B,)
            clip_mean = (mel * valid_t[:, None, :]).sum(dim=(-2, -1)) / valid_cells
            clip_mean = clip_mean[:, None, None]                 # (B, 1, 1)
        else:
            clip_mean = mel.mean(dim=(-2, -1), keepdim=True)
        energy_mask = mel > (energy_factor * clip_mean)

    # 2) Time filter: max over a sliding window in time only (kernel = (1, W)).
    if time_window > 1:
        time_max = F.max_pool2d(
            mel.unsqueeze(1),               # (B, 1, F, T)
            kernel_size=(1, time_window),
            stride=1,
            padding=(0, time_window // 2),
        ).squeeze(1)
        time_mask = mel == time_max
    else:
        time_mask = torch.ones_like(mel, dtype=torch.bool)

    # 3) Frequency filter: keep the top-k loudest bands per time frame.
    if freq_top_k <= 1:
        freq_max = mel.max(dim=-2, keepdim=True).values
        freq_mask = mel == freq_max
    else:
        k = min(int(freq_top_k), Fb)
        kth = mel.topk(k, dim=-2).values[:, -1:, :]   # (B,1,T) = k-th largest per frame
        freq_mask = mel >= kth                        # the k loudest bands

    mask = energy_mask & time_mask & freq_mask              # (B, F, T)

    # Zero out padded frames per clip.
    if valid_t is not None:
        mask = mask & valid_t[:, None, :]
    return mask


def _topk_per_bin(t_hash: np.ndarray, ff: np.ndarray, energies: np.ndarray,
                  max_per_bin: int) -> Tuple[np.ndarray, np.ndarray]:
    """Within each t_hash bin keep up to ``max_per_bin`` highest-energy keypoints.

    Returns ``(t_kept, f_kept)`` sorted by t_hash ascending.  ``max_per_bin=1``
    reproduces the original "one (highest-energy) peak per bin".  Pure numpy so it
    is unit-testable without torch.
    """
    if t_hash.size == 0:
        return t_hash, ff
    order = np.lexsort((-energies, t_hash))   # t_hash asc, energy desc within bin
    t_s = t_hash[order]
    f_s = ff[order]
    is_new = np.empty(t_s.shape, dtype=bool)
    is_new[0] = True
    is_new[1:] = t_s[1:] != t_s[:-1]
    if max_per_bin <= 1:
        keep = is_new
    else:
        pos = np.arange(t_s.size)
        group_start = np.maximum.accumulate(np.where(is_new, pos, 0))
        keep = (pos - group_start) < max_per_bin   # rank within bin (0 = loudest)
    t_kept = t_s[keep]
    f_kept = f_s[keep]
    s = np.argsort(t_kept, kind="stable")          # final t_hash-ascending order
    return t_kept[s], f_kept[s]


def extract_keypoints_per_cut(
    mask: torch.Tensor,        # (B, F, T) bool
    mel: torch.Tensor,         # (B, F, T) float — for energy disambiguation
    hash_frame_rate: float,
    mel_frame_rate: float,
    max_per_bin: int = 1,
) -> List[List[Tuple[int, int]]]:
    """For each cut in the batch return ``[(t_hash, freq), ...]`` sorted by t_hash.

    Keypoints colliding on the same 12.5 Hz t_hash bin: keep the up-to
    ``max_per_bin`` highest-energy ones (k=1 = the original single-peak-per-bin;
    k>1 densifies the fingerprint, #9).
    """
    B, Fb, T = mask.shape
    # Move to CPU once — keypoint counts per cut are O(seconds), small.
    mask_cpu = mask.cpu().numpy()
    mel_cpu  = mel.cpu().numpy()
    ratio = float(hash_frame_rate) / float(mel_frame_rate)   # 12.5/40 = 0.3125

    out: List[List[Tuple[int, int]]] = []
    for b in range(B):
        ff, tt = np.nonzero(mask_cpu[b])         # both (K,) int
        if ff.size == 0:
            out.append([])
            continue
        t_hash = np.floor(tt.astype(np.float64) * ratio).astype(np.int64)
        energies = mel_cpu[b, ff, tt]
        t_kept, f_kept = _topk_per_bin(t_hash, ff.astype(np.int64), energies, max_per_bin)
        out.append(list(zip(t_kept.tolist(), f_kept.tolist())))
    return out


def encode_signatures(
    keypoints: List[Tuple[int, int]],   # sorted by t_hash
    m: int,
    M: int,
) -> List[Tuple[int, int]]:
    """Return ``[(hash_int, t_hash), ...]`` for one cut.

    For each keypoint k, find:
      - forward partner f: smallest j>k with t_j in [tk+m, tk+M)
      - backward partner b: largest j<k with t_j in (tk-M, tk-m]
    Emit a signature only if both exist.
    """
    n = len(keypoints)
    if n < 3:
        return []
    out: List[Tuple[int, int]] = []
    ts = [kp[0] for kp in keypoints]
    fs = [kp[1] for kp in keypoints]

    # Two-pointer scan forward and backward.
    j_lo_b = 0     # backward window low end (oldest still in window)
    for k in range(n):
        tk, fk = ts[k], fs[k]

        # Backward: largest index b<k with ts[b] in (tk-M, tk-m].
        while j_lo_b < k and ts[j_lo_b] <= tk - M:
            j_lo_b += 1
        b_idx = -1
        for j in range(k - 1, j_lo_b - 1, -1):
            if ts[j] <= tk - m:
                b_idx = j
                break

        # Forward: smallest index f>k with ts[f] in [tk+m, tk+M).
        f_idx = -1
        for j in range(k + 1, n):
            if ts[j] >= tk + m and ts[j] < tk + M:
                f_idx = j
                break
            if ts[j] >= tk + M:
                break

        if b_idx < 0 or f_idx < 0:
            continue
        dtb = tk - ts[b_idx]
        dtf = ts[f_idx] - tk
        h = _pack_hash(fs[b_idx], fk, fs[f_idx], dtb, dtf)
        out.append((h, tk))
    return out


# ---------------------------------------------------------------------------
# Resume sidecar
# ---------------------------------------------------------------------------


def _cutids_sidecar_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(parquet_path.suffix + ".cutids")


def _load_seen(parquet_path: Path) -> Set[Tuple[str, str]]:
    """Read the sidecar; on miss, parse the parquet itself + rebuild."""
    sc = _cutids_sidecar_path(parquet_path)
    seen: Set[Tuple[str, str]] = set()
    if sc.exists():
        with open(sc) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                ds, _, cid = line.partition("\t")
                if ds and cid:
                    seen.add((ds, cid))
        logger.info("Resume: loaded %d (dataset,cut_id) from %s",
                    len(seen), sc.name)
        return seen
    if parquet_path.exists():
        try:
            t = pq.read_table(parquet_path, columns=["dataset", "cut_id"])
            for ds, cid in zip(t.column("dataset").to_pylist(),
                                t.column("cut_id").to_pylist()):
                seen.add((ds, cid))
        except Exception:
            logger.warning("Could not read %s for resume — starting fresh.", parquet_path)
            seen = set()
        if seen:
            with open(sc, "w") as f:
                for ds, cid in seen:
                    f.write(f"{ds}\t{cid}\n")
            logger.info("Resume: rebuilt sidecar from parquet (%d items).", len(seen))
    return seen


def _append_to_sidecar(parquet_path: Path, items: List[Tuple[str, str]]) -> None:
    if not items:
        return
    sc = _cutids_sidecar_path(parquet_path)
    with open(sc, "a") as f:
        for ds, cid in items:
            f.write(f"{ds}\t{cid}\n")


# ---------------------------------------------------------------------------
# Per-rank driver
# ---------------------------------------------------------------------------


_FP_SCHEMA = pa.schema([
    pa.field("dataset",          pa.string(), nullable=False),
    pa.field("cut_id",           pa.string(), nullable=False),
    pa.field("hash",             pa.int64(),  nullable=True),   # NULL = sentinel
    pa.field("t_hash",           pa.int32(),  nullable=False),  # -1 = sentinel
    pa.field("n_hashes_total",   pa.int32(),  nullable=False),
])


_stop_requested = False


def _install_signal_handlers(rank: int) -> None:
    def _h(signum, _frame):
        global _stop_requested
        if not _stop_requested:
            logger.warning("[rank %d] caught %s — finishing batch then exiting.",
                           rank, signal.Signals(signum).name)
        _stop_requested = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _h)
        except (OSError, ValueError):
            pass


def _write_shard(shard_path: Path, ds_a: List[str], cid_a: List[str],
                 h_a: List[Optional[int]], t_a: List[int],
                 n_a: List[int]) -> None:
    """Write one flush as a self-contained shard parquet.

    O(rows-in-flush), independent of how much was previously written by
    this rank.  Finalize merges all shards under part_{rank:04d}_*.parquet
    into part_{rank:04d}.parquet at the end.
    """
    tbl = pa.Table.from_arrays(
        [
            pa.array(ds_a),
            pa.array(cid_a),
            pa.array(h_a, type=pa.int64()),
            pa.array(t_a, type=pa.int32()),
            pa.array(n_a, type=pa.int32()),
        ],
        schema=_FP_SCHEMA,
    )
    tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
    pq.write_table(tbl, str(tmp), compression="zstd", compression_level=3,
                   row_group_size=200_000)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, shard_path)


def _merge_rank_shards(fp_dir: Path, rank: int) -> None:
    """Concatenate part_{rank:04d}_*.parquet into part_{rank:04d}.parquet.

    Idempotent: if the merged file already exists and is newer than every
    shard, this is a no-op.  Old shards are deleted only after the merge
    parquet has been atomically renamed into place + fsynced.
    """
    final = fp_dir / f"part_{rank:04d}.parquet"
    shards = sorted(fp_dir.glob(f"part_{rank:04d}_*.parquet"))
    if not shards:
        return
    if final.exists():
        final_mtime = final.stat().st_mtime
        if all(s.stat().st_mtime <= final_mtime for s in shards):
            # Merge already up to date; clean up shards.
            for s in shards:
                try: s.unlink()
                except OSError: pass
            return

    tables = [pq.read_table(s) for s in shards]
    merged = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    tmp = final.with_suffix(final.suffix + ".tmp")
    pq.write_table(merged, str(tmp), compression="zstd", compression_level=3,
                   row_group_size=200_000)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    for s in shards:
        try: s.unlink()
        except OSError: pass


def run_rank(cfg: dict, rank: int, world_size: int) -> None:
    # Defensive: the pipeline runs all stages in one process, so an earlier GPU
    # stage (text_dedup's RMM pool) may still hold the device. Reclaim it before
    # the mel/constellation work so large batches don't CUDA-OOM. Best-effort.
    try:
        import rmm
        rmm.reinitialize(pool_allocator=False)
    except Exception:
        pass
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    output_dir = Path(cfg["output_dir"])
    text_clusters = output_dir / "text_dedup" / "clusters.parquet"
    if not text_clusters.exists():
        raise RuntimeError(f"Missing Stage C output: {text_clusters}")
    if not (output_dir / "text_dedup" / SUCCESS_MARKER).exists():
        raise RuntimeError("Stage C not finalized.  Run text_dedup.py first.")

    fp_dir = output_dir / "fingerprint"
    fp_dir.mkdir(parents=True, exist_ok=True)
    for p in fp_dir.glob("*.parquet.tmp"):
        try:
            p.unlink()
        except OSError:
            pass

    out_path = fp_dir / f"part_{rank:04d}.parquet"
    # If a previous run produced a merged final, treat it as resume input.
    # Otherwise, scan any existing shards for what's already done.
    seen = _load_seen(out_path)
    if not seen:
        # Sidecar was empty; also check shards from a previous run.
        for shard in sorted(fp_dir.glob(f"part_{rank:04d}_*.parquet")):
            try:
                t = pq.read_table(shard, columns=["dataset", "cut_id"])
                for d, c in zip(t.column("dataset").to_pylist(),
                                 t.column("cut_id").to_pylist()):
                    seen.add((d, c))
            except Exception:
                logger.warning("Could not read existing shard %s", shard)
        if seen:
            logger.info("[rank %d] Resume: %d cuts found in shards (rebuilding sidecar).",
                        rank, len(seen))
            sc = _cutids_sidecar_path(out_path)
            with open(sc, "w") as f:
                for ds, cid in seen:
                    f.write(f"{ds}\t{cid}\n")

    cfg_d = cfg.get("audio_fingerprint", {})
    n_mels         = int(cfg_d.get("mel_n_bins", 64))
    f_min          = float(cfg_d.get("mel_f_min", 200.0))
    f_max          = float(cfg_d.get("mel_f_max", 3000.0))
    mel_fr         = float(cfg_d.get("mel_frame_rate", 40.0))
    hash_fr        = float(cfg_d.get("hash_frame_rate", 12.5))
    m_frames       = int(cfg_d.get("m_frames", 4))
    M_frames       = int(cfg_d.get("M_frames", 20))
    time_window    = int(cfg_d.get("time_window", 9))
    energy_factor  = float(cfg_d.get("energy_factor", 1.0))
    # #9 — constellation density.  Both default to 1 (the original sparse
    # behavior); raise together to densify (top-k bands per frame + up to k
    # keypoints per t_hash bin) so short/degraded clips clear min_keypoints.
    freq_top_k        = int(cfg_d.get("freq_top_k", 1))
    keypoints_per_bin = int(cfg_d.get("keypoints_per_bin", 1))
    # Shazam-style crop-invariant keypoint selection (default "global" = original).
    peak_mode         = str(cfg_d.get("peak_mode", "global")).lower()
    local_norm_time   = int(cfg_d.get("local_norm_time", 41))
    silence_rel_floor = float(cfg_d.get("silence_rel_floor", 0.02))

    target_sr      = int(cfg.get("target_sample_rate", 16000))
    truncate_secs  = cfg.get("truncate_secs")

    # ----- Whitelist -----
    from .loader_audio import (
        load_whitelist, whitelist_for_dataset, load_locator,
        RandomAccessSharReader, CandidateAudioBatch,
    )
    whitelist = load_whitelist(text_clusters)
    logger.info("[rank %d] candidate whitelist size: %d", rank, len(whitelist))

    # Stage-A tar locator (one projected scan) for random-access reads.
    locator = load_locator(output_dir / "manifest", whitelist)
    logger.info("[rank %d] random-access locator covers %d dataset(s).",
                rank, len(locator))

    # ----- Set device -----
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    logger.info("[rank %d] device=%s", rank, device)

    mel_xform = _build_mel(target_sr, n_mels, f_min, f_max, mel_fr, device)

    _install_signal_handlers(rank)

    # ----- Iterate per-shar (so we always know the dataset tag) -----
    shar_dirs = cfg["shar_dirs"]
    if isinstance(shar_dirs, str):
        shar_dirs = [shar_dirs]

    overrides    = cfg.get("dataset_overrides") or {}
    dataset_root = cfg.get("dataset_root")

    from .manifest import _derive_dataset_name  # type: ignore

    shar_specs: List[Tuple[str, str]] = []   # (shar_dir_abs, dataset_tag)
    for sd in shar_dirs:
        sp = str(Path(sd).resolve())
        ds = overrides.get(sp, _derive_dataset_name(sp, dataset_root))
        shar_specs.append((sp, ds))

    n_processed = 0
    n_sentinels = 0
    flush_every_secs = 30.0
    last_flush = time.time()

    pending_ds:  List[str] = []
    pending_cid: List[str] = []
    pending_h:   List[Optional[int]] = []
    pending_t:   List[int] = []
    pending_n:   List[int] = []
    pending_seen_keys: List[Tuple[str, str]] = []
    shard_idx = 0
    # Pick a starting shard_idx that doesn't collide with existing shards.
    for existing in sorted(fp_dir.glob(f"part_{rank:04d}_*.parquet")):
        try:
            n = int(existing.stem.split("_")[-1])
            shard_idx = max(shard_idx, n + 1)
        except ValueError:
            pass

    def _flush() -> None:
        nonlocal pending_ds, pending_cid, pending_h, pending_t, pending_n
        nonlocal pending_seen_keys, last_flush, shard_idx
        if not pending_ds:
            return
        shard_path = fp_dir / f"part_{rank:04d}_{shard_idx:06d}.parquet"
        _write_shard(shard_path, pending_ds, pending_cid, pending_h, pending_t, pending_n)
        _append_to_sidecar(out_path, pending_seen_keys)
        shard_idx += 1
        pending_ds, pending_cid = [], []
        pending_h, pending_t, pending_n = [], [], []
        pending_seen_keys = []
        last_flush = time.time()

    for shar_dir, dataset in shar_specs:
        cut_id_set = whitelist_for_dataset(whitelist, dataset)
        if not cut_id_set:
            logger.info("[rank %d] no candidates for dataset %s — skipping shar %s",
                        rank, dataset, shar_dir)
            continue

        per_shar_cfg = dict(cfg)
        per_shar_cfg["shar_dir"] = shar_dir

        ds_locator = locator.get(dataset)
        if not ds_locator:
            raise RuntimeError(
                f"No tar locator for dataset {dataset!r} — the manifest predates "
                "the random-access locator; re-run the manifest stage.")
        logger.info("[rank %d] dataset %s: random-access reader "
                    "(%d located candidates).", rank, dataset, len(ds_locator))
        loader = RandomAccessSharReader(
            cfg=per_shar_cfg,
            dataset=dataset,
            shar_dir=shar_dir,
            locator=ds_locator,
            rank=rank,
            world_size=world_size,
            num_workers=int(cfg.get("decode_num_workers", 4)),
            skip_keys=seen,
        )

        for batch in loader:
            if _stop_requested:
                break

            # Resume filter — drop already-processed cuts.
            new_idx = [i for i, c in enumerate(batch.cut_ids)
                       if (batch.dataset, c) not in seen]
            if not new_idx:
                continue

            # Truncate clips on host (cheap).
            if truncate_secs is not None:
                T_max_samples = int(float(truncate_secs) * target_sr)
                audio = batch.audio[:, :T_max_samples]
                lengths = torch.minimum(batch.lengths,
                                        torch.tensor(T_max_samples, dtype=torch.int64))
            else:
                audio = batch.audio
                lengths = batch.lengths

            audio_gpu = audio.to(device, non_blocking=True)
            lengths_gpu = lengths.to(device, non_blocking=True)

            with torch.no_grad():
                mel = mel_xform(audio_gpu)              # (B, F, T)
            # Keep only the rows we still need (post-resume filter).
            if len(new_idx) != mel.shape[0]:
                idx_t = torch.tensor(new_idx, device=device)
                mel = mel.index_select(0, idx_t)
                lengths_gpu = lengths_gpu.index_select(0, idx_t)
                cut_ids_b = [batch.cut_ids[i] for i in new_idx]
            else:
                cut_ids_b = list(batch.cut_ids)

            # Mel-frame counts per clip = ceil(samples / hop).
            hop = int(round(target_sr / mel_fr))
            mel_frames = ((lengths_gpu.float() + (hop - 1)) / hop).long()

            with torch.no_grad():
                mask = constellation_map(mel, mel_frames, energy_factor, time_window,
                                         freq_top_k=freq_top_k, peak_mode=peak_mode,
                                         local_norm_time=local_norm_time,
                                         silence_rel_floor=silence_rel_floor)
                kp_per_cut = extract_keypoints_per_cut(mask, mel, hash_fr, mel_fr,
                                                       max_per_bin=keypoints_per_bin)

            for cid, kp in zip(cut_ids_b, kp_per_cut):
                seen_key = (batch.dataset, cid)
                if seen_key in seen:
                    continue
                if not kp:
                    pending_ds.append(batch.dataset); pending_cid.append(cid)
                    pending_h.append(None); pending_t.append(-1); pending_n.append(0)
                    pending_seen_keys.append(seen_key); seen.add(seen_key)
                    n_sentinels += 1
                    continue
                sigs = encode_signatures(kp, m=m_frames, M=M_frames)
                if not sigs:
                    pending_ds.append(batch.dataset); pending_cid.append(cid)
                    pending_h.append(None); pending_t.append(-1); pending_n.append(0)
                    pending_seen_keys.append(seen_key); seen.add(seen_key)
                    n_sentinels += 1
                    continue
                n_total = len(sigs)
                for h_int, t_h in sigs:
                    pending_ds.append(batch.dataset); pending_cid.append(cid)
                    pending_h.append(int(h_int))
                    pending_t.append(int(t_h))
                    pending_n.append(int(n_total))
                pending_seen_keys.append(seen_key); seen.add(seen_key)
            n_processed += len(cut_ids_b)

            if (time.time() - last_flush) >= flush_every_secs:
                _flush()

        _flush()
        if _stop_requested:
            break

    _flush()

    # Each rank merges its own shards into part_{rank:04d}.parquet.  Cheap
    # (single sequential scan + concat); avoids cross-rank coordination
    # in finalize().
    _merge_rank_shards(fp_dir, rank)

    # A rank with nothing to do still writes an empty part — a missing part
    # file would hang the rank barrier in pipeline.py forever.
    if not out_path.exists():
        pq.write_table(_FP_SCHEMA.empty_table(), str(out_path))

    logger.info("[rank %d] Stage D done: %d cuts processed (%d sentinels).",
                rank, n_processed, n_sentinels)


def finalize(cfg: dict) -> None:
    """Rank-0 only: write _SUCCESS once every part_*.parquet exists."""
    fp_dir = Path(cfg["output_dir"]) / "fingerprint"
    parts = sorted(fp_dir.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError(f"Stage D: no parts in {fp_dir}")

    n_rows = 0
    n_sentinels = 0
    for p in parts:
        t = pq.read_table(p, columns=["hash"])
        n_rows += t.num_rows
        n_sentinels += sum(1 for v in t.column("hash").to_pylist() if v is None)

    from . import run_layout
    run_layout.finalize_stage(fp_dir.parent, "fingerprint", rows=n_rows,
                              extra={"parts": [p.name for p in parts],
                                     "sentinels": n_sentinels})
    logger.info("Wrote %s", fp_dir / SUCCESS_MARKER)


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage D: audio fingerprint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank", type=int,
                        default=int(os.environ.get("RANK",
                                    os.environ.get("SLURM_PROCID", 0))))
    parser.add_argument("--world-size", type=int,
                        default=int(os.environ.get("WORLD_SIZE",
                                    os.environ.get("SLURM_NTASKS", 1))))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = _load_cfg(args.config)
    if args.finalize:
        finalize(cfg)
    else:
        run_rank(cfg, rank=args.rank, world_size=args.world_size)


if __name__ == "__main__":
    main()
