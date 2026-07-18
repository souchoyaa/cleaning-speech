"""Batched audio loader for Lhotse Shar with duration bucketing.

Yields ``AudioBatch`` (padded ``(B, T_max)`` tensor + sample-count lengths +
full Lhotse cuts). Uses ``DynamicBucketingSampler`` so intra-batch padding
stays small — important because metric forwards run on the padded GPU
tensor.
"""

import logging
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
from lhotse.cut import MixedCut, PaddingCut

from audio_tokenization.pipelines.lhotse.data import (
    build_cutset,
    _set_resampling_backend,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioBatch:
    cut_ids: List[str]
    audio: torch.Tensor    # (B, T_max), CPU, float32 — padded with zeros
    lengths: torch.Tensor  # (B,),       CPU, int64  — valid sample counts
    sr: int
    cuts: list             # list[Cut]


def _is_valid_cut(cut) -> bool:
    # Filters out zero-duration cuts that would crash soundfile during decode.
    if isinstance(cut, MixedCut):
        return all(
            t.cut.duration > 0
            for t in cut.tracks
            if not isinstance(t.cut, PaddingCut)
        )
    return cut.duration > 0


class QualityAssessmentDataset(torch.utils.data.Dataset):
    """Returns padded audio + the original Cut objects (for text/speaker/custom)."""

    def __getitem__(self, cuts):
        from lhotse import CutSet
        from lhotse.dataset.collation import collate_audio

        cuts_list = list(cuts)
        valid = [c for c in cuts_list if _is_valid_cut(c)]
        if len(valid) < len(cuts_list):
            logger.warning(
                "Dropped %d zero-duration cut(s).",
                len(cuts_list) - len(valid),
            )
        if not valid:
            return {
                "inputs": torch.zeros(0, 1),
                "input_lengths": torch.zeros(0, dtype=torch.int64),
                "cuts": [],
            }
        cuts = CutSet.from_cuts(valid)
        try:
            audio, audio_lens = collate_audio(cuts)
        except Exception as e:
            from lhotse.audio.utils import AudioLoadingError
            if not isinstance(e, AudioLoadingError):
                raise
            # Retry cut-by-cut so one bad audio entry doesn't kill the batch.
            survivors, audios, survivor_lens = [], [], []
            for cut in cuts:
                try:
                    a, l = collate_audio(CutSet.from_cuts([cut]))
                    survivors.append(cut)
                    audios.append(a[0])
                    survivor_lens.append(l[0])
                except AudioLoadingError:
                    logger.warning("Skipping unloadable cut %r.", cut.id)
            if not survivors:
                return {
                    "inputs": torch.zeros(0, 1),
                    "input_lengths": torch.zeros(0, dtype=torch.int64),
                    "cuts": [],
                }
            audio = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
            audio_lens = torch.stack(survivor_lens)
            cuts = CutSet.from_cuts(survivors)
        return {"inputs": audio, "input_lengths": audio_lens, "cuts": list(cuts)}


class SharAudioLoader:
    """Iterate ``AudioBatch`` from a Lhotse Shar with dynamic bucketing.

    Shard-level rank split happens inside ``build_cutset``; the sampler then
    operates on the local rank's cuts only.
    """

    def __init__(
        self,
        cfg: dict,
        rank: int = 0,
        world_size: int = 1,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 4,
        dataloader_timeout: int = 300,
    ):
        self._rank = rank
        _set_resampling_backend(rank)

        # Default to audio_only so cuts without text supervisions stay.
        cfg = {**cfg, "mode": cfg.get("mode", "audio_only")}
        cuts = build_cutset(cfg, rank=rank, world_size=world_size)

        from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler

        max_batch_duration = float(cfg.get("max_batch_duration", 300.0))
        num_buckets = int(cfg.get("num_buckets", 20))
        buffer_size = int(cfg.get("bucket_buffer_size", 10000))

        sampler_kwargs = dict(
            max_duration=max_batch_duration,
            num_buckets=num_buckets,
            buffer_size=buffer_size,
            shuffle=bool(cfg.get("sampler_shuffle", True)),
            seed=int(cfg.get("sampler_seed", 42)),
            world_size=1,
            rank=0,
            drop_last=False,
        )
        if cfg.get("max_batch_cuts") is not None:
            sampler_kwargs["max_cuts"] = int(cfg["max_batch_cuts"])
        if cfg.get("quadratic_duration") is not None:
            sampler_kwargs["quadratic_duration"] = float(cfg["quadratic_duration"])

        # When num_shards < world_size in data.py's stride split, this rank
        # may receive zero shards. DynamicBucketingSampler asserts on empty
        # cut sets; short-circuit so the iteration ends cleanly with
        # cuts_seen=0 instead of crashing.
        try:
            sampler = DynamicBucketingSampler(cuts, **sampler_kwargs)
        except AssertionError as e:
            if "buckets" in str(e) or "cuts" in str(e):
                logger.warning(
                    "[rank %d] no cuts assigned (likely num_shards<world_size); "
                    "skipping iteration: %s", rank, e,
                )
                self._empty = True
                self._dataloader = None
                return
            raise
        self._empty = False
        logger.info(
            "[rank %d] DynamicBucketingSampler: max_duration=%.1fs num_buckets=%d",
            rank, max_batch_duration, num_buckets,
        )

        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            gpu_count = max(torch.cuda.device_count(), 1)
            num_workers = min(4, cpu_count // gpu_count)

        self._dataloader = torch.utils.data.DataLoader(
            QualityAssessmentDataset(),
            sampler=sampler,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            timeout=dataloader_timeout if num_workers > 0 else 0,
        )

    def __iter__(self) -> Iterator[AudioBatch]:
        if getattr(self, "_empty", False):
            return
        for batch in self._dataloader:
            cuts: list = batch["cuts"]
            yield AudioBatch(
                cut_ids=[cut.id for cut in cuts],
                audio=batch["inputs"],
                lengths=batch["input_lengths"],
                sr=cuts[0].sampling_rate,
                cuts=cuts,
            )
