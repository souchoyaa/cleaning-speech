"""Parakeet-TDT worker — local NeMo forward with OOM-aware micro-batching.

Disables the cuda-graph TDT decoder (NeMo issues #15164 / #15423) and
exposes word-level timestamps via NeMo's frame-offset → seconds conversion.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from omegaconf import open_dict

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_MICRO_BATCH = 32

_EMPTY_HYP = {"text": "", "avg_logp": 0.0, "word_timestamps": []}


def _is_likely_transcribable(audio: np.ndarray) -> bool:
    """Reject only mathematically silent / empty clips (would crash NeMo's
    timestamp post-processing). Anything with a single nonzero sample is sent."""
    if audio.size == 0:
        return False
    return bool(audio.any())


class ParakeetWorker:
    """Owns one loaded Parakeet model + micro-batch knob."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        self._device = device
        self._model_name = str(cfg.get("model", DEFAULT_MODEL))
        self._micro = int(cfg.get("nemo_micro_batch_size", DEFAULT_MICRO_BATCH))
        dtype_name = str(cfg.get("dtype", "bfloat16")).lower()
        self._dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)
        self._compute_word_ts = bool(cfg.get("compute_word_timestamps", False))

        logger.info(
            "ParakeetWorker loading: model=%s micro=%d dtype=%s timestamps=%s",
            self._model_name, self._micro, self._dtype, self._compute_word_ts,
        )
        self._model, self._sec_per_frame = _build_parakeet(
            self._model_name, device,
            compute_word_timestamps=self._compute_word_ts,
        )

    def transcribe(self, audio_list: list) -> list:
        """Run a macro-batch (caller sorts longest-first) through micro-batches."""
        if not audio_list:
            return []

        if self._compute_word_ts:
            keep_mask = [_is_likely_transcribable(a) for a in audio_list]
            keep_audio = [a for a, k in zip(audio_list, keep_mask) if k]
            hyps_kept = (
                _transcribe_macro(self._model, keep_audio, self._micro,
                                  self._dtype, self._sec_per_frame, True)
                if keep_audio else []
            )
            kept_iter = iter(hyps_kept)
            return [
                next(kept_iter) if k else dict(_EMPTY_HYP)
                for k in keep_mask
            ]

        return _transcribe_macro(
            self._model, audio_list, self._micro, self._dtype,
            self._sec_per_frame, compute_ts=False,
            catch_empty_hyp_indexerror=False,
        )


def _build_parakeet(
    model_name: str,
    device: torch.device,
    *,
    compute_word_timestamps: bool,
) -> tuple:
    """Load Parakeet-TDT and switch to the streaming (non-cuda-graph) decoder.

    NeMo defaults to a cuda-graph TDT decoder which leaks LabelLoopingState
    (issue #15423) and is slower than streaming when timestamps are on
    (issue #15164). ``decoding_computer.disable_cuda_graphs()`` is the
    canonical workaround.
    """
    import nemo.collections.asr as nemo_asr

    logger.info("Loading Parakeet (%s)…", model_name)
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_name, map_location=device,
    )
    model.eval()
    model.freeze()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy_batch"
        # preserve_alignments=False closes the BatchedAlignments OOM path
        # regardless of the cuda-graph state.
        decoding_cfg.preserve_alignments = False
        decoding_cfg.preserve_frame_confidence = False
        decoding_cfg.compute_timestamps = bool(compute_word_timestamps)
        if compute_word_timestamps:
            for sec in ("greedy", "greedy_batch"):
                sub = decoding_cfg.get(sec)
                if sub is None:
                    from omegaconf import OmegaConf
                    decoding_cfg[sec] = OmegaConf.create({})
                    sub = decoding_cfg[sec]
                with open_dict(sub):
                    sub.use_cuda_graph_decoder = False
                    sub.allow_cuda_graphs = False
    model.change_decoding_strategy(decoding_cfg)

    if compute_word_timestamps:
        try:
            model.decoding.decoding.decoding_computer.disable_cuda_graphs()
            logger.info("CUDA graphs disabled on TDT decoding_computer.")
        except AttributeError:
            logger.warning(
                "decoding_computer.disable_cuda_graphs() not available — "
                "relying on cfg flags.",
            )

    sec_per_frame = float(model.cfg.preprocessor.window_stride) * int(
        model.encoder.subsampling_factor,
    )
    logger.info(
        "Parakeet word timestamps: %s (s/frame=%.4f)",
        "enabled" if compute_word_timestamps else "disabled", sec_per_frame,
    )
    return model, sec_per_frame


def _extract_word_ts(h, sec_per_frame: float) -> list:
    ts = getattr(h, "timestamp", None)
    if not isinstance(ts, dict):
        return []
    out: list = []
    for w in (ts.get("word") or []):
        text = w.get("word")
        so = w.get("start_offset")
        eo = w.get("end_offset")
        if text is None or so is None or eo is None:
            continue
        out.append({
            "w": str(text).strip(),
            "s": round(float(so) * sec_per_frame, 3),
            "e": round(float(eo) * sec_per_frame, 3),
        })
    return out


@torch.inference_mode()
def _transcribe_subbatch(
    model,
    audio_list: list,
    batch_size: int,
    dtype: torch.dtype,
    sec_per_frame: float,
    compute_ts: bool,
) -> list:
    """One micro-batch under bf16 autocast.

    Don't pass ``timestamps=True`` here — ``compute_timestamps`` is already
    set in the decoding cfg. The kwarg triggers ``change_decoding_strategy``
    per call which silently re-enables cuda graphs.
    """
    with torch.autocast(device_type="cuda", dtype=dtype):
        hyps = model.transcribe(
            audio=audio_list,
            batch_size=batch_size,
            return_hypotheses=True,
            verbose=False,
        )

    out: list = []
    for h in hyps:
        # h.y_sequence is a 1-D LongTensor; use len() instead of `or []`
        # which triggers bool(tensor) and raises on tensors with >1 elem.
        y_seq = getattr(h, "y_sequence", None)
        y_len = len(y_seq) if y_seq is not None else 0
        avg_logp = float(getattr(h, "score", 0.0)) / max(y_len, 1)
        out.append({
            "text": getattr(h, "text", "") or "",
            "avg_logp": round(avg_logp, 4),
            "word_timestamps": _extract_word_ts(h, sec_per_frame) if compute_ts else [],
        })
    return out


def _transcribe_macro(
    model,
    audio_list: list,
    micro: int,
    dtype: torch.dtype,
    sec_per_frame: float,
    compute_ts: bool,
    *,
    catch_empty_hyp_indexerror: bool = True,
) -> list:
    """Process a macro-batch in micro chunks with OOM split-retry and
    optional silent-clip IndexError fallback.
    """
    out: list = []
    target = max(1, int(micro))
    i = 0
    while i < len(audio_list):
        sub = audio_list[i:i + target]
        try:
            out.extend(_transcribe_subbatch(
                model, sub, len(sub), dtype, sec_per_frame, compute_ts,
            ))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if target == 1:
                raise
            target = max(1, target // 2)
            logger.warning(
                "OOM at micro=%d → retry at micro=%d (remaining %d)",
                target * 2, target, len(audio_list) - i,
            )
            continue
        except IndexError:
            # NeMo bug: empty word_offsets in compute_rnnt_timestamps.
            if not catch_empty_hyp_indexerror:
                raise
            if len(sub) > 1:
                for one in sub:
                    try:
                        out.extend(_transcribe_subbatch(
                            model, [one], 1, dtype, sec_per_frame, compute_ts,
                        ))
                    except IndexError:
                        out.append(dict(_EMPTY_HYP))
            else:
                out.append(dict(_EMPTY_HYP))
            i += len(sub)
            continue
        i += len(sub)
    return out
