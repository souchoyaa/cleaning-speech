"""NeMo Frame-VAD (MarbleNet v2.0) worker — runs on the local GPU node.

Frame-VAD v2.0 emits sigmoid probabilities of shape ``(B, T_frames)`` at a
fixed 20 ms hop (the head is 2× strided). The legacy clip-level
``vad_multilingual_marblenet`` is NOT supported.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "nvidia/frame_vad_multilingual_marblenet_v2.0"
DEFAULT_THRESHOLD = 0.5     # P(speech) frame cutoff
DEFAULT_MIN_SPEECH_MS = 250  # drop spans shorter than this
DEFAULT_MIN_SILENCE_MS = 250  # merge spans separated by less

_FRAME_HOP_SEC = 0.02        # fixed 20 ms hop in v2.0


class VadWorker:
    """Owns one loaded Frame-VAD model + per-call threshold/merge knobs."""

    def __init__(self, cfg: dict, device: torch.device) -> None:
        self._device = device
        self._model_name = str(cfg.get("model", DEFAULT_MODEL))
        logger.info("VadWorker loading: model=%s", self._model_name)
        self._model = _build_vad(self._model_name, device)

    def run(
        self,
        audio_list: list,
        sr: int,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
        min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
    ) -> list:
        """Batched VAD inference. Returns ``[{speech_pct, speech_spans}, ...]``."""
        return _run_vad(
            self._model, audio_list, sr,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
        )


def _build_vad(model_name: str, device: torch.device):
    import nemo.collections.asr as nemo_asr

    # strict=False works around the loss.weight state_dict mismatch
    # documented on HF (Frame_VAD_Multilingual_MarbleNet_v2.0/discussions/3).
    model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
        model_name=model_name, map_location=device, strict=False,
    )
    model.eval()
    model.freeze()
    return model


@torch.inference_mode()
def _run_vad(
    model,
    audio_list: list,
    sr: int,
    *,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
) -> list:
    if not audio_list:
        return []

    device = next(model.parameters()).device
    lengths = [len(a) for a in audio_list]
    T_max = max(lengths)
    B = len(audio_list)

    audio_padded = torch.zeros(B, T_max, dtype=torch.float32, device=device)
    for i, a in enumerate(audio_list):
        audio_padded[i, :len(a)] = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    lengths_t = torch.tensor(lengths, device=device, dtype=torch.long)

    probs = model(input_signal=audio_padded, input_signal_length=lengths_t)
    if probs.dim() == 3 and probs.shape[-1] == 2:
        # Defensive: future checkpoints may revert to (B, T, 2) softmax logits.
        probs = torch.softmax(probs, dim=-1)[:, :, 1]
    elif probs.dim() != 2:
        raise RuntimeError(
            f"Unexpected Frame-VAD output shape {tuple(probs.shape)}",
        )

    probs_np = probs.cpu().numpy()
    T_frames = probs_np.shape[1]

    min_speech_frames = max(1, int(min_speech_ms * 0.001 / _FRAME_HOP_SEC))
    min_silence_frames = max(1, int(min_silence_ms * 0.001 / _FRAME_HOP_SEC))

    out: list = []
    for i, n_samples in enumerate(lengths):
        n_frames = min(int(n_samples / sr / _FRAME_HOP_SEC), T_frames)
        if n_frames <= 0:
            out.append({"speech_pct": 0.0, "speech_spans": []})
            continue

        is_speech = probs_np[i, :n_frames] >= threshold

        spans: list = []
        in_span = False
        span_start = 0
        for k in range(n_frames):
            if is_speech[k] and not in_span:
                in_span = True
                span_start = k
            elif not is_speech[k] and in_span:
                in_span = False
                spans.append([span_start, k])
        if in_span:
            spans.append([span_start, n_frames])

        merged: list = []
        for s in spans:
            if merged and s[0] - merged[-1][1] < min_silence_frames:
                merged[-1][1] = s[1]
            else:
                merged.append(list(s))
        merged = [s for s in merged if s[1] - s[0] >= min_speech_frames]

        speech_spans = [
            [round(s[0] * _FRAME_HOP_SEC, 3), round(s[1] * _FRAME_HOP_SEC, 3)]
            for s in merged
        ]
        speech_pct = round(float(is_speech.mean()), 4)
        out.append({"speech_pct": speech_pct, "speech_spans": speech_spans})

    return out


def make_silent_record(duration: float) -> dict:
    """Synthesise a VAD record for a cut we never ran the model on."""
    return {"speech_pct": 0.0, "speech_spans": []}
