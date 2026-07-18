"""Canary-1B-v2 worker — local NeMo forward with NFA word timestamps.

Canary AED can't emit frame-aligned word boundaries directly — its .nemo
bundle ships an auxiliary CTC head that NFA uses for timestamp alignment
when ``timestamps=True`` is passed. Requires NeMo ≥ 2.5.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "nvidia/canary-1b-v2"
DEFAULT_MICRO_BATCH = 16

_EMPTY_HYP = {"text": "", "avg_logp": 0.0, "word_timestamps": []}


def _is_likely_transcribable(audio: np.ndarray) -> bool:
    if audio.size == 0:
        return False
    return bool(audio.any())


def _resolve_languages(cuts: list, cfg_default) -> list:
    """Per-cut language: cfg default > cut.custom['language'] > None."""
    if cfg_default:
        return [cfg_default] * len(cuts)
    out: list = []
    for c in cuts:
        custom = getattr(c, "custom", None) or {}
        out.append(custom.get("language"))
    return out


class CanaryWorker:
    """Owns one loaded Canary model + micro-batch / language knobs."""

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
        self._cfg_language = cfg.get("language")
        self._compute_word_ts = bool(cfg.get("compute_word_timestamps", True))

        logger.info(
            "CanaryWorker loading: model=%s micro=%d dtype=%s lang=%s timestamps=%s",
            self._model_name, self._micro, self._dtype,
            self._cfg_language, self._compute_word_ts,
        )
        self._model = _build_canary(
            self._model_name, device, self._dtype,
            compute_word_timestamps=self._compute_word_ts,
        )

    def resolve_languages(self, cuts: list) -> list:
        return _resolve_languages(cuts, self._cfg_language)

    def transcribe(self, audio_list: list, languages: list) -> list:
        """Macro-batch with per-cut language hints (caller sorts longest-first)."""
        if not audio_list:
            return []

        if self._compute_word_ts:
            keep_mask = [_is_likely_transcribable(a) for a in audio_list]
            keep_audio = [a for a, k in zip(audio_list, keep_mask) if k]
            keep_langs = [l for l, k in zip(languages, keep_mask) if k]
            hyps_kept = (
                _transcribe_macro(
                    self._model, keep_audio, keep_langs, self._micro,
                    self._compute_word_ts,
                )
                if keep_audio else []
            )
            kept_iter = iter(hyps_kept)
            out: list = []
            for k, lang in zip(keep_mask, languages):
                if k:
                    out.append(next(kept_iter))
                else:
                    empty = dict(_EMPTY_HYP)
                    if lang is not None:
                        empty["language"] = lang
                    out.append(empty)
            return out

        return _transcribe_macro(
            self._model, audio_list, languages, self._micro,
            self._compute_word_ts,
            catch_empty_hyp_indexerror=False,
        )


def _build_canary(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    compute_word_timestamps: bool,
):
    """Load Canary-1b-v2 and (optionally) cast weights to bf16."""
    import nemo.collections.asr as nemo_asr

    logger.info("Loading Canary (%s)…", model_name)
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_name, map_location=device,
    )
    model.eval()
    model.freeze()

    if compute_word_timestamps:
        # Two cfg edits required for Canary AED to emit timestamp tokens:
        # strategy='greedy' (beam path doesn't emit them) and
        # compute_timestamps=True. Both must be set BEFORE
        # change_decoding_strategy() to survive transcribe()'s rebuilds.
        try:
            from omegaconf import open_dict
            with open_dict(model.cfg.decoding):
                model.cfg.decoding.strategy = "greedy"
                model.cfg.decoding.compute_timestamps = True
            model.change_decoding_strategy(model.cfg.decoding)
        except Exception as e:
            logger.warning(
                "Canary decoding cfg failed (%s); transcribe(timestamps=True) "
                "may still work but is not guaranteed.", e,
            )

    if dtype != torch.float32:
        try:
            model = model.to(dtype)
        except (RuntimeError, TypeError) as e:
            logger.warning("Canary cast to %s failed (%s) — staying in fp32.", dtype, e)

    return model


def _extract_word_ts(h) -> list:
    """Pull NFA word timestamps from a Canary hypothesis (already in seconds)."""
    ts = getattr(h, "timestamp", None)
    if not isinstance(ts, dict):
        return []
    out: list = []
    for w in (ts.get("word") or []):
        text = w.get("word")
        s = w.get("start")
        e = w.get("end")
        if text is None or s is None or e is None:
            continue
        out.append({
            "w": str(text).strip(),
            "s": round(float(s), 3),
            "e": round(float(e), 3),
        })
    return out


@torch.inference_mode()
def _transcribe_subbatch(
    model,
    audio_list: list,
    language,
    batch_size: int,
    compute_ts: bool,
) -> list:
    """One micro-batch through ``model.transcribe``. Caller groups by language
    so each call is monolingual (Canary's source_lang/target_lang are scalar
    prompt slots, not per-cut)."""
    kwargs: dict = dict(
        audio=audio_list, batch_size=batch_size,
        return_hypotheses=True, verbose=False,
    )
    if compute_ts:
        kwargs["timestamps"] = True
    if language is not None:
        kwargs["task"] = "asr"
        kwargs["source_lang"] = language
        kwargs["target_lang"] = language
        kwargs["pnc"] = "yes"

    hyps = model.transcribe(**kwargs)

    out: list = []
    for h in hyps:
        y_seq = getattr(h, "y_sequence", None)
        y_len = len(y_seq) if y_seq is not None else 0
        avg_logp = float(getattr(h, "score", 0.0)) / max(y_len, 1)
        entry = {
            "text": getattr(h, "text", "") or "",
            "avg_logp": round(avg_logp, 4),
            "word_timestamps": _extract_word_ts(h) if compute_ts else [],
        }
        if language is not None:
            entry["language"] = language
        out.append(entry)
    return out


def _transcribe_macro_one_lang(
    model,
    audio_list: list,
    language,
    micro: int,
    compute_ts: bool,
    *,
    catch_empty_hyp_indexerror: bool = True,
) -> list:
    out: list = []
    target = max(1, int(micro))
    i = 0
    while i < len(audio_list):
        sub = audio_list[i:i + target]
        try:
            out.extend(_transcribe_subbatch(model, sub, language, len(sub), compute_ts))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if target == 1:
                raise
            target = max(1, target // 2)
            logger.warning(
                "OOM at micro=%d → retry at micro=%d (remaining %d, lang=%s)",
                target * 2, target, len(audio_list) - i, language,
            )
            continue
        except IndexError:
            if not catch_empty_hyp_indexerror:
                raise
            if len(sub) > 1:
                for one in sub:
                    try:
                        out.extend(_transcribe_subbatch(model, [one], language, 1, compute_ts))
                    except IndexError:
                        empty = dict(_EMPTY_HYP)
                        if language is not None:
                            empty["language"] = language
                        out.append(empty)
            else:
                empty = dict(_EMPTY_HYP)
                if language is not None:
                    empty["language"] = language
                out.append(empty)
            i += len(sub)
            continue
        i += len(sub)
    return out


def _transcribe_macro(
    model,
    audio_list: list,
    languages: list,
    micro: int,
    compute_ts: bool,
    *,
    catch_empty_hyp_indexerror: bool = True,
) -> list:
    if not audio_list:
        return []

    by_lang: dict = {}
    for i, lang in enumerate(languages):
        by_lang.setdefault(lang, []).append(i)

    results: list = [None] * len(audio_list)
    for lang, indices in by_lang.items():
        sub_audio = [audio_list[i] for i in indices]
        sub_results = _transcribe_macro_one_lang(
            model, sub_audio, lang, micro, compute_ts,
            catch_empty_hyp_indexerror=catch_empty_hyp_indexerror,
        )
        for src_i, res in zip(indices, sub_results):
            results[src_i] = res
    return results
