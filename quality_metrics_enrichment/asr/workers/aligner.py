"""NFA forced-alignment worker — NeMo ≥ 2.5 in-package API.

Aligns a transcript against decoded audio using NeMo's
``aligner_utils.viterbi_decoding``. We bypass ``get_batch_variables`` and
run preprocessor + encoder + CTC head ourselves so we can reuse the audio
tensors already in memory (no path-based reload). For hybrid models we
must call ``change_decoding_strategy(decoder_type='ctc')`` to wire up the
CTC head.
"""

from __future__ import annotations

import logging
import unicodedata

import numpy as np
import torch

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "nvidia/stt_fr_fastconformer_hybrid_large_pc"
DEFAULT_MICRO_BATCH = 16


def build_aligner(cfg: dict, device: torch.device):
    """Load the alignment model and switch hybrid → CTC mode."""
    import nemo.collections.asr as nemo_asr

    name = str(cfg.get("model", DEFAULT_MODEL))
    dtype_name = str(cfg.get("dtype", "bfloat16")).lower()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)

    logger.info("Loading aligner (%s) → %s", name, dtype)
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=name, map_location=device,
    )
    model.eval()
    model.freeze()

    try:
        model.change_decoding_strategy(decoder_type="ctc")
    except (TypeError, AttributeError, ValueError):
        # Pure-CTC model — no decoder swap needed.
        pass

    if dtype != torch.float32:
        try:
            model = model.to(dtype)
        except (RuntimeError, TypeError) as e:
            logger.warning("Aligner cast to %s failed (%s) — staying in fp32.", dtype, e)

    return model


@torch.inference_mode()
def _ctc_log_probs(model, audio_list: list) -> tuple:
    """Run preprocessor + encoder + CTC head on a batch.

    Hybrid models: ``model.forward()`` returns ``(spec, spec_len, ...)`` —
    the first element is the mel spectrogram, NOT log-probs. Calling Viterbi
    on the spec triggers a CUDA device-side assert.
    """
    sigs = [torch.from_numpy(a).float() for a in audio_list]
    lens = torch.tensor([s.shape[0] for s in sigs], dtype=torch.long, device=model.device)
    sig = torch.nn.utils.rnn.pad_sequence(sigs, batch_first=True).to(model.device)

    ctc_head = getattr(model, "ctc_decoder", None)
    if ctc_head is not None:
        proc_signal, proc_signal_len = model.preprocessor(input_signal=sig, length=lens)
        enc_out, enc_len = model.encoder(audio_signal=proc_signal, length=proc_signal_len)
        log_probs = ctc_head(encoder_output=enc_out)
        return log_probs, enc_len

    out = model.forward(input_signal=sig, input_signal_length=lens)
    if not (isinstance(out, (tuple, list)) and len(out) >= 2):
        raise RuntimeError(f"Unexpected pure-CTC forward output shape: {type(out)}")
    log_probs, encoded_len = out[0], out[1]
    if log_probs.ndim != 3:
        raise RuntimeError(
            f"Pure-CTC forward returned non-3D first element (shape={log_probs.shape})",
        )
    return log_probs, encoded_len


def align_batch(
    model,
    audio_list: list,
    text_list: list,
    output_timestep_duration: float,
) -> list:
    """Align ``audio_list`` against ``text_list``; return per-cut word ts lists."""
    from nemo.collections.asr.parts.utils.aligner_utils import (
        get_utt_obj,
        add_t_start_end_to_utt_obj,
        viterbi_decoding,
    )

    log_probs, encoded_lens = _ctc_log_probs(model, audio_list)
    log_probs = log_probs.float()
    B, T_max, V = log_probs.shape

    utts: list = []
    for i, txt in enumerate(text_list):
        text_norm = unicodedata.normalize("NFC", txt or "").strip()
        if not text_norm:
            utts.append(None)
            continue
        try:
            u = get_utt_obj(
                text_norm, model,
                segment_separators=[".", "?", "!", "..."],
                word_separator=" ",
                T=int(encoded_lens[i].item()),
                audio_filepath=f"cut_{i}",
                utt_id=f"utt_{i}",
            )
        except (RuntimeError, ValueError, TypeError):
            u = None
        # viterbi backtrack does v_prev[..., U-2:U]; trivial transcripts
        # (only the leading blank) make that slice empty → IndexError.
        if u is not None and len(u.token_ids_with_blanks) < 2:
            u = None
        utts.append(u)

    valid_idx = [i for i, u in enumerate(utts) if u is not None]
    if not valid_idx:
        return [[] for _ in text_list]

    valid_idx_t = torch.tensor(valid_idx, dtype=torch.long, device=log_probs.device)
    valid_log_probs = log_probs.index_select(0, valid_idx_t)
    valid_T = encoded_lens.index_select(0, valid_idx_t)
    valid_utts = [utts[i] for i in valid_idx]
    valid_U = [len(u.token_ids_with_blanks) for u in valid_utts]
    valid_U_max = max(valid_U)

    valid_y = (V) * torch.ones((len(valid_utts), valid_U_max), dtype=torch.int64)
    for b, u in enumerate(valid_utts):
        ids = u.token_ids_with_blanks
        valid_y[b, :len(ids)] = torch.tensor(ids, dtype=torch.int64)

    try:
        aligns = viterbi_decoding(
            valid_log_probs,
            valid_y,
            torch.tensor([int(x) for x in valid_T.cpu().tolist()], dtype=torch.long),
            torch.tensor(valid_U, dtype=torch.long),
            viterbi_device=log_probs.device,
        )
    except (IndexError, RuntimeError) as e:
        logger.warning(
            "viterbi_decoding raised %s on a valid-utt batch of %d.",
            type(e).__name__, len(valid_utts),
        )
        return [[] for _ in text_list]

    out: list = [[] for _ in text_list]
    for new_b, orig_i in enumerate(valid_idx):
        utt = valid_utts[new_b]
        try:
            add_t_start_end_to_utt_obj(utt, aligns[new_b], output_timestep_duration)
        except (KeyError, IndexError, RuntimeError):
            continue
        words: list = []
        for seg in utt.segments_and_tokens:
            wat = getattr(seg, "words_and_tokens", None)
            if wat is None:
                continue
            for w in wat:
                # Duck-type Word vs Token: Word has a `tokens` list, Token doesn't.
                if not hasattr(w, "tokens"):
                    continue
                t_start = getattr(w, "t_start", None)
                t_end = getattr(w, "t_end", None)
                text = getattr(w, "text", None) or getattr(w, "word", None)
                if t_start is None or t_end is None or not text:
                    continue
                if t_start < 0 or t_end < 0:
                    # NeMo emits -1 for blanks / unreached tokens.
                    continue
                words.append({
                    "w": str(text).strip(),
                    "s": round(float(t_start), 3),
                    "e": round(float(t_end), 3),
                })
        out[orig_i] = words
    return out


class AlignmentWorker:
    """Owns one loaded aligner model + micro-batch / output-timestep knobs."""

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        *,
        smoke_test: bool = True,
    ) -> None:
        self._device = device
        self._micro = int(cfg.get("nemo_micro_batch_size", DEFAULT_MICRO_BATCH))
        self._model = build_aligner(cfg, device)

        # Encoder frame stride = preprocessor.window_stride * subsampling_factor
        # (FastConformer: 0.01 s × 8 = 0.08 s).
        self._sec_per_frame = float(self._model.cfg.preprocessor.window_stride) * int(
            getattr(self._model.encoder, "subsampling_factor", 8),
        )
        logger.info(
            "AlignmentWorker ready | micro=%d | sec_per_frame=%.4f",
            self._micro, self._sec_per_frame,
        )

        if smoke_test:
            try:
                _ = align_batch(
                    self._model,
                    [np.zeros(16000, dtype=np.float32)],
                    ["bonjour"],
                    self._sec_per_frame,
                )
            except Exception:
                logger.exception("Aligner smoke test failed — aborting.")
                raise

    def align(self, audio_list: list, text_list: list) -> list:
        """Align (audio, text) pairs in micro-sized chunks."""
        n = len(audio_list)
        if n == 0:
            return []
        if len(text_list) != n:
            raise ValueError(
                f"audio_list ({n}) and text_list ({len(text_list)}) length mismatch",
            )

        out: list = [[] for _ in range(n)]
        i = 0
        target = max(1, self._micro)
        while i < n:
            chunk_end = min(i + target, n)
            chunk_audio = audio_list[i:chunk_end]
            chunk_text = text_list[i:chunk_end]
            try:
                aligned = align_batch(
                    self._model, chunk_audio, chunk_text, self._sec_per_frame,
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if target == 1:
                    logger.warning("Aligner OOM at micro=1 — emitting empty for idx %d.", i)
                    i += 1
                    continue
                target = max(1, target // 2)
                logger.warning(
                    "Aligner OOM at micro=%d → retry at micro=%d (remaining %d)",
                    target * 2, target, n - i,
                )
                continue
            except Exception:
                logger.exception("Aligner crashed on chunk [%d:%d).", i, chunk_end)
                i = chunk_end
                continue
            for j, words in enumerate(aligned):
                out[i + j] = words
            i = chunk_end
        return out

    @property
    def micro_batch(self) -> int:
        return self._micro
