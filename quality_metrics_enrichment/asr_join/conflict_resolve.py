"""Conflict resolution — LLM adjudication of ASR-vs-reference disagreements.

NEW Phase-2 flow (Task #8). Does NOT touch itn.py / the blanket-ITN path.

For each cut: parakeet transcript (the 1-slot rover.text) is compared to the
dataset's reference transcript. We compute WER/CER (kept as per-cut scores);
when they disagree (WER > gate) we ask the gemma server to RESOLVE the
conflict — produce the single most accurate, written-form transcript using
both sides — stored as ``rover.text_resolved``.

Per-row failure (network / HTTP 5xx) raises ``ResolveTransportError`` so the
caller can count it for a circuit breaker; everything else (4xx, bad JSON,
empty, length-sanity) falls back to the REFERENCE (the authoritative base) so
the run never breaks.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# WER / CER  (mirrors dup_retrieval/core/apply_to_shar._wer_cer so the score
# is computed identically; duplicated here to keep this module standalone).
# --------------------------------------------------------------------------
_ERR_PUNC = re.compile(r"[^a-z0-9'\s]", re.UNICODE)


def _err_norm(s):
    return _ERR_PUNC.sub(" ", (s or "").lower()).split()


try:
    from rapidfuzz.distance import Levenshtein as _Lev
    def _edit(a, b):
        return _Lev.distance(a, b)
except Exception:                       # pragma: no cover - rapidfuzz is in the image
    def _edit(a, b):
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[m]


def wer_cer(ref, hyp) -> Tuple[Optional[float], Optional[float]]:
    """(WER, CER) of hyp vs ref over normalized text; (None, None) if ref empty."""
    rw, hw = _err_norm(ref), _err_norm(hyp)
    if not rw:
        return None, None
    wer = _edit(rw, hw) / len(rw)
    rc, hc = " ".join(rw), " ".join(hw)
    cer = (_edit(rc, hc) / len(rc)) if rc else None
    return wer, cer


# --------------------------------------------------------------------------
class ResolveTransportError(RuntimeError):
    """5xx / network failure — caller counts it for the circuit breaker."""


_LANG_NAME = {"en": "English", "de": "German", "fr": "French",
              "es": "Spanish", "it": "Italian"}
_MIN_RATIO, _MAX_RATIO = 0.25, 3.0

_PROMPT = (
    "Two transcripts of the same {lang} speech segment disagree.\n"
    "REFERENCE (the official transcript, usually correct):\n{ref}\n\n"
    "ASR (automatic speech recognizer, may reveal errors in the reference):\n{asr}\n\n"
    "Output the single most accurate verbatim transcript of what was actually said. "
    "Use the REFERENCE as the base and change it ONLY where the ASR makes a clear, "
    "plausible correction (a misspelled or wrong word, a missed word); keep the "
    "reference's wording wherever you are unsure. Write numbers, dates, times and "
    "currency in their standard written form for {lang}. "
    "Output ONLY the final transcript text — no quotes, labels, or commentary."
)


class ConflictResolveClient:
    """Gemma chat-completions client that resolves one ASR-vs-reference conflict."""

    def __init__(self, cfg: dict) -> None:
        api_base = (
            os.environ.get("ITN_API_BASE")
            or os.environ.get("VLLM_API_BASE")
            or cfg.get("api_base")
        )
        if not api_base:
            raise ValueError("resolve: api_base required (ITN_API_BASE / VLLM_API_BASE / config).")
        api_model = (
            os.environ.get("ITN_MODEL")
            or os.environ.get("VLLM_MODEL")
            or cfg.get("api_model")
        )
        if not api_model:
            raise ValueError("resolve: api_model required (ITN_MODEL / VLLM_MODEL / config).")
        self.api_base = api_base.rstrip("/")
        self.api_model = api_model
        self.api_key = str(cfg.get("api_key", "EMPTY"))
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.min_len_ratio = float(cfg.get("min_len_ratio", _MIN_RATIO))
        self.max_len_ratio = float(cfg.get("max_len_ratio", _MAX_RATIO))

    def _prompt(self, asr: str, ref: str, lang: Optional[str]) -> str:
        name = _LANG_NAME.get((lang or "en").lower(), lang or "the")
        return _PROMPT.format(lang=name, ref=ref, asr=asr)

    async def resolve(self, client: httpx.AsyncClient, asr_text: str,
                      ref_text: str, lang: Optional[str]) -> str:
        """Resolve one conflict → corrected transcript. Fallback = reference."""
        if not (ref_text and ref_text.strip()):
            return asr_text                      # no reference to anchor on → keep ASR
        body = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": self._prompt(asr_text, ref_text, lang)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = await client.post(f"{self.api_base}/v1/chat/completions",
                                     headers=headers, json=body)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise ResolveTransportError(f"network: {type(e).__name__}: {e}") from e
        if resp.status_code != 200:
            if resp.status_code >= 500:
                raise ResolveTransportError(f"http_{resp.status_code}")
            return ref_text                      # 4xx → passthrough to reference
        try:
            out = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ref_text
        if not out:
            return ref_text
        ratio = len(out) / max(len(ref_text), 1)
        if ratio < self.min_len_ratio or ratio > self.max_len_ratio:
            logger.warning("resolve: length ratio %.2f out of bounds lang=%s — passthrough to ref.",
                           ratio, lang)
            return ref_text
        return out


def build(cfg: dict) -> ConflictResolveClient:
    return ConflictResolveClient(cfg)
