"""Inverse text normalization — LLM-based via vLLM chat completions.

Per-language prompts (30 languages, aligned with Qwen3-ASR) live in
``itn_prompts.py``; each carries the locale's separators / currency / date
formats and a few native examples (incl. a negative case so idioms aren't
normalized). An async httpx client posts to ``<api_base>/v1/chat/completions``;
unknown languages fall back to ``"en"``.

Config flags: ``native`` toggles English- vs native-instruction prompts;
``resolve_conflicts`` (default True) keeps the ROVER-tie block so the LLM
resolves split votes (drop it when transcripts are already trusted).

Best-effort: a per-row failure or a wildly different output length (<25% / >300%)
passes the text through verbatim — ITN never breaks the join. The caller drives
concurrency; this module exposes ``normalize``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from .itn_prompts import PROMPTS_EN, PROMPTS_NATIVE, build_prompt

logger = logging.getLogger(__name__)


def _format_ambiguity_section(ambiguous_words: list) -> str:
    """Build the inline instructions block for word-level disambiguation.

    Returns ``""`` when there are no ambiguous words. Otherwise returns a
    block with a leading + trailing newline so it inserts cleanly between
    the examples and the final ``Input:``/``Output:`` lines in any of the
    per-language prompt templates.
    """
    if not ambiguous_words:
        return ""
    lines = []
    for w in ambiguous_words:
        cands = ", ".join(f'"{c}"' for c in w.get("candidates", []))
        # "default" is the deterministic weighted-ROVER winner; fall back to the
        # primary word for older ambiguous-word records that predate it.
        default = w.get("default", w.get("primary"))
        lines.append(
            f'  - position {w.get("position")}: pick from [{cands}] the word '
            f'that best fits the sentence; if not confident, keep "{default}"'
        )
    return (
        "\nAdditionally, the upstream ASR vote was split at some word positions "
        "in the input. For each listed position, REPLACE the word with whichever "
        "candidate best fits the spoken context (what was actually said aloud). "
        "IMPORTANT: if you are not confident, KEEP the given default word exactly "
        "— do not guess; this keeps the result deterministic. Positions are "
        "0-indexed in the input's whitespace-split word list:\n"
        + "\n".join(lines) + "\n"
    )

# Reject obviously broken LLM outputs and fall back to verbatim. Bounds are
# wide because legitimate ITN can shrink ("two thousand twenty four" → "2024")
# or grow ("3" → "three") substantially.
_MIN_LEN_RATIO = 0.25
_MAX_LEN_RATIO = 3.0


class ITNTransportError(RuntimeError):
    """Raised on network / HTTP / JSON parse failures — caller can count
    these for circuit-breaker logic. Length-sanity rejection stays a
    silent passthrough (it's the LLM editorializing, not a server fault)."""


class ITNClient:
    """Async LLM-based inverse text normalizer.

    Construction validates only that ``api_base`` and ``api_model`` are
    set — model load happens server-side. Env vars take priority over
    YAML values (same precedence as asr_vllm):
      - ``ITN_API_BASE``  → ``api_base``  (falls back to ``VLLM_API_BASE``)
      - ``ITN_MODEL``     → ``api_model`` (falls back to ``VLLM_MODEL``)
    """

    def __init__(self, cfg: dict) -> None:
        api_base = (
            os.environ.get("ITN_API_BASE")
            or os.environ.get("VLLM_API_BASE")
            or cfg.get("api_base")
        )
        if not api_base:
            raise ValueError(
                "ITN: api_base required (config.api_base, ITN_API_BASE, or VLLM_API_BASE).",
            )
        api_model = (
            os.environ.get("ITN_MODEL")
            or os.environ.get("VLLM_MODEL")
            or cfg.get("api_model")
        )
        if not api_model:
            raise ValueError(
                "ITN: api_model required (config.api_model, ITN_MODEL, or VLLM_MODEL).",
            )

        self.api_base = api_base.rstrip("/")
        self.api_model = api_model
        self.api_key = str(cfg.get("api_key", "EMPTY"))
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.0))
        # native=True picks instructions in the target language; native=False
        # uses the English-instruction variant (examples are always in the
        # target language). Default native because the locale-specific
        # conventions read more naturally in their own language.
        self.native = bool(cfg.get("native", True))
        # resolve_conflicts=True keeps the per-word ROVER-tie block in the
        # prompt whenever the row carries ambiguous words (the heavier task:
        # the LLM picks the best candidate per tied position). Set False for
        # datasets whose transcripts are already trusted — ITN then does only
        # number/date/currency normalization, ignoring any ambiguous words.
        self.resolve_conflicts = bool(cfg.get("resolve_conflicts", True))
        # Sanity-check bounds are exposed for users who hit ITNs that are
        # legitimately very long/short (e.g. number-heavy tables).
        self.min_len_ratio = float(cfg.get("min_len_ratio", _MIN_LEN_RATIO))
        self.max_len_ratio = float(cfg.get("max_len_ratio", _MAX_LEN_RATIO))

    def _build_prompt(
        self,
        text: str,
        lang: Optional[str],
        ambiguous_words: Optional[list] = None,
    ) -> str:
        # Drop the disambiguation block when conflict resolution is off → the
        # plain per-language ITN prompt (lighter task), regardless of any
        # ambiguous words the caller passed.
        words = (ambiguous_words or []) if self.resolve_conflicts else []
        ambig_section = _format_ambiguity_section(words)
        lang_key = (lang or "en").lower()
        table = PROMPTS_NATIVE if self.native else PROMPTS_EN
        if lang_key not in table:
            lang_key = "en"
        return build_prompt(lang_key, text, ambig_section, native=self.native)

    async def normalize(
        self,
        client: httpx.AsyncClient,
        text: str,
        lang: Optional[str],
        *,
        ambiguous_words: Optional[list] = None,
    ) -> str:
        """Return ITN'd text. Raises ``ITNTransportError`` on
        network / HTTP / JSON-parse failures so the caller can count them
        for circuit-breaker logic. Length-sanity rejections are returned
        as-is (verbatim passthrough — not a server fault, just the LLM
        going off-rails on one row)."""
        if not text or not text.strip():
            return text

        body = {
            "model":       self.api_model,
            "messages":    [{
                "role": "user",
                "content": self._build_prompt(text, lang, ambiguous_words),
            }],
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = await client.post(
                f"{self.api_base}/v1/chat/completions",
                json=body, headers=headers,
            )
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise ITNTransportError(
                f"network: {type(e).__name__}: {e}",
            ) from e

        if resp.status_code != 200:
            raise ITNTransportError(
                f"http_{resp.status_code}: {resp.text[:200]}",
            )

        try:
            payload = resp.json()
            out = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as e:
            raise ITNTransportError(f"bad_response: {e}") from e

        out = out.strip()
        # Strip surrounding quotes if the LLM wrapped its output.
        if len(out) >= 2 and out[0] in '"\'`' and out[-1] == out[0]:
            out = out[1:-1].strip()

        if not out:
            return text

        # Sanity bound: drastic length change suggests the LLM editorialized
        # or refused. Better to keep the verbatim consensus than emit garbage.
        # NOT a transport error — single-row issue, don't count it for the
        # circuit breaker.
        ratio = len(out) / max(len(text), 1)
        if ratio < self.min_len_ratio or ratio > self.max_len_ratio:
            logger.warning(
                "ITN: output length ratio %.2f outside [%.2f, %.2f] for "
                "lang=%s — passthrough. Input len=%d, output len=%d.",
                ratio, self.min_len_ratio, self.max_len_ratio,
                lang, len(text), len(out),
            )
            return text

        return out
