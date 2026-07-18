"""Transcription backend Protocol + factory.

A backend converts (audio_bytes, language_hint) into a transcription
dict via one HTTP call. Backends are *stateless transformers* — they
do not own retries, concurrency, or output writing. The feeder loop
handles those.

Exceptions:
  - ``TransientError``: caller should retry (5xx, network blip).
  - Any other exception: treated as permanent failure.
  - Return value with non-None ``error``: also treated as permanent
    failure (the writer commits an error row so resume doesn't retry).
"""

from __future__ import annotations

import importlib
from typing import Optional, Protocol

import httpx


class TransientError(RuntimeError):
    """5xx or network failure — caller retries with backoff."""


class TranscriptionBackend(Protocol):
    """Interface every vLLM-fronted backend implements."""

    slot_name: str
    """Subdir + JSONL slot name (e.g. ``voxtral``, ``qwen_audio``).

    Doubles as the field name in the merged JSONL's
    ``hypotheses:{...}`` block. Choose carefully — downstream ROVER
    keys on these names.
    """

    async def transcribe(
        self,
        client: httpx.AsyncClient,
        blob: bytes,
        language: Optional[str],
        *,
        cut_id: str,
        ext: str = "flac",
    ) -> dict:
        """Send one cut, return ``{text, avg_logp, language?, error?}``.

        Caller owns retries. Raise ``TransientError`` to request a retry;
        return ``{"error": ...}`` for a permanent failure (e.g. 4xx).
        """
        ...


def build_backend(cfg: dict) -> TranscriptionBackend:
    """Instantiate a backend from a ``vllm_feeder:`` config block.

    Selects on ``cfg['backend']`` (default ``transcription``). Each impl
    module exposes a ``build(cfg)`` factory returning a backend
    instance. Adding a new backend = add a new module + import here.
    """
    name = (cfg.get("backend") or "transcription").lower()
    try:
        mod = importlib.import_module(f"asr_vllm.backends.{name}")
    except ImportError as e:
        raise ValueError(
            f"Unknown vllm_feeder.backend={name!r}. "
            f"Available: transcription, chat_completion. Error: {e}",
        ) from e
    if not hasattr(mod, "build"):
        raise AttributeError(
            f"Backend module asr_vllm.backends.{name} missing build(cfg) factory.",
        )
    return mod.build(cfg)
