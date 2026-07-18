"""ASR backends (transcription, chat_completion).

A backend is anything that turns ``(flac_bytes, language_hint)`` into
``{text, avg_logp, language?, error?}`` via one HTTP call to a remote
vLLM server. The feeder loop owns retries / concurrency; backends are
stateless transformers.
"""

from .base import TranscriptionBackend, TransientError, build_backend

__all__ = ["TranscriptionBackend", "TransientError", "build_backend"]
