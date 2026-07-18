"""Chat-completion backend: ``/v1/chat/completions`` with audio modality.

Selected via ``backend: "chat_completion"``.
Supports any vLLM-served model that accepts audio in the OpenAI chat
messages format (Qwen2-Audio-7B-Instruct, etc.). The audio is sent as
a base64 data URI inside a ``content`` list — vLLM decodes it.

Not for text-only models (Qwen3-1.7B etc.) — those cannot transcribe
audio directly. Use Voxtral / Qwen2-Audio / any audio-capable model.

The prompt template controls what we ask for. Default asks for a
verbatim transcript. Override per-language via the ``prompt_template``
config key (uses str.format with ``{language}``).

Logprobs: set ``request_logprobs: true`` to ask the server for per-token
logprobs; they're averaged over the transcript region into ``avg_logp``
(ROVER confidence weighting). This is the only Qwen3-ASR path that
returns logprobs — its /v1/audio/transcriptions endpoint does not.

Output markers: Qwen3-ASR wraps its transcript as
``language <Lang><asr_text><TRANSCRIPT>``. Set
``strip_to_marker: "<asr_text>"`` (and optionally
``strip_after_marker: "</asr_text>"``) to recover the clean transcript
and restrict the avg_logp average to the transcript tokens.
"""

from __future__ import annotations

import base64
import json
from typing import Optional

import httpx

from .base import TransientError


_DEFAULT_PROMPT = (
    "Transcribe the following audio exactly. Output only the transcript text, "
    "with no additional commentary."
)
_DEFAULT_MAX_TOKENS = 1024


class ChatCompletionBackend:
    """Generic audio-in-chat-completions backend.

    Slot name defaults to ``vllm_chat`` but is overridable via
    ``cfg['slot_name']`` so multiple chat backends (Qwen, etc.) can
    coexist as distinct hypothesis slots in the joined output.
    """

    def __init__(self, cfg: dict) -> None:
        base = cfg.get("api_base")
        if not base:
            raise ValueError(
                "vllm_feeder.api_base is required (or set VLLM_API_BASE).",
            )
        self.bases = [b.strip().rstrip("/") for b in str(base).split(",") if b.strip()]
        self.model = cfg.get("api_model")
        if not self.model:
            raise ValueError(
                "vllm_feeder.api_model is required for chat_completion backend "
                "(must match --served-model-name on the vLLM server).",
            )
        self.prompt_template = cfg.get("prompt_template") or _DEFAULT_PROMPT
        self.max_tokens = int(cfg.get("max_tokens", _DEFAULT_MAX_TOKENS))
        self.temperature = float(cfg.get("api_temperature", 0.0))
        self.slot_name = str(cfg.get("slot_name", "vllm_chat"))
        # Logprobs + Qwen3-ASR output-marker handling (all opt-in).
        self.request_logprobs = bool(cfg.get("request_logprobs", False))
        self.strip_to_marker = cfg.get("strip_to_marker") or None
        self.strip_after_marker = cfg.get("strip_after_marker") or None
        api_key = str(cfg.get("api_key", "EMPTY"))
        self.headers = (
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            if api_key
            else {"Content-Type": "application/json"}
        )

    def pick_base(self, cut_id: str) -> str:
        return self.bases[hash(cut_id) % len(self.bases)]

    def _clean_text(self, content: str) -> str:
        """Recover the transcript from a model's raw chat output.

        Qwen3-ASR emits ``language <Lang><asr_text><TRANSCRIPT>`` (and may
        close with ``</asr_text>``). With ``strip_to_marker`` /
        ``strip_after_marker`` configured we slice out the transcript;
        otherwise the content is returned verbatim.
        """
        text = content
        if self.strip_to_marker and self.strip_to_marker in text:
            text = text.split(self.strip_to_marker, 1)[1]
        if self.strip_after_marker and self.strip_after_marker in text:
            text = text.split(self.strip_after_marker, 1)[0]
        return text.strip()

    def _avg_logp(self, payload: dict) -> float:
        """Mean per-token logprob over the transcript region.

        Restricts the average to tokens after ``strip_to_marker`` and
        before ``strip_after_marker`` so the near-deterministic wrapper
        tokens (``language``, ``<asr_text>``, …) don't dominate. Returns
        0.0 when logprobs weren't requested / returned.
        """
        try:
            toks = payload["choices"][0]["logprobs"]["content"]
        except (KeyError, TypeError, IndexError):
            return 0.0
        if not toks:
            return 0.0
        start, end = 0, len(toks)
        if self.strip_to_marker:
            for i, t in enumerate(toks):
                if self.strip_to_marker in (t.get("token") or ""):
                    start = i + 1
                    break
        if self.strip_after_marker:
            for i in range(start, len(toks)):
                if self.strip_after_marker in (toks[i].get("token") or ""):
                    end = i
                    break
        region = toks[start:end] or toks
        vals = [float(t["logprob"]) for t in region if t.get("logprob") is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _build_prompt(self, language: Optional[str]) -> str:
        # str.format tolerates missing keys only if they're in the template;
        # if the user provided a template without {language}, plain string
        # is returned unchanged.
        try:
            return self.prompt_template.format(language=language or "auto")
        except (KeyError, IndexError):
            return self.prompt_template

    async def transcribe(
        self,
        client: httpx.AsyncClient,
        blob: bytes,
        language: Optional[str],
        *,
        cut_id: str,
        ext: str = "flac",
    ) -> dict:
        if not blob:
            return {"text": "", "avg_logp": 0.0, "error": "no_audio_blob"}

        url = self.pick_base(cut_id) + "/v1/chat/completions"
        audio_b64 = base64.b64encode(blob).decode("ascii")
        mime = _mime_for(ext)
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
                },
                {"type": "text", "text": self._build_prompt(language)},
            ],
        }]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.request_logprobs:
            body["logprobs"] = True

        try:
            resp = await client.post(url, headers=self.headers, content=json.dumps(body))
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise TransientError(f"network: {type(e).__name__}: {e}") from e

        if resp.status_code == 200:
            try:
                payload = resp.json()
                content = payload["choices"][0]["message"]["content"] or ""
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                return {"text": "", "avg_logp": 0.0, "error": f"bad_response: {e}"}
            text = self._clean_text(content)
            entry: dict = {"text": text, "avg_logp": self._avg_logp(payload)}
            if language:
                entry["language"] = language
            return entry
        if resp.status_code >= 500:
            raise TransientError(f"http_{resp.status_code}: {resp.text[:200]}")
        return {
            "text": "", "avg_logp": 0.0,
            "error": f"http_{resp.status_code}: {resp.text[:400]}",
        }


def _mime_for(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    return {
        "flac": "audio/flac",
        "wav":  "audio/wav",
        "ogg":  "audio/ogg",
        "opus": "audio/opus",
        "mp3":  "audio/mpeg",
    }.get(ext, "application/octet-stream")


def build(cfg: dict) -> ChatCompletionBackend:
    return ChatCompletionBackend(cfg)
