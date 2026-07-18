"""Transcription backend: OpenAI ``/v1/audio/transcriptions`` (multipart).

Selected via ``backend: "transcription"``. Stateless (no shared session,
no thread pool, no joiner coupling). Works for any model vLLM exposes on
the transcription endpoint — Voxtral-Small-24B-2507, Qwen3-ASR, etc. The
bytes are sent verbatim; vLLM decodes FLAC server-side.

Set ``api_response_format: verbose_json`` to get per-segment logprobs
(aggregated into ``avg_logp``); plain ``json`` carries none.
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from .base import TransientError


_DEFAULT_API_MODEL = "mistralai/Voxtral-Small-24B-2507"
_DEFAULT_RESPONSE_FORMAT = "json"
_DEFAULT_TEMPERATURE = 0.0


class AudioTranscriptionBackend:
    def __init__(self, cfg: dict) -> None:
        # Overridable so the same backend can serve non-Voxtral models
        # (e.g. Qwen3-ASR via /v1/audio/transcriptions) under a distinct
        # slot — otherwise outputs collide under the hardcoded "voxtral"
        # subdir and ROVER merges the wrong hypotheses.
        self.slot_name = str(cfg.get("slot_name", "voxtral"))
        base = cfg.get("api_base")
        if not base:
            raise ValueError(
                "vllm_feeder.api_base is required (or set VLLM_API_BASE).",
            )
        # Multi-replica fronting: comma-separated list of URLs.
        # The async client uses one transport; round-robin across bases
        # is handled by the feeder, not the backend.
        self.bases = [b.strip().rstrip("/") for b in str(base).split(",") if b.strip()]
        self.model = str(cfg.get("api_model", _DEFAULT_API_MODEL))
        self.response_format = str(cfg.get("api_response_format", _DEFAULT_RESPONSE_FORMAT))
        self.temperature = float(cfg.get("api_temperature", _DEFAULT_TEMPERATURE))
        api_key = str(cfg.get("api_key", "EMPTY"))
        # vLLM launched with --api-key=EMPTY rejects requests without the
        # bearer header — always send it when the key is set.
        self.headers = (
            {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )

    def pick_base(self, cut_id: str) -> str:
        """Stable round-robin assignment of a cut to one of the replicas.

        Hashes cut_id so reruns hit the same replica (gives vLLM's
        prefix cache a slight chance to help).
        """
        return self.bases[hash(cut_id) % len(self.bases)]

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

        url = self.pick_base(cut_id) + "/v1/audio/transcriptions"
        files = {"file": (f"clip.{ext}", blob, _mime_for(ext))}
        data: dict = {
            "model": self.model,
            "response_format": self.response_format,
            "temperature": str(self.temperature),
        }
        if language:
            data["language"] = language

        try:
            resp = await client.post(url, headers=self.headers, files=files, data=data)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise TransientError(f"network: {type(e).__name__}: {e}") from e

        if resp.status_code == 200:
            return _parse_response(resp, language)
        if resp.status_code >= 500:
            raise TransientError(f"http_{resp.status_code}: {resp.text[:200]}")
        # 4xx — permanent (auth, payload). Commit so reruns don't retry forever.
        # Include the server's response body in the error so a probe / log
        # diff actually shows WHY (model name mismatch, unsupported field,
        # auth, etc.) instead of just "http_400".
        return {
            "text": "", "avg_logp": 0.0,
            "error": f"http_{resp.status_code}: {resp.text[:400]}",
        }


def _parse_response(resp: httpx.Response, lang_in: Optional[str]) -> dict:
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        return {"text": "", "avg_logp": 0.0, "error": "bad_json"}
    text = (payload.get("text") or "").strip()
    entry: dict = {"text": text, "avg_logp": _extract_avg_logp(payload)}
    detected = payload.get("language")
    if detected:
        entry["language"] = detected
    elif lang_in:
        entry["language"] = lang_in
    return entry


def _extract_avg_logp(payload: dict) -> float:
    """Aggregate transcription logprobs into a single per-utterance mean.

    Populated when the request asks for ``response_format=verbose_json``
    (or ``logprobs``); plain ``json`` carries none, so this returns 0.0
    and the previous behaviour is preserved. Handles the two shapes vLLM's
    ``/v1/audio/transcriptions`` returns:

      - verbose_json (Whisper-style): ``segments: [{avg_logprob, start,
        end, ...}]`` → duration-weighted mean of ``avg_logprob`` (longer
        segments dominate the utterance-level confidence).
      - logprobs include: ``logprobs: [{token, logprob}, ...]`` → simple
        mean of the per-token ``logprob``.

    The result lands in ``avg_logp``, which ROVER uses for
    confidence-weighted voting (asr_join/rover_offline.py).
    """
    segments = payload.get("segments")
    if isinstance(segments, list) and segments:
        num = den = 0.0
        any_lp = False
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            lp = seg.get("avg_logprob")
            if lp is None:
                continue
            any_lp = True
            w = max(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 0.0) or 1.0
            num += float(lp) * w
            den += w
        if any_lp and den > 0:
            return num / den
    logprobs = payload.get("logprobs")
    if isinstance(logprobs, list) and logprobs:
        vals = [
            float(t["logprob"])
            for t in logprobs
            if isinstance(t, dict) and t.get("logprob") is not None
        ]
        if vals:
            return sum(vals) / len(vals)
    return 0.0


def _mime_for(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    return {
        "flac": "audio/flac",
        "wav":  "audio/wav",
        "ogg":  "audio/ogg",
        "opus": "audio/opus",
        "mp3":  "audio/mpeg",
    }.get(ext, "application/octet-stream")


def build(cfg: dict) -> AudioTranscriptionBackend:
    return AudioTranscriptionBackend(cfg)
