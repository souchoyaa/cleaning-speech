"""Per-model workers for the fused ASR pipeline.

- ``voxtral`` — sliding-window HTTP client to a remote vLLM Voxtral server.
- ``vad``     — NeMo Frame-VAD (MarbleNet v2.0) on the local GPU.
- ``parakeet`` / ``canary`` — local NeMo ASR forwards with OOM-aware
  micro-batching.
- ``aligner`` — optional NFA forced alignment that replaces one ASR's
  word_timestamps with NFA-aligned boundaries.
"""
