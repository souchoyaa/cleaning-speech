"""vLLM transcription feeder.

Reads a Lhotse Shar directly (no bucketing, no decode) and streams FLAC
blobs to a remote vLLM server via the configured backend (Voxtral audio
API, chat-completions with audio modality, etc.).

Designed to run on the interactive node alongside the orchestrator —
the SLURM allocation is only for the vLLM server itself.
"""
