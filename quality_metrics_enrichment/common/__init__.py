"""Shared infrastructure for the ASR and MOS pipelines.

- ``loader``  — Lhotse-Shar audio loader with dynamic bucketing.
- ``logging`` — pipeline + library log routing (one file per side).
- ``timing``  — ``RunMetrics`` + ``StageTimer`` wall-clock instrumentation.
- ``runtime`` — environment helpers (PIPELINE_LOG_DIR / PIPELINE_RUN_NAME).
"""
