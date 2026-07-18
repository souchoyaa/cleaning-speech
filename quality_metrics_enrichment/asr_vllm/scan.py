"""Standalone shar duration scanner — pre-compute the total work.

Invoked from launch.sh in the background while we wait for vLLM
``/health`` to return 200. Reads only the gzipped ``cuts.*.jsonl.gz``
manifests (no tar audio), in parallel across N threads, and prints
``TOTAL_SECONDS=<n>`` on stdout so the wrapping shell can capture it
and export it to the feeder via ``ASR_VLLM_TOTAL_SECONDS``.

Applies the same filters the feeder will:
  - excludes cuts already listed in the sidecar (resume-aware)
  - excludes cuts outside ``loader.min_duration`` / ``loader.max_duration``

so the printed total is exactly the audio this run will process.

Usage::

    python -m asr_vllm.scan --config asr_vllm/egs/cv_fr_qwen3_asr.yaml
    python -m asr_vllm.scan --config X.yaml --num-workers 32 --quiet

Exit code is 0 even if the scan partially failed — the printed total
is whatever we managed to sum, and the feeder degrades gracefully
(no ETA when total is 0).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from .backends import build_backend
from .shar_reader import scan_total_duration


logger = logging.getLogger("asr_vllm.scan")


def _expand_user_in(d: dict) -> dict:
    user = os.environ.get("USER", "")
    return {k: (v.replace("${USER}", user) if isinstance(v, str) else v) for k, v in d.items()}


def _sidecar_seen(jsonl_path: Path, sidecar_path: Path) -> set[str]:
    """Same logic as main._load_seen_cut_ids — duplicated here so scan
    doesn't import main (which pulls in httpx + async setup)."""
    seen: set[str] = set()
    if sidecar_path.exists():
        for line in sidecar_path.open():
            cid = line.strip()
            if cid:
                seen.add(cid)
    # Fall back to JSONL scan only if it might add anything (otherwise
    # we double-cost the slow part).
    if not jsonl_path.exists():
        return seen
    import json as _json
    with jsonl_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                row = _json.loads(raw)
            except _json.JSONDecodeError:
                break  # partial write — stop scanning
            cid = row.get("cut_id")
            if cid and cid not in seen:
                seen.add(cid)
    return seen


def main() -> int:
    p = argparse.ArgumentParser(description="Pre-compute total audio duration for a shar.")
    p.add_argument("--config", required=True)
    p.add_argument("--num-workers", type=int, default=16,
                   help="Parallel shard readers (default 16).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress logging (just the TOTAL_SECONDS= line).")
    args = p.parse_args()

    logging.basicConfig(
        level=("WARNING" if args.quiet else "INFO"),
        format="[scan] %(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("shar_dir") or not cfg.get("output_dir"):
        logger.error("config.shar_dir and config.output_dir are required.")
        # Emit a 0 so the caller can detect "scan failed" without parsing stderr.
        print("TOTAL_SECONDS=0")
        return 0

    # Build backend just to learn the slot name (where the sidecar lives).
    # Stub api_base because the backend constructor requires it.
    feeder_cfg = dict(cfg.get("vllm_feeder") or {})
    feeder_cfg = _expand_user_in(feeder_cfg)
    feeder_cfg.setdefault("api_base", "http://localhost:0")
    try:
        backend = build_backend(feeder_cfg)
        slot = backend.slot_name
    except Exception as e:
        logger.warning("Could not build backend (%s); defaulting slot='vllm'.", e)
        slot = "vllm"

    output_dir = Path(cfg["output_dir"])
    out_subdir = output_dir / "quality_asr" / slot
    jsonl_path   = out_subdir / f"{slot}_rank_0000.jsonl"
    sidecar_path = out_subdir / f"{slot}_rank_0000.jsonl.cutids"
    seen = _sidecar_seen(jsonl_path, sidecar_path)
    logger.info("Resume-aware: %d cut_ids already in sidecar.", len(seen))

    loader_cfg = cfg.get("loader") or {}
    min_dur = loader_cfg.get("min_duration")
    max_dur = loader_cfg.get("max_duration")

    t0 = time.monotonic()
    def _progress(done: int, total: int, partial_s: float) -> None:
        logger.info(
            "shards %d/%d  partial=%.1fh  (%.0fs elapsed)",
            done, total, partial_s / 3600.0, time.monotonic() - t0,
        )

    total_s = scan_total_duration(
        cfg["shar_dir"],
        skip_cut_ids=seen,
        min_duration=min_dur,
        max_duration=max_dur,
        num_workers=args.num_workers,
        on_progress=_progress,
    )
    elapsed = time.monotonic() - t0
    logger.info(
        "scan done: %.1fh of unseen audio (%.0fs) in %.1fs with %d workers.",
        total_s / 3600.0, total_s, elapsed, args.num_workers,
    )
    # Machine-readable result for launch.sh. Keep it on stdout, single line.
    print(f"TOTAL_SECONDS={total_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
