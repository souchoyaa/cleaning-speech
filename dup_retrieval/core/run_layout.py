"""Canonical run layout — the single source of truth for output names.

Every stage writes (and every downstream reader reads) through these constants
and helpers, so the on-disk hierarchy is uniform and a rename happens in ONE
place.  Layout under a run's ``output_dir``::

    <output_dir>/
      run_manifest.json                 # index of stages (status, rows, schema, paths)
      manifest/        part-00000.parquet ...      _SUCCESS  stage.json
      text_dedup/      clusters.parquet  candidate_edges.parquet   _SUCCESS  stage.json
      fingerprint/     part-00000.parquet ...                      _SUCCESS  stage.json
      audio_match/     clusters.parquet  match_edges.parquet  huge_clusters.parquet  _SUCCESS
      quality/         merged.parquet                              _SUCCESS  stage.json
      retention/       assignments.parquet                         _SUCCESS  stage.json
      output_shar/     <cloned shar>                               _SUCCESS

Conventions:
  * stage dir names: see ``STAGE`` (clear, lower_snake, no abbreviations).
  * sharded outputs: ``part-{NNNNN}.{ext}`` via ``shard_name`` (one per rank/chunk).
  * every stage writes ``_SUCCESS`` + ``stage.json`` (counts + cfg hash) via
    ``finalize_stage``; ``write_run_manifest`` rolls them up into run_manifest.json.
  * rows are keyed by ``(dataset, cut_id)`` with ``dataset`` an explicit column.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

SUCCESS_MARKER = "_SUCCESS"
STAGE_JSON = "stage.json"
RUN_MANIFEST = "run_manifest.json"

# Canonical stage directory names (output_dir / STAGE[...]).
STAGE = {
    "manifest":    "manifest",
    "text_dedup":  "text_dedup",
    "fingerprint": "fingerprint",
    "audio_match": "audio_match",
    "quality":     "quality",
    "retention":   "retention",
    "output_shar": "output_shar",
}

# Pipeline order (for the run manifest + readable indexing).
STAGE_ORDER = ["manifest", "text_dedup", "fingerprint", "audio_match",
               "quality", "retention", "output_shar"]

# Canonical file names within a stage dir.
CLUSTERS        = "clusters.parquet"        # text_dedup + audio_match (cols differ by dir)
CANDIDATE_EDGES = "candidate_edges.parquet"
MATCH_EDGES     = "match_edges.parquet"
HUGE_CLUSTERS   = "huge_clusters.parquet"
QUALITY_MERGED  = "merged.parquet"
ASSIGNMENTS     = "assignments.parquet"


def stage_dir(output_dir, stage: str) -> Path:
    """``output_dir / STAGE[stage]`` (raises on an unknown stage key)."""
    return Path(output_dir) / STAGE[stage]


def shard_name(rank: int, ext: str = "parquet", chunk: Optional[int] = None) -> str:
    """Uniform shard file name: ``part-00003.parquet`` (or ``part-00003_000001``
    for per-rank sub-shards before merge)."""
    base = f"part-{rank:05d}"
    if chunk is not None:
        base += f"_{chunk:06d}"
    return f"{base}.{ext}"


def finalize_stage(output_dir, stage: str, *, rows: Optional[int] = None,
                   config: Optional[dict] = None, extra: Optional[dict] = None) -> None:
    """Write the stage's completion payload to ``stage.json`` and ``_SUCCESS``."""
    d = stage_dir(output_dir, stage)
    d.mkdir(parents=True, exist_ok=True)
    info = {"stage": stage, "rows": rows,
            "config_hash": _cfg_hash(config) if config is not None else None}
    if extra:
        info.update(extra)
    payload = json.dumps(info, indent=2)
    (d / STAGE_JSON).write_text(payload)
    (d / SUCCESS_MARKER).write_text(payload)


def _cfg_hash(config: dict) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str)
                        .encode()).hexdigest()[:12]


def write_run_manifest(output_dir, *, dataset_tags=None) -> Path:
    """Roll up every present stage's stage.json into ``run_manifest.json`` — the
    clean, single-glance index of a run."""
    out = Path(output_dir)
    stages = []
    for s in STAGE_ORDER:
        d = out / STAGE[s]
        if not d.exists():
            continue
        sj = d / STAGE_JSON
        entry = {"stage": s, "dir": STAGE[s],
                 "complete": (d / SUCCESS_MARKER).exists()}
        if sj.exists():
            try:
                entry["info"] = json.loads(sj.read_text())
            except Exception:
                pass
        stages.append(entry)
    manifest = {"output_dir": str(out), "dataset_tags": dataset_tags or [],
                "stages": stages}
    p = out / RUN_MANIFEST
    p.write_text(json.dumps(manifest, indent=2))
    return p
