"""Notebook helper: suggest per-dataset quality floors for Stage F retention.

Reads quality JSONLs (mos_rank_*.jsonl, asr_moe_rank_*.jsonl), joins to the
Stage A manifest to attach the ``dataset`` tag, computes the scalar quality
score per cut (same formula Stage F uses), and prints / returns a per-dataset
percentile table.

Usage in a Jupyter cell:

    from audio_tokenization.utils.data_selection.dup_retrieval.notebook \
        import quality_floor_suggester as qfs
    df, yaml_stub = qfs.suggest(
        manifest_dir="/scratch/dedup_run/manifest",
        quality_search_paths=["/users/sgodey/.../quality_assesment/results"],
        weights={"utmos": 0.4, "dnsmos": 0.3, "audiobox": 0.2, "rover": 0.1, "language": 0.0},
        dnsmos_variant="nisqa",
    )
    print(yaml_stub)         # paste into egs/full.yaml under retention.per_dataset_quality_floor

The default ``suggested_floor`` is per-dataset p25 — adjust per dataset when
you have prior knowledge that the metric is unreliable on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq

from audio_tokenization.utils.data_selection.dup_retrieval.core.retention import (
    _quality_components,
    raw_score,
    axes_for_mode,
    load_quality_for_datasets,
)


def _load_manifest(manifest_dir: Path) -> dict:
    """Return ``{(dataset, cut_id_norm): None}`` plus parallel arrays."""
    parts = sorted(Path(manifest_dir).glob("part_*.parquet"))
    if not parts:
        raise RuntimeError(f"No manifest parts under {manifest_dir}")
    rows: List[dict] = []
    for p in parts:
        t = pq.read_table(p, columns=["dataset", "cut_id"])
        ds = t.column("dataset").to_pylist()
        cid = t.column("cut_id").to_pylist()
        for d, c in zip(ds, cid):
            rows.append({"dataset": d, "cut_id": c})
    return rows


def suggest(
    manifest_dir: str,
    quality_search_paths: List[str],
    weights: Optional[dict] = None,
    dnsmos_variant: str = "nisqa",
    pct_for_floor: float = 25.0,
):
    """Return ``(per_dataset_table, yaml_stub_string)``."""
    weights = weights or {"utmos": 0.3, "dnsmos": 0.4, "audiobox": 0.2,
                          "rover": 0.1, "language": 0.0}

    rows = _load_manifest(Path(manifest_dir))
    quality = load_quality_for_datasets([Path(p) for p in quality_search_paths])

    per_dataset: Dict[str, List[float]] = {}
    for r in rows:
        flat = quality.get((r["dataset"], r["cut_id"]))
        if flat is None:
            continue
        # Floor is the raw (un-normalized) weighted sum — suggest on that scale.
        comp = _quality_components(flat, dnsmos_variant)
        s = raw_score(comp, axes_for_mode("both"), weights) if comp else None
        if s is None:
            continue
        per_dataset.setdefault(r["dataset"], []).append(float(s))

    table_rows = []
    for ds, scores in sorted(per_dataset.items()):
        arr = np.asarray(scores, dtype=np.float64)
        table_rows.append({
            "dataset":         ds,
            "n":               int(arr.size),
            "p10":             float(np.percentile(arr, 10)),
            "p25":             float(np.percentile(arr, 25)),
            "p50":             float(np.percentile(arr, 50)),
            "p75":             float(np.percentile(arr, 75)),
            "p90":             float(np.percentile(arr, 90)),
            "suggested_floor": float(np.percentile(arr, pct_for_floor)),
        })

    # Pretty-print a YAML stub.
    lines = ["per_dataset_quality_floor:"]
    for r in table_rows:
        lines.append(f"  {r['dataset']}: {round(r['suggested_floor'], 3)}")
    yaml_stub = "\n".join(lines)
    return table_rows, yaml_stub
