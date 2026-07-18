"""Quality ingest — flatten the per-rank quality JSONLs into one flat,
cuDF-joinable parquet keyed by ``(dataset, cut_id)``.

This is the single JSONL → Parquet bridge between the streaming
``quality_assesment`` outputs (nested ``quality_v1`` rows: ``metrics.*``,
``rover``, ``language_consistency``, ``vad``) and the parquet "spine" that the
dedup + selection stages join on.  The dedup branch stays parquet-native (cuDF
reads it directly); the QA branch stays JSONL (append-only, nested — the right
tool for incremental GPU-worker writers).  Selection then joins everything in
parquet (optionally on GPU via cuDF).

Inputs : quality JSONLs under ``quality_search_paths`` (``mos_rank_*.jsonl`` +
         ``asr_moe_rank_*.jsonl``), validated against ``contracts/quality_v1.json``.
Outputs: ``<output_dir>/quality/merged.parquet``  (one row per (dataset,cut_id))
         ``<output_dir>/quality/_SUCCESS``
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa

logger = logging.getLogger(__name__)

SUCCESS_MARKER = "_SUCCESS"

# rover_primary_fallbacks is the only integer-valued metric; the rest are floats.
_INT_METRICS = frozenset({"rover_primary_fallbacks"})


def _pivot_quality(
    quality: Dict[Tuple[str, str], dict]
) -> Tuple[List[str], List[str], List[str], Dict[str, list]]:
    """``{(dataset,cut_id): {metric: val}}`` → columnar lists.

    Returns ``(datasets, cut_ids, metric_keys, cols)`` where ``metric_keys`` is
    the sorted union of all metrics present and ``cols[k][i]`` is row *i*'s value
    for metric *k* (``None`` where that row lacks it).  Pure Python (no pyarrow)
    so it is unit-testable without the GPU/parquet stack.
    """
    keys = sorted({k for flat in quality.values() for k in flat})
    datasets: List[str] = []
    cut_ids: List[str] = []
    cols: Dict[str, list] = {k: [] for k in keys}
    for (ds, cid), flat in quality.items():
        datasets.append(ds)
        cut_ids.append(cid)
        for k in keys:
            cols[k].append(flat.get(k))
    return datasets, cut_ids, keys, cols


def _resolve_search_paths(cfg: dict) -> List[Path]:
    """Quality JSONL search roots: ``quality_ingest.quality_search_paths`` →
    ``retention.quality_search_paths`` → the in-repo ``quality_assesment/results``."""
    cfg_q = cfg.get("quality_ingest", {}) or {}
    search = (cfg_q.get("quality_search_paths")
              or (cfg.get("retention", {}) or {}).get("quality_search_paths")
              or [])
    if not search:
        repo_results = (Path(__file__).resolve().parents[2]
                        / "quality_assesment" / "results")
        search = [str(repo_results)]
    return [Path(p) for p in search if Path(p).is_dir()]


def run(cfg: dict) -> None:
    # Lazy import: keeps module-top deps to pyarrow only so _pivot_quality is
    # importable/testable without jsonschema + the full retention module.
    from .retention import load_quality_for_datasets, _atomic_write_parquet

    output_dir = Path(cfg["output_dir"])
    qdir = output_dir / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    for p in qdir.glob("*.parquet.tmp"):
        try:
            p.unlink()
        except OSError:
            pass

    paths = _resolve_search_paths(cfg)
    if not paths:
        logger.warning("quality_ingest: no quality_search_paths exist — writing "
                       "an empty merged.parquet.")

    t0 = time.time()
    quality = load_quality_for_datasets(paths)
    datasets, cut_ids, keys, cols = _pivot_quality(quality)

    fields = [pa.field("dataset", pa.string(), nullable=False),
              pa.field("cut_id", pa.string(), nullable=False)]
    arrays = [pa.array(datasets, type=pa.string()),
              pa.array(cut_ids, type=pa.string())]
    for k in keys:
        typ = pa.int64() if k in _INT_METRICS else pa.float64()
        if k in _INT_METRICS:
            vals = [int(v) if v is not None else None for v in cols[k]]
        else:
            vals = [float(v) if v is not None else None for v in cols[k]]
        fields.append(pa.field(k, typ, nullable=True))
        arrays.append(pa.array(vals, type=typ))

    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    out = qdir / "merged.parquet"
    _atomic_write_parquet(out, table)

    from . import run_layout
    run_layout.finalize_stage(qdir.parent, "quality", rows=len(cut_ids),
                              extra={"metric_columns": keys,
                                     "search_paths": [str(p) for p in paths]})
    logger.info("quality_ingest: %d rows, %d metric cols → %s (%.1fs)",
                len(cut_ids), len(keys), out, time.time() - t0)


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quality ingest: nested quality JSONL → flat parquet")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(_load_cfg(args.config))


if __name__ == "__main__":
    main()
