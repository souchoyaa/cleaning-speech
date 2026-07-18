"""Stage F — quality-aware retention.

``quality_mode`` ∈ {mos, asr, both} selects the active axes (mos = utmos,
dnsmos, aes_ovl, si_sdr, stoi, pesq; asr = rover; both = the union). Each axis is
z-scored over the corpus so native scales are comparable; AudioBox OVL =
mean(CE, CU, PQ), excluding PC (it rises with degradation, inverting the signal).

  CASE 1 — DUPLICATES: keep one cut per audio cluster (Phase 1), then cap each
    text cluster above ``max_cluster_size`` to its top-X (Phase 2); both rank by
    mean(z over active axes).
  CASE 2 — UNIQUE (cluster_size == 1): gate_score = 0.5*mean(z) + 0.5*min(z); the
    bottom ``gate_percentile`` % are flagged ``low_quality`` (dropped if
    ``gate_drop``). An optional ``per_dataset_quality_floor`` also flags.

See ``quality_metrics_enrichment/mos/eval/FINDINGS.md`` for the empirical basis.

Inputs  : manifest/part_*.parquet, audio_match/clusters.parquet,
          text_dedup/clusters.parquet, quality JSONLs (per dataset)
Outputs : final/assignments.parquet, final/_SUCCESS
"""

import argparse
import glob
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SUCCESS_MARKER = "_SUCCESS"


# ---------------------------------------------------------------------------
# Quality JSONL loading + schema validation
# ---------------------------------------------------------------------------


_QUALITY_SCHEMA_PATH = Path(__file__).parent / "contracts" / "quality_v1.json"


def _load_validator():
    """Return a callable ``validate(d) -> bool`` from the JSON schema.

    ``jsonschema`` is a hard requirement (see PREFLIGHT.md).  Silent
    permissive validation would let every row pass and mask schema drift.
    """
    import jsonschema  # required — see PREFLIGHT.md
    with open(_QUALITY_SCHEMA_PATH) as f:
        schema = json.load(f)
    validator = jsonschema.Draft202012Validator(schema)
    def _validate(d: dict) -> bool:
        try:
            validator.validate(d)
            return True
        except jsonschema.ValidationError:
            return False
    return _validate


def _strip_cut_id_prefix(s: str) -> str:
    return s[2:] if isinstance(s, str) and s.startswith("./") else s


def _flatten_metrics(rec: dict) -> dict:
    """Pull a flat ``{metric_key: value}`` dict from a JSONL record.

    Recognized keys (any subset is fine; missing -> not in output):
      utmos                   (from metrics.utmos.score.utmos)
      dnsmos_<variant>        (from metrics.dnsmos_<variant>.score.<variant>)
      audiobox_CE/CU/PC/PQ    (from metrics.audiobox.score)
      rover_voting / rover_primary_fallbacks (from rover.*)
      language_consistency    (1.0 - 0.0; from language_consistency.all_consistent)
    """
    out: dict = {}
    metrics = rec.get("metrics") or {}

    if "utmos" in metrics:
        sc = (metrics["utmos"] or {}).get("score") or {}
        if "utmos" in sc:
            out["utmos"] = float(sc["utmos"])

    for variant in ("nisqa", "bvcc", "vcc2018"):
        key = f"dnsmos_{variant}"
        if key in metrics:
            sc = (metrics[key] or {}).get("score") or {}
            v = sc.get(variant)
            if v is not None:
                out[f"dnsmos_{variant}"] = float(v)

    if "squim" in metrics:
        sc = (metrics["squim"] or {}).get("score") or {}
        for axis in ("stoi", "pesq", "si_sdr"):
            if sc.get(axis) is not None:
                out[axis] = float(sc[axis])

    if "audiobox" in metrics:
        sc = (metrics["audiobox"] or {}).get("score") or {}
        for axis in ("CE", "CU", "PC", "PQ"):
            if axis in sc:
                out[f"audiobox_{axis}"] = float(sc[axis])
        # AES_OVL = mean of CE/CU/PQ.  PC (production complexity) is deliberately
        # EXCLUDED: it is a content descriptor, not a quality axis — degradations
        # (noise, clipping, music) raise it, so including it inverts the signal
        # (empirically AUROC ~0.05 vs degraded).  See mos/eval/FINDINGS.md.
        if all(f"audiobox_{a}" in out for a in ("CE", "CU", "PQ")):
            out["audiobox_OVL"] = sum(out[f"audiobox_{a}"]
                                       for a in ("CE", "CU", "PQ")) / 3.0

    if "rover" in rec:
        rv = rec["rover"] or {}
        if "primary_fallbacks" in rv:
            out["rover_primary_fallbacks"] = int(rv["primary_fallbacks"])

    lc = rec.get("language_consistency") or {}
    if "all_consistent" in lc:
        out["language_consistency"] = 1.0 if lc["all_consistent"] else 0.0

    return out


def _dataset_from_jsonl_path(jsonl: Path, search_root: Path) -> str:
    """Heuristic: dataset tag = relative path from search_root to jsonl's parent.

    e.g. /.../results/mos_results/granary_ytc/en/mos_rank_0.jsonl
         search_root=/.../results
         -> "granary_ytc/en"

    Strips a leading "mos_results/" or "asr_moe_results/" prefix so the tag
    matches the manifest's dataset_root convention.
    """
    rel = jsonl.parent.resolve().relative_to(search_root.resolve())
    parts = list(rel.parts)
    if parts and parts[0] in ("mos_results", "mos_results_no_audiobox",
                              "asr_moe_results"):
        parts = parts[1:]
    return "/".join(parts) if parts else "default"


def load_quality_for_datasets(quality_search_paths: List[Path]
                              ) -> Dict[Tuple[str, str], dict]:
    """Glob mos_rank_*.jsonl + asr_moe_rank_*.jsonl across the given dirs.

    Returns ``{(dataset, cut_id): flat_metrics_dict}``.  cut_ids with leading
    "./" have it stripped to match manifest convention.  Dataset tag is
    derived from the JSONL's path relative to its search_root.

    Schema-validates every row; aborts if zero rows pass validation across
    all files (defends against silent schema drift).
    """
    validate = _load_validator()
    out: Dict[Tuple[str, str], dict] = {}
    n_total = 0
    n_valid = 0
    n_files = 0
    for sp in quality_search_paths:
        for pattern in ("part_*.jsonl", "merged.jsonl",
                        "mos_rank_*.jsonl", "asr_moe_rank_*.jsonl"):
            for jsonl in sorted(sp.rglob(pattern)):
                n_files += 1
                file_ds_tag = _dataset_from_jsonl_path(jsonl, sp)
                with open(jsonl) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        n_total += 1
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not validate(d):
                            continue
                        n_valid += 1
                        cid = _strip_cut_id_prefix(d["cut_id"])
                        # Prefer the explicit in-row dataset tag (M3); fall back to
                        # the path-derived tag when the producer didn't emit one.
                        ds_tag = d.get("dataset") or file_ds_tag
                        key = (ds_tag, cid)
                        flat = out.get(key, {})
                        flat.update(_flatten_metrics(d))
                        out[key] = flat
    logger.info("Loaded quality JSONLs from %d files (%d rows total, %d valid).",
                n_files, n_total, n_valid)
    if n_files > 0 and n_valid == 0:
        raise RuntimeError(
            "Stage F: zero quality JSONL rows passed schema validation.  Either "
            "the JSONL is empty or its schema has drifted from quality_v1.  "
            "Inspect contracts/quality_v1.json + a sample row.")
    return out


def _load_quality_flat(flat_path: Path) -> Dict[Tuple[str, str], dict]:
    """Read ``quality/merged.parquet`` → ``{(dataset,cut_id): {metric: val}}``.

    Inverse of ``quality_ingest._pivot_quality``: null cells are dropped so the
    shape matches ``load_quality_for_datasets`` (only the metrics present per cut).
    """
    t = pq.read_table(flat_path)
    metric_names = [n for n in t.column_names if n not in ("dataset", "cut_id")]
    ds  = t.column("dataset").to_pylist()
    cid = t.column("cut_id").to_pylist()
    metric_cols = {m: t.column(m).to_pylist() for m in metric_names}
    out: Dict[Tuple[str, str], dict] = {}
    for i in range(len(cid)):
        flat = {m: metric_cols[m][i] for m in metric_names
                if metric_cols[m][i] is not None}
        out[(ds[i], cid[i])] = flat
    return out


def _load_quality(output_dir: Path,
                  quality_search_paths: List[Path]) -> Dict[Tuple[str, str], dict]:
    """Prefer the cuDF-joinable flat parquet (from ``quality_ingest``); fall back
    to globbing the nested JSONLs directly when it is absent."""
    flat_path = output_dir / "quality" / "merged.parquet"
    if flat_path.exists():
        q = _load_quality_flat(flat_path)
        if q:
            logger.info("Stage F: loaded %d quality rows from %s (flat bridge).",
                        len(q), flat_path)
            return q
        # An empty flat parquet (e.g. quality_ingest ran before any quality
        # JSONLs existed) must NOT shadow the JSONL fallback — otherwise every
        # cut silently scores as ungraded and retention falls back to longest.
        logger.warning("Stage F: %s has 0 rows — ignoring it and globbing quality "
                       "JSONLs directly (re-run quality_ingest to refresh it).",
                       flat_path)
    else:
        logger.info("Stage F: no merged.parquet — globbing quality JSONLs directly.")
    return load_quality_for_datasets(quality_search_paths)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


# Positive quality axes (higher = better), grouped by signal family.  ``aes_ovl``
# is the AudioBox OVL (mean CE/CU/PQ, PC excluded); ``rover`` is the ASR-consensus
# surrogate.  language is handled separately as a penalty.
_MOS_AXES = ("utmos", "dnsmos", "aes_ovl", "si_sdr", "stoi", "pesq")
_ASR_AXES = ("rover",)
# Weight key per axis for the legacy raw floor score (groups the MOS sub-axes).
_RAW_WEIGHT_KEY = {"utmos": "utmos", "dnsmos": "dnsmos", "aes_ovl": "audiobox",
                   "si_sdr": "audiobox", "stoi": "audiobox", "pesq": "audiobox",
                   "rover": "rover"}


def axes_for_mode(mode: str) -> Tuple[str, ...]:
    """Active quality axes for ``quality_mode`` ∈ {mos, asr, both}."""
    m = (mode or "both").lower()
    if m == "mos":
        return _MOS_AXES
    if m == "asr":
        return _ASR_AXES
    return _MOS_AXES + _ASR_AXES


def _quality_components(flat: dict, dnsmos_variant: str) -> dict:
    """Raw per-axis quality components present in *flat* (before z-scoring).

    Axes: utmos, dnsmos, aes_ovl (AudioBox OVL), si_sdr, stoi, pesq (SQUIM),
    rover (ASR-consensus surrogate in 0..1), language_penalty (1 - consistency).
    Missing axes omitted.
    """
    comp: dict = {}
    if "utmos" in flat:
        comp["utmos"] = float(flat["utmos"])
    dnsmos_key = f"dnsmos_{dnsmos_variant}"
    if dnsmos_key in flat:
        comp["dnsmos"] = float(flat[dnsmos_key])
    if "audiobox_OVL" in flat:
        comp["aes_ovl"] = float(flat["audiobox_OVL"])
    for axis in ("si_sdr", "stoi", "pesq"):
        if axis in flat:
            comp[axis] = float(flat[axis])
    if "rover_primary_fallbacks" in flat:
        # No "words" denom without raw transcript length; exp(-fallbacks/10) is a
        # soft consensus surrogate in (0, 1].
        comp["rover"] = math.exp(-float(flat["rover_primary_fallbacks"]) / 10.0)
    if "language_consistency" in flat:
        comp["language_penalty"] = 1.0 - float(flat["language_consistency"])
    return comp


def _component_stats(components: List[dict],
                     axes: Tuple[str, ...]) -> Dict[str, Tuple[float, float]]:
    """Corpus ``(mean, std)`` per axis, over cuts where the axis is present."""
    stats: Dict[str, Tuple[float, float]] = {}
    for axis in axes:
        vals = [c[axis] for c in components if axis in c]
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
        stats[axis] = (mu, math.sqrt(var))
    return stats


def _active_zs(comp: dict, axes: Tuple[str, ...],
               stats: Dict[str, Tuple[float, float]]) -> List[float]:
    """Z-scores of the active axes present in *comp* (zero-variance -> 0)."""
    zs: List[float] = []
    for axis in axes:
        if axis in comp and axis in stats:
            mu, sd = stats[axis]
            zs.append((comp[axis] - mu) / sd if sd > 0 else 0.0)
    return zs


def _lang_penalty(comp: dict, weights: dict) -> float:
    w = weights.get("language", 0)
    return float(w) * comp["language_penalty"] if w and "language_penalty" in comp else 0.0


def rank_score(comp: dict, axes: Tuple[str, ...],
               stats: Dict[str, Tuple[float, float]], weights: dict) -> Optional[float]:
    """Ranking score for the DUPLICATE case (keep the best copy) and the text cap.

    Mean of the active axes' z-scores — a robust consensus that beat both
    DNSMOS-weighted-sum and any single metric on the labelled keep-best benchmark
    (mos/eval/results_combo_experiment.txt).  Language penalty subtracted.
    """
    zs = _active_zs(comp, axes, stats)
    if not zs:
        return None
    return sum(zs) / len(zs) - _lang_penalty(comp, weights)


def gate_score(comp: dict, axes: Tuple[str, ...],
               stats: Dict[str, Tuple[float, float]], weights: dict) -> Optional[float]:
    """Quality-gate score for the UNIQUE case.

    ``0.5*mean(z) + 0.5*min(z)`` over the active axes: the mean captures overall
    quality, the min (worst axis) captures the failure mode no single metric
    catches (naturalness->UTMOS, band-limit->AudioBox, clipping->DNSMOS, ...).
    This blend gave the best balanced per-family recall on the benchmark.
    """
    zs = _active_zs(comp, axes, stats)
    if not zs:
        return None
    return 0.5 * (sum(zs) / len(zs)) + 0.5 * min(zs) - _lang_penalty(comp, weights)


def raw_score(comp: dict, axes: Tuple[str, ...], weights: dict) -> Optional[float]:
    """Un-normalized weighted sum on native scales — used only for the absolute
    ``per_dataset_quality_floor`` (preserves its interpretable, configured value).
    """
    score, used = 0.0, 0
    for axis in axes:
        if axis not in comp:
            continue
        w = weights.get(_RAW_WEIGHT_KEY.get(axis, axis), 0)
        if w:
            score += float(w) * comp[axis]
            used += 1
    score -= _lang_penalty(comp, weights)
    return score if used > 0 else None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """The ``pct``-th percentile (0-100) of *values* (None if empty)."""
    xs = sorted(values)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(pct / 100.0 * (len(xs) - 1)))))
    return xs[k]


# ---------------------------------------------------------------------------
# Text-cluster size cap
# ---------------------------------------------------------------------------


def select_text_cluster_drops(
    member_idxs: List[int],
    scores: List[Optional[float]],
    datasets: List[str],
    durations: List[float],
    pref_rank: Dict[str, int],
    cap: int,
) -> List[int]:
    """Return the member indices to DROP so at most ``cap`` survivors remain.

    Keeps the highest-quality members; ranking mirrors the Phase-1 keeper
    choice — quality score desc, then preferred-dataset order, then longer
    duration.  Members with no quality score rank last (dropped first).
    """
    if cap <= 0 or len(member_idxs) <= cap:
        return []
    ranked = sorted(
        member_idxs,
        key=lambda i: (
            scores[i] if scores[i] is not None else float("-inf"),
            -pref_rank.get(datasets[i], 1_000_000),
            durations[i],
        ),
        reverse=True,
    )
    return ranked[cap:]


def _selection_columns(
    datasets: List[str],
    scores_raw: List[Optional[float]],
    floor_map: dict,
    cluster_sz: List[int],
    is_kept: List[bool],
    retention: List[str],
    gate_flagged: Optional[List[bool]] = None,
) -> Tuple[List[bool], List[bool], List[str]]:
    """Derive ``(is_duplicate, low_quality, keep_reason)`` for the output.

    - ``is_duplicate``: the cut has an acoustic duplicate, i.e. it shares its
      audio cluster with >=1 other cut (``cluster_size > 1``).  A cut that is the
      sole member of its audio cluster is *unique* even if it sits in a text
      cluster (e.g. a reverberant copy the fingerprint correctly did NOT match).
    - ``low_quality``: below the absolute per-dataset floor (raw score) OR
      flagged by the unique-sample quality gate.  Purely a flag here — dropping
      (if any) was already applied upstream.
    - ``keep_reason``: dropped → the dedup/gate reason; kept + low_quality →
      ``low_quality``; otherwise ``kept``.

    Pure / no I/O so it is unit-testable without the parquet stack.
    """
    n = len(datasets)
    gate_flagged = gate_flagged or [False] * n
    is_duplicate = [cluster_sz[i] > 1 for i in range(n)]
    low_quality = [False] * n
    for i in range(n):
        sc = scores_raw[i]
        fl = floor_map.get(datasets[i])
        below_floor = sc is not None and fl is not None and sc < float(fl)
        low_quality[i] = bool(below_floor or gate_flagged[i])
    keep_reason: List[str] = []
    for i in range(n):
        if not is_kept[i]:
            keep_reason.append(retention[i])
        elif low_quality[i]:
            keep_reason.append("low_quality")
        else:
            keep_reason.append("kept")
    return is_duplicate, low_quality, keep_reason


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _atomic_write_parquet(out_path: Path, table: pa.Table) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, str(tmp), compression="zstd", compression_level=3,
                   row_group_size=200_000)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)


def run(cfg: dict) -> None:
    output_dir   = Path(cfg["output_dir"])
    manifest_dir = output_dir / "manifest"
    audio_match_dir = output_dir / "audio_match"
    final_dir    = output_dir / "retention"
    final_dir.mkdir(parents=True, exist_ok=True)
    for p in final_dir.glob("*.parquet.tmp"):
        try: p.unlink()
        except OSError: pass

    if not (manifest_dir / SUCCESS_MARKER).exists():
        raise RuntimeError("Stage A not finalized.")
    if not (audio_match_dir / SUCCESS_MARKER).exists():
        raise RuntimeError("Stage E not finalized.")

    cfg_f = cfg.get("retention", {})
    # Which signal families to use: mos | asr | both.  Selects the active quality
    # axes for both the keep-best ranking and the unique-sample gate.
    quality_mode   = str(cfg_f.get("quality_mode", "both")).lower()
    active_axes     = axes_for_mode(quality_mode)
    # Legacy raw-floor weights (used only for per_dataset_quality_floor).
    weights        = cfg_f.get("weights", {"utmos": 0.3, "dnsmos": 0.4,
                                            "audiobox": 0.2, "rover": 0.1,
                                            "language": 0.0})
    dnsmos_variant = cfg_f.get("dnsmos_variant", "nisqa")
    floor_map      = cfg_f.get("per_dataset_quality_floor", {}) or {}
    # Unique-sample quality gate: flag (and optionally drop) cuts whose gate_score
    # is in the bottom ``gate_percentile`` % of the corpus.  null disables.
    gate_percentile = cfg_f.get("gate_percentile")
    gate_drop       = bool(cfg_f.get("gate_drop", False))   # drop flagged UNIQUE cuts
    if cfg_f.get("fallback", "longest") != "longest":
        raise ValueError("retention.fallback: only 'longest' is supported")
    max_cluster_size = cfg_f.get("max_cluster_size")   # text-cluster cap; None disables
    pref_order     = cfg_f.get("preferred_dataset_order", []) or []
    quality_search = cfg_f.get("quality_search_paths") or []
    if not quality_search:
        # Reasonable default: look inside the project's quality_assesment/results.
        # Users override via cfg.
        repo_results = (Path(__file__).resolve().parents[2]
                        / "quality_assesment" / "results")
        quality_search = [str(repo_results)]
    quality_search_paths = [Path(p) for p in quality_search if Path(p).is_dir()]

    pref_rank = {ds: i for i, ds in enumerate(pref_order)}

    # ----- Manifest pass -----
    t0 = time.time()
    manifest_rows: List[Tuple[str, str, str, float]] = []  # (dataset, cut_id, recording_id, duration)
    for p in sorted(manifest_dir.glob("part_*.parquet")):
        t = pq.read_table(p, columns=["dataset", "cut_id", "recording_id", "duration_secs"])
        ds = t.column("dataset").to_pylist()
        cid = t.column("cut_id").to_pylist()
        rid = t.column("recording_id").to_pylist()
        du  = t.column("duration_secs").to_pylist()
        for d, c, r, dur in zip(ds, cid, rid, du):
            manifest_rows.append((d, c, r or "", float(dur) if dur is not None else 0.0))
    logger.info("Stage F: %d manifest rows (%.1fs)",
                len(manifest_rows), time.time() - t0)

    # ----- Audio clusters pass (Phase 1: acoustic-duplicate collapse) -----
    ac = pq.read_table(audio_match_dir / "clusters.parquet")
    ac_ds  = ac.column("dataset").to_pylist()
    ac_cid = ac.column("cut_id").to_pylist()
    ac_aid = ac.column("audio_cluster_id").to_pylist()
    cluster_members: Dict[int, List[Tuple[str, str]]] = {}
    for d, c, aid in zip(ac_ds, ac_cid, ac_aid):
        cluster_members.setdefault(int(aid), []).append((d, c))

    clustered_keys = {k for ms in cluster_members.values() for k in ms}
    logger.info("Stage F: %d audio clusters (%d cuts in clusters)",
                len(cluster_members), len(clustered_keys))

    # ----- Text clusters (Stage C) — full membership for the Phase-2 cap -----
    # clusters.parquet only covers cuts Stage E actually matched (it skips
    # oversize "huge" text clusters).  The size cap operates on *text* clusters,
    # so read the authoritative (dataset, cut_id) -> text_cluster_id from Stage C.
    text_clusters_path = output_dir / "text_dedup" / "clusters.parquet"
    if not text_clusters_path.exists():
        raise RuntimeError(f"Missing Stage C output: {text_clusters_path}")
    tc = pq.read_table(text_clusters_path,
                       columns=["dataset", "cut_id", "text_cluster_id"])
    text_cluster_of: Dict[Tuple[str, str], int] = {
        (d, c): int(t) for d, c, t in zip(tc.column("dataset").to_pylist(),
                                          tc.column("cut_id").to_pylist(),
                                          tc.column("text_cluster_id").to_pylist())
    }
    logger.info("Stage F: %d cuts carry a text_cluster_id", len(text_cluster_of))

    # ----- Quality (prefer the cuDF-joinable flat parquet from quality_ingest;
    #       fall back to globbing the JSONLs directly) -----
    quality = _load_quality(output_dir, quality_search_paths)

    # Map (dataset, cut_id) -> manifest index for cheap lookups.
    key_to_idx: Dict[Tuple[str, str], int] = {(d, c): i
                                              for i, (d, c, _, _) in enumerate(manifest_rows)}
    durations  = [m[3] for m in manifest_rows]
    recording_ids = [m[2] for m in manifest_rows]

    # Per-cut quality, three scores over the active axes (quality_mode):
    #   scores      = rank_score (mean z)        -> DUPLICATE keep-best + text cap
    #   gate_scores = 0.5*mean + 0.5*min z       -> UNIQUE-sample quality gate
    #   scores_raw  = legacy weighted raw sum    -> absolute per_dataset_quality_floor
    # All None when no active metric is present.
    components: List[Optional[dict]] = [None] * len(manifest_rows)
    for i, (d, c, _r, _du) in enumerate(manifest_rows):
        flat = quality.get((d, c))
        if flat is None:
            continue
        comp = _quality_components(flat, dnsmos_variant)
        if comp:
            components[i] = comp
    qstats = _component_stats([c for c in components if c], active_axes)
    logger.info("Stage F: quality_mode=%s, active axes=%s; z-score stats (mean,std): %s",
                quality_mode, list(qstats.keys()),
                {k: (round(m, 3), round(s, 3)) for k, (m, s) in qstats.items()})
    scores: List[Optional[float]] = [None] * len(manifest_rows)
    gate_scores: List[Optional[float]] = [None] * len(manifest_rows)
    scores_raw: List[Optional[float]] = [None] * len(manifest_rows)
    for i, comp in enumerate(components):
        if comp:
            scores[i] = rank_score(comp, active_axes, qstats, weights)
            gate_scores[i] = gate_score(comp, active_axes, qstats, weights)
            scores_raw[i] = raw_score(comp, active_axes, weights)
    # Corpus threshold for the unique-sample gate (bottom gate_percentile %).
    gate_thr = None
    if gate_percentile is not None:
        gate_thr = _percentile([g for g in gate_scores if g is not None],
                               float(gate_percentile))
        logger.info("Stage F: unique-sample gate at p%.1f -> gate_score < %s (drop=%s)",
                    float(gate_percentile), None if gate_thr is None else round(gate_thr, 4),
                    gate_drop)

    # ----- Per-cluster retention -----
    is_kept     = [True]  * len(manifest_rows)
    retention   = ["singleton"] * len(manifest_rows)
    cluster_ids = [-1]    * len(manifest_rows)
    cluster_sz  = [1]     * len(manifest_rows)
    text_cluster_ids = [text_cluster_of.get((m[0], m[1]), -1)
                        for m in manifest_rows]

    for aid, members in cluster_members.items():
        idxs = [key_to_idx.get(k) for k in members]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            continue
        for i in idxs:
            cluster_ids[i] = int(aid)
            cluster_sz[i]  = len(members)

        # Reliable members under per-dataset floor (absolute, raw score).
        reliable: List[int] = []
        for i in idxs:
            ds_tag = manifest_rows[i][0]
            sc_raw = scores_raw[i]
            floor = floor_map.get(ds_tag)
            if sc_raw is None or floor is None:
                # No floor specified -> treat any scored cut as reliable.
                if sc_raw is not None:
                    reliable.append(i)
            elif sc_raw >= float(floor):
                reliable.append(i)

        if reliable:
            # Rank reliable members by the z-scored quality (relative best copy).
            keeper = max(reliable, key=lambda i: (
                scores[i] or float("-inf"),
                -pref_rank.get(manifest_rows[i][0], 1_000_000),
                durations[i],
            ))
            reason = "quality"
        else:
            keeper = max(idxs, key=lambda i: (
                durations[i],
                -pref_rank.get(manifest_rows[i][0], 1_000_000),
            ))
            reason = "longest_fallback"

        for i in idxs:
            is_kept[i]   = (i == keeper)
            retention[i] = reason if i == keeper else f"{reason}_dropped"

    # ----- Phase 2: text-cluster size cap -----
    # Independently of acoustic dedup, stop any single text cluster from
    # dominating training: among the cuts that survived Phase 1, if a text
    # cluster still has more than ``max_cluster_size`` members, keep only the
    # top-X by quality and drop the rest — even when their audio differs.
    n_text_capped = 0
    n_capped_clusters = 0
    if max_cluster_size is not None and int(max_cluster_size) > 0:
        cap = int(max_cluster_size)
        survivors_by_tc: Dict[int, List[int]] = {}
        for i in range(len(manifest_rows)):
            tc_id = text_cluster_ids[i]
            if is_kept[i] and tc_id >= 0:
                survivors_by_tc.setdefault(tc_id, []).append(i)
        datasets = [m[0] for m in manifest_rows]
        for members in survivors_by_tc.values():
            drops = select_text_cluster_drops(
                members, scores, datasets, durations, pref_rank, cap)
            if not drops:
                continue
            n_capped_clusters += 1
            for i in drops:
                is_kept[i]   = False
                retention[i] = "text_cluster_cap_dropped"
                n_text_capped += 1
        logger.info("Stage F: text-cluster cap (max=%d) dropped %d cuts across "
                    "%d oversize clusters.", cap, n_text_capped, n_capped_clusters)

    # ----- Unique-sample quality gate (report case 2) -----
    # A cut is "unique" when it is acoustically ALONE (cluster_size == 1) — it has
    # no acoustic duplicate, whether or not it sits in a text cluster.  Flag those
    # whose gate_score is in the bottom gate_percentile %; optionally drop them
    # (gate_drop).  Cuts WITH acoustic duplicates (cluster_size > 1) are Case 1:
    # handled by keep-best above and never dropped by the gate (only flagged).
    gate_flagged = [False] * len(manifest_rows)
    n_gate_dropped = 0
    if gate_thr is not None:
        for i in range(len(manifest_rows)):
            g = gate_scores[i]
            if g is not None and g < gate_thr:
                gate_flagged[i] = True
                if gate_drop and is_kept[i] and cluster_sz[i] <= 1:
                    is_kept[i] = False
                    retention[i] = "below_quality_gate_dropped"
                    n_gate_dropped += 1
        logger.info("Stage F: unique-sample gate flagged %d cuts (%d unique cuts "
                    "dropped).", sum(gate_flagged), n_gate_dropped)

    # low_quality = below the absolute per-dataset floor (raw) OR below the gate.
    is_duplicate, low_quality, keep_reason = _selection_columns(
        [m[0] for m in manifest_rows], scores_raw, floor_map,
        cluster_sz, is_kept, retention, gate_flagged)

    # ----- Write parquet -----
    schema = pa.schema([
        pa.field("dataset",          pa.string(),  nullable=False),
        pa.field("cut_id",           pa.string(),  nullable=False),
        pa.field("recording_id",     pa.string(),  nullable=True),
        pa.field("text_cluster_id",  pa.int64(),   nullable=False),
        pa.field("audio_cluster_id", pa.int64(),   nullable=False),
        pa.field("cluster_size",     pa.int64(),   nullable=False),
        pa.field("is_duplicate",     pa.bool_(),   nullable=False),
        pa.field("is_kept",          pa.bool_(),   nullable=False),
        pa.field("low_quality",      pa.bool_(),   nullable=False),
        pa.field("keep_reason",      pa.string(),  nullable=False),
        pa.field("retention_reason", pa.string(),  nullable=False),
        pa.field("quality_score",    pa.float32(), nullable=True),
        pa.field("gate_score",       pa.float32(), nullable=True),
        pa.field("duration_secs",    pa.float32(), nullable=True),
    ])

    n = len(manifest_rows)
    table = pa.Table.from_arrays(
        [
            pa.array([m[0] for m in manifest_rows]),
            pa.array([m[1] for m in manifest_rows]),
            pa.array(recording_ids),
            pa.array(text_cluster_ids, type=pa.int64()),
            pa.array(cluster_ids,      type=pa.int64()),
            pa.array(cluster_sz,       type=pa.int64()),
            pa.array(is_duplicate,     type=pa.bool_()),
            pa.array(is_kept,          type=pa.bool_()),
            pa.array(low_quality,      type=pa.bool_()),
            pa.array(keep_reason),
            pa.array(retention),
            pa.array([s if s is not None else None for s in scores], type=pa.float32()),
            pa.array([g if g is not None else None for g in gate_scores], type=pa.float32()),
            pa.array(durations, type=pa.float32()),
        ],
        schema=schema,
    )
    out_assignments = final_dir / "assignments.parquet"
    _atomic_write_parquet(out_assignments, table)

    n_kept = int(sum(1 for v in is_kept if v))
    n_dropped = n - n_kept
    n_clustered = int(sum(1 for s in cluster_sz if s > 1))
    n_low_quality = int(sum(1 for v in low_quality if v))
    n_kept_low_quality = int(sum(1 for i in range(n) if is_kept[i] and low_quality[i]))
    logger.info("Stage F done: %d total cuts, %d kept (%d dropped: %d acoustic, "
                "%d text-cap), %d in audio clusters, %d kept-but-low-quality.",
                n, n_kept, n_dropped, n_dropped - n_text_capped, n_text_capped,
                n_clustered, n_kept_low_quality)

    from . import run_layout
    run_layout.finalize_stage(final_dir.parent, "retention", rows=n, extra={
        "n_kept":      n_kept,
        "n_dropped":   n_dropped,
        "n_clustered": n_clustered,
        "n_low_quality":      n_low_quality,
        "n_kept_low_quality": n_kept_low_quality,
        "max_cluster_size":         max_cluster_size,
        "n_text_cluster_capped":    n_text_capped,
        "n_text_clusters_capped":   n_capped_clusters,
        "quality_mode":   quality_mode,
        "active_axes":    list(qstats.keys()),
        "gate_percentile": gate_percentile,
        "gate_drop":       gate_drop,
        "gate_threshold":  gate_thr,
        "n_gate_dropped":  n_gate_dropped,
        "weights":     weights,
        "dnsmos_variant": dnsmos_variant,
        "quality_zscore_stats": {k: [m, s] for k, (m, s) in qstats.items()},
    })
    logger.info("Wrote %s", final_dir / SUCCESS_MARKER)


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage F: retention")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(_load_cfg(args.config))


if __name__ == "__main__":
    main()
