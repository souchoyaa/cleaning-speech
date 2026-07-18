"""Calibration helper for Stage D + E parameters.

Given a small labeled set of duplicate / non-duplicate pairs, sweep over
``match_threshold``, ``min_relative_match``, ``energy_factor``, ``time_window``
and report recall / precision / F1.

Inputs (you provide):
- A YAML config (same as pipeline.py)
- A labeled CSV: header ``a_dataset,a_cut_id,b_dataset,b_cut_id,is_duplicate``
  one row per pair.

Workflow:
1. Build a tiny manifest restricted to the cuts in the labeled set.
2. Run Stage D fingerprint compute on those cuts (small N, single GPU).
3. For each (energy_factor, time_window) pair: recompute fingerprints.
4. For each (match_threshold, min_relative_match) pair: evaluate Hough recall/precision.
5. Print a sorted table.

This is a sanity check before running on 100 TB.  Recommend on the order of
50-200 labeled pairs.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _load_pairs(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            r["is_duplicate"] = str(r.get("is_duplicate", "")).strip().lower() in (
                "1", "true", "yes", "y", "t",
            )
            rows.append(r)
    return rows


def _evaluate(pairs: List[dict],
              fp_index: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
              durations: Dict[Tuple[str, str], float],
              match_threshold_abs: int,
              match_frac: float,
              min_rel_match: float,
              hist_tolerance: int = 0) -> dict:
    from .audio_match import _hough_match_score

    HASH_FRAME_RATE = 12.5
    tp = fp = fn = tn = 0
    for r in pairs:
        a_key = (r["a_dataset"], r["a_cut_id"])
        b_key = (r["b_dataset"], r["b_cut_id"])
        is_dup = bool(r["is_duplicate"])

        fp_a = fp_index.get(a_key)
        fp_b = fp_index.get(b_key)
        if fp_a is None or fp_b is None:
            # Treat as "no signal" -> not duplicate.
            pred = False
        else:
            ha, ta = fp_a; hb, tb = fp_b
            min_kp = min(ha.size, hb.size)
            if min_kp == 0:
                pred = False
            else:
                threshold = max(match_threshold_abs, int(match_frac * min_kp))
                max_bin, span_a, span_b = _hough_match_score(ha, ta, hb, tb, hist_tolerance)
                pred = max_bin >= threshold
                if pred and (durations.get(a_key, 0.0) > 0 and durations.get(b_key, 0.0) > 0):
                    matched_secs = max(span_a, span_b) / HASH_FRAME_RATE
                    shorter = min(durations[a_key], durations[b_key])
                    if shorter > 0 and (matched_secs / shorter) < min_rel_match:
                        pred = False

        if pred and is_dup:    tp += 1
        elif pred and not is_dup: fp += 1
        elif (not pred) and is_dup: fn += 1
        else: tn += 1

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate Stage D + E parameters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs",  required=True,
                        help="CSV with columns: a_dataset,a_cut_id,b_dataset,b_cut_id,is_duplicate")
    parser.add_argument("--match-thresholds-abs", type=str, default="4,6,8,10,15",
                        help="Comma-separated absolute match_threshold floors.")
    parser.add_argument("--match-fracs", type=str, default="0.20,0.30,0.40",
                        help="Comma-separated match_frac values.")
    parser.add_argument("--min-rel-matches", type=str, default="0.10,0.20,0.30,0.40")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pairs = _load_pairs(args.pairs)
    logger.info("Loaded %d labeled pairs", len(pairs))

    # Defer loading fingerprints + durations to caller; calibration runs *after*
    # Stage D + manifest finished (but before Stage E + F).
    output_dir = Path(__import__("yaml").safe_load(open(args.config))["output_dir"])

    import pyarrow.parquet as pq
    fp_dir   = output_dir / "fingerprint"
    fp_index: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for p in sorted(fp_dir.glob("part_*.parquet")):
        t = pq.read_table(p)
        ds = t.column("dataset").to_pylist()
        cid = t.column("cut_id").to_pylist()
        h = t.column("hash").to_pylist()
        th = t.column("t_hash").to_pylist()
        per: Dict[Tuple[str, str], Tuple[List[int], List[int]]] = {}
        for d, c, hv, tv in zip(ds, cid, h, th):
            key = (d, c)
            if key not in per:
                per[key] = ([], [])
            if hv is not None and tv is not None and tv >= 0:
                per[key][0].append(int(hv))
                per[key][1].append(int(tv))
        for key, (hh, tt) in per.items():
            if key in fp_index:
                # Append to existing.
                hh_old, tt_old = fp_index[key]
                fp_index[key] = (
                    np.concatenate([hh_old, np.asarray(hh, dtype=np.int64)]),
                    np.concatenate([tt_old, np.asarray(tt, dtype=np.int64)]),
                )
            else:
                fp_index[key] = (np.asarray(hh, dtype=np.int64),
                                 np.asarray(tt, dtype=np.int64))
    logger.info("Loaded fingerprints for %d cuts", len(fp_index))

    manifest_dir = output_dir / "manifest"
    durations: Dict[Tuple[str, str], float] = {}
    for p in sorted(manifest_dir.glob("part_*.parquet")):
        t = pq.read_table(p, columns=["dataset", "cut_id", "duration_secs"])
        for d, c, du in zip(t.column("dataset").to_pylist(),
                             t.column("cut_id").to_pylist(),
                             t.column("duration_secs").to_pylist()):
            durations[(d, c)] = float(du) if du is not None else 0.0

    abs_list = [int(x) for x in args.match_thresholds_abs.split(",") if x]
    frac_list = [float(x) for x in args.match_fracs.split(",") if x]
    rel_list  = [float(x) for x in args.min_rel_matches.split(",") if x]

    results = []
    for ab in abs_list:
        for fr in frac_list:
            for rm in rel_list:
                m = _evaluate(pairs, fp_index, durations,
                              match_threshold_abs=ab,
                              match_frac=fr,
                              min_rel_match=rm)
                results.append({"match_min_abs": ab, "match_frac": fr,
                                "min_relative_match": rm, **m})

    results.sort(key=lambda r: -r["f1"])
    print(json.dumps(results[:10], indent=2))
    print(f"\nBest config: {results[0]}")


if __name__ == "__main__":
    main()
