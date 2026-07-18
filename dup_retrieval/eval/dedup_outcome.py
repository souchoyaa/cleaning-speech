#!/usr/bin/env python3
"""End-to-end dedup OUTCOME — the corpus-level result of the pipeline.

Unlike evaluate_dedup_eval.py / evaluate_retention.py (which need synthetic ground
truth), this reports what the pipeline actually DID and needs **no labels**, so it
is the headline number you report on any real corpus: how much of the corpus
(cuts AND hours) was removed as redundancy, broken down by reason, plus the
text/audio cluster-size distributions and quality-flag counts.

Reads retention/assignments.parquet (one row per cut: is_kept / is_duplicate /
text_cluster_id / audio_cluster_id / retention_reason / low_quality / duration).

  python3 dedup_outcome.py --final <output_dir>/retention/assignments.parquet \
      [--out report.md] [--by-dataset]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_COLS = ["dataset", "text_cluster_id", "audio_cluster_id", "is_duplicate",
         "is_kept", "low_quality", "retention_reason", "duration_secs"]


def _sizes(rows, col):
    by = defaultdict(int)
    for r in rows:
        cid = r[col]
        if cid is not None and cid >= 0:
            by[cid] += 1
    return [s for s in by.values() if s > 1]   # only real clusters (size > 1)


def _dist(sizes):
    if not sizes:
        return "n=0"
    a = np.asarray(sizes)
    return (f"n={len(a):,}  cuts={int(a.sum()):,}  mean={a.mean():.2f}  "
            f"p90={int(np.percentile(a,90))}  p99={int(np.percentile(a,99))}  max={int(a.max())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--by-dataset", action="store_true")
    args = ap.parse_args()

    t = pq.read_table(args.final, columns=_COLS).to_pylist()

    L = []
    def o(s=""):
        print(s); L.append(s)

    def hrs(rows):
        return sum((r["duration_secs"] or 0.0) for r in rows) / 3600.0

    def summarize(rows, title):
        n = len(rows)
        if n == 0:
            return
        kept = [r for r in rows if r["is_kept"]]
        h_all, h_kept = hrs(rows), hrs(kept)
        h_drop = h_all - h_kept
        n_kept, n_drop = len(kept), n - len(kept)
        n_dup = sum(1 for r in rows if r["is_duplicate"])
        n_lowq = sum(1 for r in rows if r["low_quality"])
        n_kept_lowq = sum(1 for r in rows if r["is_kept"] and r["low_quality"])

        o("=" * 74)
        o(title)
        o("=" * 74)
        o(f"corpus      {n:>10,} cuts   {h_all:9.1f} h")
        o(f"kept        {n_kept:>10,} cuts ({100*n_kept/n:5.1f}%)   {h_kept:9.1f} h ({100*h_kept/max(h_all,1e-9):5.1f}%)")
        o(f"REMOVED     {n_drop:>10,} cuts ({100*n_drop/n:5.1f}%)   {h_drop:9.1f} h ({100*h_drop/max(h_all,1e-9):5.1f}%)   <- redundancy")
        o("removed by reason (cuts / hours):")
        reasons = defaultdict(lambda: [0, 0.0])
        for r in rows:
            if not r["is_kept"]:
                reasons[r["retention_reason"]][0] += 1
                reasons[r["retention_reason"]][1] += (r["duration_secs"] or 0.0) / 3600.0
        for k in sorted(reasons, key=lambda k: -reasons[k][0]):
            c, h = reasons[k]
            o(f"    {k:30s} {c:>9,}   ({h:6.1f} h)")
        o(f"cuts in an audio cluster (is_duplicate): {n_dup:,} ({100*n_dup/n:.1f}%)")
        o(f"low_quality flagged: {n_lowq:,}   (kept-but-flagged: {n_kept_lowq:,})")
        o(f"text  cluster sizes: {_dist(_sizes(rows,'text_cluster_id'))}")
        o(f"audio cluster sizes: {_dist(_sizes(rows,'audio_cluster_id'))}")
        o("")

    summarize(t, f"DEDUP OUTCOME — {args.final}")
    if args.by_dataset:
        by = defaultdict(list)
        for r in t:
            by[r["dataset"]].append(r)
        for ds in sorted(by):
            summarize(by[ds], f"dataset = {ds}")

    if args.out:
        args.out.write_text("\n".join(L) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
