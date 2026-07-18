#!/usr/bin/env python3
"""Evaluate Stage-F retention against the eval-dataset ground truth, for the two
report cases.

CASE 1 — DUPLICATES (keep the best copy): among audio clusters that contain both
  a clean member (original/exact/clean_control) and a low-quality member
  (mos_eval.expect_low_quality), is the KEPT cut a clean one (not the degraded)?

CASE 2 — UNIQUE samples (quality gate): among cuts in no audio cluster that carry
  a mos_eval label, does ``low_quality`` match ``expect_low_quality``
  (precision/recall/F1 + per-family recall)?  Plus the false-positive rate on
  clean unique originals (should ≈ gate_percentile).

  python3 evaluate_retention.py --final <dedup_out/final/dedup.parquet> \
      --gt <eval/en/ground_truth.jsonl> [--out report.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = pq.read_table(args.final).to_pylist()
    gt = {}
    for l in open(args.gt):
        d = json.loads(l)
        gt[d["cut_id"]] = d
    rows = [r for r in rows if r["cut_id"] in gt]

    def is_clean(cid):
        g = gt[cid]
        me = g.get("mos_eval")
        if me:
            return me["family"] == "clean_control"
        return g.get("dup_type") in ("original", "exact")

    def is_lowq(cid):
        me = gt[cid].get("mos_eval")
        return bool(me and me.get("expect_low_quality"))

    def family(cid):
        me = gt[cid].get("mos_eval")
        return me["family"] if me else gt[cid].get("dup_type", "?")

    L = []
    def out(s=""):
        print(s); L.append(s)

    n = len(rows)
    n_kept = sum(1 for r in rows if r["is_kept"])
    n_dup = sum(1 for r in rows if r["is_duplicate"])
    n_low = sum(1 for r in rows if r["low_quality"])
    reasons = defaultdict(int)
    for r in rows:
        reasons[r["keep_reason"]] += 1
    out(f"# Retention evaluation ({n} cuts)\n")
    out(f"kept={n_kept}  dropped={n-n_kept}  in-audio-cluster={n_dup}  low_quality={n_low}")
    out(f"keep_reason: {dict(reasons)}\n")

    # ---------------- CASE 1 ----------------
    out("=" * 78)
    out("CASE 1 — DUPLICATES: keep-best within audio clusters")
    out("=" * 78)
    clusters = defaultdict(list)
    for r in rows:
        aid = r["audio_cluster_id"]
        if aid is not None and aid >= 0:
            clusters[aid].append(r)
    contrast = 0          # clusters with >=1 clean AND >=1 low-quality member
    kept_clean = 0        # kept member is clean
    kept_lowq = 0         # kept member is low-quality (an error)
    for aid, members in clusters.items():
        cids = [m["cut_id"] for m in members]
        if not (any(is_clean(c) for c in cids) and any(is_lowq(c) for c in cids)):
            continue
        contrast += 1
        kept = [m for m in members if m["is_kept"]]
        if kept:
            kc = kept[0]["cut_id"]
            kept_clean += is_clean(kc)
            kept_lowq += is_lowq(kc)
    out(f"audio clusters: {len(clusters)}")
    out(f"clusters with quality contrast (clean + low-quality member): {contrast}")
    if contrast:
        out(f"  kept member is CLEAN:        {kept_clean}/{contrast} = {100*kept_clean/contrast:.1f}%")
        out(f"  kept member is LOW-QUALITY:  {kept_lowq}/{contrast} = {100*kept_lowq/contrast:.1f}%  (errors)")
    out("")

    # ---------------- CASE 2 ----------------
    # "unique" = acoustically alone (is_duplicate False, i.e. cluster_size == 1).
    out("=" * 78)
    out("CASE 2 — UNIQUE samples (acoustically alone): quality gate")
    out("=" * 78)

    def report_gate(pop, title):
        tp = sum(1 for r in pop if is_lowq(r["cut_id"]) and r["low_quality"])
        fn = sum(1 for r in pop if is_lowq(r["cut_id"]) and not r["low_quality"])
        fp = sum(1 for r in pop if not is_lowq(r["cut_id"]) and r["low_quality"])
        tn = sum(1 for r in pop if not is_lowq(r["cut_id"]) and not r["low_quality"])
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        out(f"{title} (n={len(pop)}, low-quality={tp+fn}, clean={tn+fp}):")
        out(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  "
            f"(TP={tp} FP={fp} FN={fn} TN={tn})")
        fam_rec = defaultdict(lambda: [0, 0])
        for r in pop:
            if is_lowq(r["cut_id"]):
                fam_rec[family(r["cut_id"])][1] += 1
                fam_rec[family(r["cut_id"])][0] += r["low_quality"]
        out("  per-family recall:")
        for fam in sorted(fam_rec):
            c, t = fam_rec[fam]
            out(f"    {fam:14s} {c}/{t} = {100*c/t:.0f}%")

    uniq_mos = [r for r in rows
                if not r["is_duplicate"] and gt[r["cut_id"]].get("mos_eval")]
    report_gate(uniq_mos, "Gate on UNIQUE (acoustically-alone) MOS cuts")
    out("")
    all_mos = [r for r in rows if gt[r["cut_id"]].get("mos_eval")]
    report_gate(all_mos, "Gate flag over ALL MOS cuts (incl. acoustic duplicates)")
    # false-positive rate on clean originals (the cuts we must NOT flag)
    clean_orig = [r for r in rows if gt[r["cut_id"]].get("dup_type") == "original"]
    if clean_orig:
        fpr = sum(1 for r in clean_orig if r["low_quality"]) / len(clean_orig)
        out(f"\nfalse-positive rate on clean originals: {100*fpr:.1f}% "
            f"(target <= gate_percentile)")

    if args.out:
        args.out.write_text("\n".join(L) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
