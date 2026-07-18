#!/usr/bin/env python3
"""Dedup accuracy for the make_eval_dataset.py ground truth (dup_type schema).

(The older evaluate_dedup.py targets make_synthetic_test.py's `aug` schema.)

Reads final/dedup.parquet (carries text_cluster_id + audio_cluster_id per cut)
and ground_truth.jsonl, and reports:
  * TEXT dedup recall: each cut with expect_text_dup_of=X shares X's text cluster.
  * AUDIO dedup recall: each cut with expect_audio_dup_of=X shares X's audio
    cluster — broken down per dup_type / MOS family.
  * AUDIO cluster PURITY: fraction of clustered cuts whose audio cluster contains
    only cuts of the same origin (no cross-origin merges = precision proxy).

  python3 evaluate_dedup_eval.py --final <final/dedup.parquet> --gt <ground_truth.jsonl>
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

    t = pq.read_table(args.final,
                      columns=["cut_id", "text_cluster_id", "audio_cluster_id"]).to_pylist()
    tcl = {r["cut_id"]: r["text_cluster_id"] for r in t}
    acl = {r["cut_id"]: r["audio_cluster_id"] for r in t}
    gt = {json.loads(l)["cut_id"]: json.loads(l) for l in open(args.gt)}

    def fam(cid):
        me = gt[cid].get("mos_eval")
        return me["family"] if me else gt[cid].get("dup_type", "?")

    def origin(cid):
        g = gt[cid]
        return g.get("original_id") or cid     # synthetics -> base A; originals -> self

    L = []
    def out(s=""):
        print(s); L.append(s)

    out("# Dedup accuracy (make_eval_dataset schema)\n")

    # ---- TEXT dedup recall ----
    out("=" * 70)
    out("TEXT dedup recall — cut shares its expect_text_dup_of's text cluster")
    out("=" * 70)
    by_type = defaultdict(lambda: [0, 0])
    tot = [0, 0]
    for cid, g in gt.items():
        x = g.get("expect_text_dup_of")
        if not x or x not in tcl or cid not in tcl:
            continue
        ok = tcl[cid] == tcl[x] and tcl[cid] >= 0
        by_type[fam(cid)][1] += 1
        by_type[fam(cid)][0] += ok
        tot[1] += 1
        tot[0] += ok
    for k in sorted(by_type):
        c, n = by_type[k]
        out(f"  {k:18s} {c}/{n} = {100*c/n:.1f}%")
    if tot[1]:
        out(f"  {'OVERALL':18s} {tot[0]}/{tot[1]} = {100*tot[0]/tot[1]:.1f}%")

    # ---- AUDIO dedup recall ----
    out("\n" + "=" * 70)
    out("AUDIO dedup recall — cut shares its expect_audio_dup_of's audio cluster")
    out("=" * 70)
    by_type = defaultdict(lambda: [0, 0])
    tot = [0, 0]
    for cid, g in gt.items():
        x = g.get("expect_audio_dup_of")
        if not x or x not in acl or cid not in acl:
            continue
        ok = acl[cid] == acl[x] and acl[cid] >= 0
        by_type[fam(cid)][1] += 1
        by_type[fam(cid)][0] += ok
        tot[1] += 1
        tot[0] += ok
    for k in sorted(by_type):
        c, n = by_type[k]
        out(f"  {k:18s} {c}/{n} = {100*c/n:.1f}%")
    if tot[1]:
        out(f"  {'OVERALL':18s} {tot[0]}/{tot[1]} = {100*tot[0]/tot[1]:.1f}%")

    # ---- AUDIO cluster purity ----
    out("\n" + "=" * 70)
    out("AUDIO cluster purity — clusters whose members share one origin")
    out("=" * 70)
    members = defaultdict(list)
    for cid, a in acl.items():
        if a is not None and a >= 0 and cid in gt:
            members[a].append(cid)
    pure_clusters = pure_cuts = total_cuts = 0
    multi = 0
    for a, cids in members.items():
        if len(cids) < 2:
            continue
        multi += 1
        origins = {origin(c) for c in cids}
        total_cuts += len(cids)
        if len(origins) == 1:
            pure_clusters += 1
            pure_cuts += len(cids)
    out(f"  multi-member audio clusters: {multi}")
    if multi:
        out(f"  pure clusters: {pure_clusters}/{multi} = {100*pure_clusters/multi:.1f}%")
        out(f"  cuts in pure clusters: {pure_cuts}/{total_cuts} = {100*pure_cuts/total_cuts:.1f}%")

    if args.out:
        args.out.write_text("\n".join(L) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
