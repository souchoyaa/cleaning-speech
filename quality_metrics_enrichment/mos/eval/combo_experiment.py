#!/usr/bin/env python3
"""Find the best way to combine MOS metrics for retention's two cases.

Uses the labelled eval benchmark (MOS scores + ground_truth mos_eval):

  CASE 1 — DUPLICATES (keep the best copy):  pair every MOS cut with its clean
    reference (mos_eval.clean_ref_id) and ask whether the combiner ranks the
    clean original above the degraded copy.  Metric = keep-best accuracy.

  CASE 2 — UNIQUE samples (quality gate):  separate expect_low_quality cuts from
    clean originals.  Metric = AUROC, and per-family RECALL at a fixed 90%
    clean-retention operating point (so we see which combiner catches *all*
    failure modes, not just noise).

Pure stdlib.  Run on the login node.
  python3 combo_experiment.py --gt <ground_truth.jsonl> --mos <mos dir-or-jsonl>
"""
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict

QUALITY_AXES = ["utmos", "stoi", "pesq", "si_sdr", "dnsmos", "aes_ovl"]


def flat_metrics(m):
    out = {}
    if "utmos" in m:
        out["utmos"] = m["utmos"]["score"]["utmos"]
    if "squim" in m:
        s = m["squim"]["score"]
        out["stoi"], out["pesq"], out["si_sdr"] = s["stoi"], s["pesq"], s["si_sdr"]
    for k in m:
        if k.startswith("dnsmos_"):
            out["dnsmos"] = list(m[k]["score"].values())[0]
    if "audiobox" in m:
        a = m["audiobox"]["score"]
        for ax in ("CE", "CU", "PC", "PQ"):
            out[ax] = a[ax]
        out["aes_ovl"] = (a["CE"] + a["CU"] + a["PQ"]) / 3.0   # PC excluded
    return out


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sp = sum(r for r, (v, l) in zip(ranks, allv) if l == 1)
    return (sp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--mos", required=True)
    ap.add_argument("--clean-sample", type=int, default=8000)
    args = ap.parse_args()

    mos_files = ([args.mos] if args.mos.endswith(".jsonl")
                 else sorted(glob.glob(os.path.join(args.mos, "mos_rank_*.jsonl"))))
    scores = {}
    for mf in mos_files:
        for l in open(mf):
            d = json.loads(l)
            scores[d["cut_id"]] = flat_metrics(d["metrics"])

    gt = {}
    for l in open(args.gt):
        d = json.loads(l)
        gt[d["cut_id"]] = d

    clean_ids = [c for c, g in gt.items()
                 if c in scores and not g.get("is_synthetic", False)]
    rnd = random.Random(0)
    if len(clean_ids) > args.clean_sample:
        clean_ids = rnd.sample(clean_ids, args.clean_sample)
    clean = [scores[c] for c in clean_ids]
    mos_rows = [(c, scores[c], g["mos_eval"]) for c, g in gt.items()
                if c in scores and g.get("mos_eval")]
    print(f"clean sample: {len(clean)} | MOS cuts: {len(mos_rows)}\n")

    # ---- corpus z-score stats over clean + MOS cuts ----
    pool = clean + [f for _, f, _ in mos_rows]
    st = {ax: (mean([f[ax] for f in pool if ax in f]),
               std([f[ax] for f in pool if ax in f])) for ax in QUALITY_AXES}

    def z(f, ax):
        if ax not in f or ax not in st:
            return 0.0
        mu, sd = st[ax]
        return (f[ax] - mu) / sd if sd > 0 else 0.0

    # ---- combiners (higher = better quality) ----
    W = {"utmos": 0.3, "dnsmos": 0.4, "aes_ovl": 0.2}      # current retention (z)
    DIVERSE = ["dnsmos", "si_sdr", "utmos", "PQ_proxy"]    # PQ via aes? use pesq+utmos+dnsmos+si_sdr

    def c_dnsmos(f):  return z(f, "dnsmos")
    def c_utmos(f):   return z(f, "utmos")
    def c_pesq(f):    return z(f, "pesq")
    def c_wsum(f):    return sum(w * z(f, a) for a, w in W.items())
    def c_mean_all(f):return mean([z(f, a) for a in QUALITY_AXES])
    # worst-axis over a diverse set covering the distinct failure modes:
    #   dnsmos (noise/clip), si_sdr (distortion/crosstalk), utmos (naturalness),
    #   aes_ovl (telephony/band-limit), stoi (intelligibility)
    DIV = ["dnsmos", "si_sdr", "utmos", "aes_ovl", "stoi"]
    def c_minz(f):    return min(z(f, a) for a in DIV)
    def c_mean_div(f):return mean([z(f, a) for a in DIV])
    # blend: average of (mean rank) and (worst axis) — balances overall & worst
    def c_mean_min(f):return 0.5 * c_mean_div(f) + 0.5 * c_minz(f)

    combiners = {
        "dnsmos(z)": c_dnsmos, "utmos(z)": c_utmos, "pesq(z)": c_pesq,
        "wsum .3u.4d.2aes": c_wsum, "mean_z(6)": c_mean_all,
        "mean_z(div5)": c_mean_div, "MIN_z(div5)": c_minz,
        "0.5mean+0.5min": c_mean_min,
    }

    # ===== CASE 1: duplicate keep-best (clean ref vs degraded copy) =====
    print("=" * 92)
    print("CASE 1 — keep-best accuracy: combiner(clean_ref) > combiner(degraded copy)")
    print("  (paired by mos_eval.clean_ref_id; only pairs where the copy IS degraded)")
    print("=" * 92)
    pairs = []      # (family, ref_flat, copy_flat)
    for cid, f, me in mos_rows:
        ref = me.get("clean_ref_id")
        if ref in scores and me["family"] != "clean_control":
            pairs.append((me["family"], scores[ref], f))
    fam_order = sorted(set(p[0] for p in pairs))
    print("combiner".ljust(20) + "overall" +
          "".join(fam[:7].rjust(8) for fam in fam_order))
    for name, fn in combiners.items():
        overall = mean([1.0 if fn(r) > fn(c) else 0.0 for _, r, c in pairs])
        line = name.ljust(20) + f"{overall*100:6.1f}%"
        for fam in fam_order:
            fp = [(r, c) for f_, r, c in pairs if f_ == fam]
            acc = mean([1.0 if fn(r) > fn(c) else 0.0 for r, c in fp])
            line += f"{acc*100:7.0f}%"
        print(line)

    # ===== CASE 2: unique-sample quality gate =====
    print("\n" + "=" * 92)
    print("CASE 2 — unique-sample gate: separate expect_low_quality from clean")
    print("  AUROC, then per-family RECALL at the threshold that keeps 90% of clean cuts")
    print("=" * 92)
    low = [(f, me) for _, f, me in mos_rows if me.get("expect_low_quality")]
    fam_low = sorted(set(me["family"] for _, me in low))
    print("combiner".ljust(20) + "AUROC" + "  recall@90%spec ->" +
          "".join(fam[:7].rjust(8) for fam in fam_low))
    for name, fn in combiners.items():
        cl = [fn(f) for f in clean]
        lo = [fn(f) for f, _ in low]
        a = auc(cl, lo)
        thr = sorted(cl)[max(0, int(0.10 * len(cl)))]   # keep top 90% clean
        line = name.ljust(20) + f"{a:5.2f}" + " " * 14
        for fam in fam_low:
            fl = [fn(f) for f, me in low if me["family"] == fam]
            rec = mean([1.0 if v < thr else 0.0 for v in fl])
            line += f"{rec*100:7.0f}%"
        print(line)
    print("\n(recall@90%spec = fraction of that family's low-quality cuts flagged "
          "while keeping 90% of clean cuts. Higher across ALL families = better gate.)")


if __name__ == "__main__":
    main()
