#!/usr/bin/env python3
"""Per-family MOS-metric analysis for the labelled quality block of the eval set
built by ``dup_retrieval/eval/make_eval_dataset.py --mos-count N``.

Joins MOS scores (mos_rank_*.jsonl) with ground_truth.jsonl rows that carry a
``mos_eval`` dict, then reports, per degradation family:
  * AUROC of every metric (clean originals vs the family) — the metric tagged
    ``primary_axis`` is starred;
  * which metrics "catch" the family (AUROC >= 0.80) vs miss it — this is the
    coverage argument for keeping a *diverse* metric set;
  * AUROC vs severity, where the family has a numeric severity.

Pure stdlib; runs on the login node.

  python3 analyze_mos_eval.py --gt <ground_truth.jsonl> --mos <dir-or-jsonl>
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

METRICS = ["utmos", "stoi", "pesq", "si_sdr", "dnsmos", "CE", "CU", "PC", "PQ"]


def auc(pos, neg):
    """AUROC via Mann-Whitney U: P(score_clean > score_degraded)."""
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
    sum_pos = sum(r for r, (v, lab) in zip(ranks, allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


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
        for a in ("CE", "CU", "PC", "PQ"):
            out[a] = m["audiobox"]["score"][a]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--mos", required=True, help="mos_rank_*.jsonl file or its dir")
    args = ap.parse_args()

    mos_files = ([args.mos] if args.mos.endswith(".jsonl")
                 else sorted(glob.glob(os.path.join(args.mos, "mos_rank_*.jsonl"))))

    gt = {json.loads(l)["cut_id"]: json.loads(l) for l in open(args.gt)}
    scores = {}
    for mf in mos_files:
        for l in open(mf):
            d = json.loads(l)
            scores[d["cut_id"]] = flat_metrics(d["metrics"])

    # clean reference pool = real originals (is_synthetic == False)
    clean = [scores[c] for c, g in gt.items()
             if c in scores and not g.get("is_synthetic", False)]
    fam_rows = defaultdict(list)            # family -> list[(flat, mos_eval)]
    for c, g in gt.items():
        me = g.get("mos_eval")
        if me and c in scores:
            fam_rows[me["family"]].append((scores[c], me))
    print(f"clean originals: {len(clean)} | families: "
          f"{ {k: len(v) for k, v in sorted(fam_rows.items())} }\n")

    order = ["clean_control", "awgn", "real_noise", "music", "reverb",
             "crosstalk", "clipping", "telephony", "naturalness"]
    fams = [f for f in order if f in fam_rows] + \
           [f for f in fam_rows if f not in order]

    print("=" * 110)
    print("AUROC per family (clean originals vs family).  * = designed primary_axis. "
          "Bold-ish [x] = catches it (>=0.80)")
    print("=" * 110)
    print("family".ljust(14) + "n".rjust(5) + "lowQ".rjust(6) +
          "".join(m.rjust(9) for m in METRICS) + "   primary")
    for fam in fams:
        rows = fam_rows[fam]
        neg = [f for f, _ in rows]
        prim = rows[0][1].get("primary_axis")
        lowq = sum(1 for _, me in rows if me.get("expect_low_quality")) / len(rows)
        line = fam.ljust(14) + str(len(rows)).rjust(5) + f"{lowq*100:5.0f}%"
        for m in METRICS:
            a = auc([f[m] for f in clean], [f[m] for f in neg])
            tag = "*" if m == prim else " "
            line += f"{a:8.2f}{tag}"
        line += f"   {prim or '-'}"
        print(line)

    print("\n" + "=" * 110)
    print("COVERAGE — metrics that CATCH each family (AUROC>=0.80) vs MISS (<0.65). "
          "Diverse set needed where catchers differ.")
    print("=" * 110)
    for fam in fams:
        if fam == "clean_control":
            continue
        neg = [f for f, _ in fam_rows[fam]]
        a = {m: auc([f[m] for f in clean], [f[m] for f in neg]) for m in METRICS}
        catch = [m for m in METRICS if a[m] >= 0.80]
        miss = [m for m in METRICS if a[m] < 0.65]
        print(f"  {fam:13s} catch: {', '.join(catch) or '(none)'}")
        print(f"  {'':13s} miss : {', '.join(miss) or '(none)'}")

    print("\n" + "=" * 110)
    print("AUROC vs SEVERITY (numeric-severity families)")
    print("=" * 110)
    for fam in fams:
        sev_groups = defaultdict(list)
        for f, me in fam_rows[fam]:
            s = me.get("severity")
            if isinstance(s, (int, float)):
                sev_groups[s].append(f)
        if not sev_groups:
            continue
        prim = fam_rows[fam][0][1].get("primary_axis")
        show = [prim] if prim in METRICS else []
        show += [m for m in ["dnsmos", "si_sdr", "pesq", "utmos", "PQ", "PC"]
                 if m not in show][:5]
        print(f"  {fam} (primary={prim}):")
        for s in sorted(sev_groups):
            neg = sev_groups[s]
            cells = " ".join(f"{m}={auc([f[m] for f in clean], [f[m] for f in neg]):.2f}"
                             for m in show)
            print(f"    sev={str(s):7s} (n={len(neg):4d})  {cells}")


if __name__ == "__main__":
    main()
