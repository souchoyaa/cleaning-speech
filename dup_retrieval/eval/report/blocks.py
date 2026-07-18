"""Report building blocks — the catalog the user picks from.

Each block is a small function that turns a PipelineReport into a BlockResult
(a markdown snippet — sentence or table — plus an optional matplotlib figure).
Blocks are grouped by pipeline component so a report can be assembled
component-by-component or end-to-end.  Register with the @block decorator;
the CATALOG (and the CLI) pick them up automatically.

Conventions
-----------
* needs_gt=True   -> requires ground_truth.jsonl (synthetic eval only).
* multilingual=True -> only interesting with >1 dataset/language (still renders).
* A block whose inputs are missing returns a short "n/a" sentence, never raises.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))  # flat imports when run directly

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from loader import PipelineReport, df_to_md, norm_words  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

try:
    from rapidfuzz.distance import Levenshtein as _Lev
    def _word_ed(a, b):
        return _Lev.distance(a, b)
except Exception:                                            # pragma: no cover
    def _word_ed(a, b):                                      # tiny DP fallback
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                            prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[m]


# --------------------------------------------------------------------------- #
# block model + registry
# --------------------------------------------------------------------------- #

@dataclass
class BlockResult:
    markdown: str = ""
    fig: object = None            # matplotlib Figure | None


@dataclass
class Block:
    id: str
    title: str
    category: str
    kind: str                      # figure | table | sentence | figure+table
    fn: Callable[[PipelineReport], BlockResult]
    needs_gt: bool = False
    multilingual: bool = False
    desc: str = ""


CATALOG: List[Block] = []


def block(id, title, category, kind, needs_gt=False, multilingual=False, desc=""):
    def deco(fn):
        CATALOG.append(Block(id, title, category, kind, fn, needs_gt, multilingual, desc))
        return fn
    return deco


def _na(msg: str) -> BlockResult:
    return BlockResult(markdown=f"_(not available: {msg})_")


def _hours(df, mask=None) -> float:
    s = df["duration_secs"] if mask is None else df.loc[mask, "duration_secs"]
    return float(s.sum()) / 3600.0


# --------------------------------------------------------------------------- #
# 0. OVERVIEW / ENTIRETY
# --------------------------------------------------------------------------- #

@block("overview_summary", "Pipeline run summary", "overview", "table",
       desc="Stages run + corpus totals (cuts, hours, kept, removed).")
def overview_summary(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    n, h = len(a), _hours(a)
    kept = a["is_kept"]
    nk, hk = int(kept.sum()), _hours(a, kept)
    stages = [s["stage"] for s in rep.manifest.get("stages", []) if s.get("complete")]
    tbl = pd.DataFrame([
        ["corpus cuts", f"{n:,}"],
        ["corpus hours", f"{h:.1f}"],
        ["kept cuts", f"{nk:,} ({100*nk/n:.1f}%)"],
        ["kept hours", f"{hk:.1f} ({100*hk/max(h,1e-9):.1f}%)"],
        ["removed cuts", f"{n-nk:,} ({100*(n-nk)/n:.1f}%)"],
        ["removed hours", f"{h-hk:.1f}"],
        ["datasets", ", ".join(rep.datasets)],
        ["stages complete", ", ".join(stages)],
    ], columns=["metric", "value"])
    s = (f"The corpus holds **{n:,} cuts / {h:.1f} h**; after the pipeline "
         f"**{nk:,} cuts ({100*nk/n:.1f}%) / {hk:.1f} h** are kept and "
         f"**{n-nk:,} ({100*(n-nk)/n:.1f}%)** removed.")
    return BlockResult(markdown=s + "\n\n" + df_to_md(tbl), fig=None)


@block("stage_funnel", "Corpus reduction funnel (hours)", "overview", "figure",
       desc="Waterfall of hours: total -> minus redundant -> minus low-quality -> kept.")
def stage_funnel(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    total = _hours(a)
    dropped = ~a["is_kept"]
    redundant = _hours(a, dropped & a["is_duplicate"])
    lowq = _hours(a, dropped & ~a["is_duplicate"])
    kept = _hours(a, a["is_kept"])
    labels = ["Total", "− redundant", "− low-quality", "Kept"]
    vals = [total, -redundant, -lowq, kept]
    fig, ax = plt.subplots(figsize=(7, 4))
    running = 0.0
    for i, (lab, v) in enumerate(zip(labels, vals)):
        if lab in ("Total", "Kept"):
            ax.bar(i, v if lab == "Total" else v, color="#3b6ea5")
            running = v
        else:
            ax.bar(i, v, bottom=running, color="#c44e52")
            running += v
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel("hours"); ax.set_title("Corpus reduction (hours)")
    for i, v in enumerate(vals):
        ax.text(i, max(total * 0.02, 1), f"{abs(v):.0f}h", ha="center", va="bottom")
    fig.tight_layout()
    s = (f"Of **{total:.0f} h**, **{redundant:.0f} h** are dropped as redundancy "
         f"and **{lowq:.0f} h** by the quality gate, leaving **{kept:.0f} h** kept.")
    return BlockResult(markdown=s, fig=fig)


# --------------------------------------------------------------------------- #
# 1. MANIFEST / COMPOSITION
# --------------------------------------------------------------------------- #

@block("dataset_composition", "Dataset / language composition", "manifest",
       "figure+table", multilingual=True,
       desc="Cuts and hours per dataset (the language axis in a multilingual run).")
def dataset_composition(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    g = a.groupby("dataset").agg(cuts=("cut_id", "size"),
                                 hours=("duration_secs", lambda s: s.sum() / 3600.0))
    g = g.sort_values("hours", ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(g) + 3), 4))
    ax.bar(g["dataset"], g["hours"], color="#3b6ea5")
    ax.set_ylabel("hours"); ax.set_title("Hours per dataset")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    g["hours"] = g["hours"].round(1)
    return BlockResult(markdown=df_to_md(g), fig=fig)


@block("duration_hist", "Cut-duration distribution", "manifest", "figure",
       desc="Histogram of per-cut durations.")
def duration_hist(rep: PipelineReport) -> BlockResult:
    d = rep.assignments["duration_secs"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=60, color="#3b6ea5")
    ax.set_xlabel("duration (s)"); ax.set_ylabel("cuts")
    ax.set_title("Cut-duration distribution")
    fig.tight_layout()
    s = (f"Durations: median **{np.median(d):.1f}s**, mean **{d.mean():.1f}s**, "
         f"p95 **{np.percentile(d,95):.1f}s** (n={len(d):,}).")
    return BlockResult(markdown=s, fig=fig)


# --------------------------------------------------------------------------- #
# 2. TEXT DEDUP (Stage C)
# --------------------------------------------------------------------------- #

def _cluster_sizes(df, col):
    if df is None or col not in df.columns:
        return np.array([])
    v = df.loc[df[col] >= 0, col]
    sizes = v.value_counts().to_numpy()
    return sizes[sizes > 1]


@block("text_cluster_sizes", "Text-cluster size distribution", "text_dedup", "figure",
       desc="Histogram of near-duplicate TEXT cluster sizes (>1).")
def text_cluster_sizes(rep: PipelineReport) -> BlockResult:
    sizes = _cluster_sizes(rep.assignments, "text_cluster_id")
    if sizes.size == 0:
        return _na("no text clusters")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes, bins=range(2, int(sizes.max()) + 2), color="#55a868")
    ax.set_yscale("log"); ax.set_xlabel("cluster size"); ax.set_ylabel("# clusters")
    ax.set_title("Text near-duplicate cluster sizes")
    fig.tight_layout()
    s = (f"**{len(sizes):,}** text clusters (size>1) covering **{int(sizes.sum()):,}** "
         f"cuts; mean size **{sizes.mean():.2f}**, max **{int(sizes.max())}**.")
    return BlockResult(markdown=s, fig=fig)


@block("text_dedup_summary", "Text-dedup summary", "text_dedup", "sentence",
       desc="One-line headline: clusters, cuts in text duplicates, share of corpus.")
def text_dedup_summary(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    sizes = _cluster_sizes(a, "text_cluster_id")
    in_clusters = int(sizes.sum()) if sizes.size else 0
    n = len(a)
    s = (f"Text dedup formed **{len(sizes):,}** multi-cut clusters holding "
         f"**{in_clusters:,} cuts ({100*in_clusters/n:.1f}%** of the corpus); "
         f"largest text cluster **{int(sizes.max()) if sizes.size else 0}** cuts.")
    return BlockResult(markdown=s)


# --------------------------------------------------------------------------- #
# 3. AUDIO DEDUP (Stage D/E)
# --------------------------------------------------------------------------- #

@block("audio_cluster_sizes", "Audio-cluster size distribution", "audio_dedup", "figure",
       desc="Histogram of acoustic-duplicate AUDIO cluster sizes (>1).")
def audio_cluster_sizes(rep: PipelineReport) -> BlockResult:
    sizes = _cluster_sizes(rep.assignments, "audio_cluster_id")
    if sizes.size == 0:
        return _na("no audio clusters")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes, bins=range(2, int(sizes.max()) + 2), color="#c44e52")
    ax.set_yscale("log"); ax.set_xlabel("cluster size"); ax.set_ylabel("# clusters")
    ax.set_title("Audio duplicate cluster sizes")
    fig.tight_layout()
    s = (f"**{len(sizes):,}** audio clusters (size>1) covering **{int(sizes.sum()):,}** "
         f"cuts; mean **{sizes.mean():.2f}**, max **{int(sizes.max())}**.")
    return BlockResult(markdown=s, fig=fig)


@block("audio_match_scores", "Audio match-score distribution", "audio_dedup", "figure",
       desc="Histogram of max_match_score and matched_span_secs from Stage E.")
def audio_match_scores(rep: PipelineReport) -> BlockResult:
    ac = rep.audio_clusters
    if ac is None or "max_match_score" not in ac.columns:
        return _na("audio_match/clusters.parquet missing")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(ac["max_match_score"].dropna(), bins=50, color="#c44e52")
    axes[0].set_title("max_match_score"); axes[0].set_xlabel("score")
    axes[1].hist(ac["matched_span_secs"].dropna(), bins=50, color="#8172b3")
    axes[1].set_title("matched_span_secs"); axes[1].set_xlabel("seconds")
    fig.tight_layout()
    return BlockResult(markdown="Distribution of Stage-E match strength.", fig=fig)


@block("audio_dedup_summary", "Audio-dedup summary", "audio_dedup", "sentence",
       desc="One-line headline: acoustic clusters and share of corpus in them.")
def audio_dedup_summary(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    sizes = _cluster_sizes(a, "audio_cluster_id")
    in_c = int(sizes.sum()) if sizes.size else 0
    n = len(a)
    s = (f"Audio matching formed **{len(sizes):,}** acoustic clusters holding "
         f"**{in_c:,} cuts ({100*in_c/n:.1f}%)**; these are the redundant copies "
         f"retention then thins to one best per cluster.")
    return BlockResult(markdown=s)


# --------------------------------------------------------------------------- #
# 4. DEDUP ACCURACY (ground-truth)
# --------------------------------------------------------------------------- #

def _gt_family(gt, cid):
    me = gt[cid].get("mos_eval")
    return me["family"] if me else gt[cid].get("dup_type", "?")


def _recall_by_family(rep, expect_key, cluster_col):
    """{family: [ok, n]}, [ok, n] overall — does each labelled dup share its
    source's cluster?  Shared by the recall figure and the limitations block."""
    gt, a = rep.ground_truth, rep.assignments
    cl = dict(zip(a["cut_id"], a[cluster_col]))
    by, tot = {}, [0, 0]
    for cid, g in gt.items():
        x = g.get(expect_key)
        if not x or x not in cl or cid not in cl:
            continue
        ok = cl[cid] == cl[x] and cl[cid] >= 0
        fam = _gt_family(gt, cid)
        by.setdefault(fam, [0, 0]); by[fam][1] += 1; by[fam][0] += ok
        tot[1] += 1; tot[0] += ok
    return by, tot


@block("dedup_recall_by_family", "Dedup recall by family (TEXT vs AUDIO)",
       "dedup_accuracy", "figure+table", needs_gt=True,
       desc="Per dup-type recall: does each synthetic dup land in its source's cluster?")
def dedup_recall_by_family(rep: PipelineReport) -> BlockResult:
    gt = rep.ground_truth
    if not gt:
        return _na("no ground_truth.jsonl")
    tby, ttot = _recall_by_family(rep, "expect_text_dup_of", "text_cluster_id")
    aby, atot = _recall_by_family(rep, "expect_audio_dup_of", "audio_cluster_id")
    fams = sorted(set(tby) | set(aby))
    rows = []
    for f in fams:
        tr = 100 * tby[f][0] / tby[f][1] if f in tby and tby[f][1] else np.nan
        ar = 100 * aby[f][0] / aby[f][1] if f in aby and aby[f][1] else np.nan
        rows.append([f, tr, ar])
    df = pd.DataFrame(rows, columns=["family", "text_recall_%", "audio_recall_%"])
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(fams) + 3), 4))
    x = np.arange(len(fams)); w = 0.4
    ax.bar(x - w/2, df["text_recall_%"], w, label="TEXT", color="#55a868")
    ax.bar(x + w/2, df["audio_recall_%"], w, label="AUDIO", color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=45, ha="right")
    ax.set_ylabel("recall (%)"); ax.set_ylim(0, 105); ax.legend()
    ax.set_title("Dedup recall by family")
    fig.tight_layout()
    to = (f"Overall recall — TEXT **{100*ttot[0]/max(ttot[1],1):.1f}%** "
          f"({ttot[0]}/{ttot[1]}), AUDIO **{100*atot[0]/max(atot[1],1):.1f}%** "
          f"({atot[0]}/{atot[1]}).")
    return BlockResult(markdown=to + "\n\n" + df_to_md(df), fig=fig)


@block("audio_cluster_purity", "Audio-cluster purity (false-merge proxy)",
       "dedup_accuracy", "table", needs_gt=True,
       desc="Fraction of multi-member audio clusters whose members share one origin.")
def audio_cluster_purity(rep: PipelineReport) -> BlockResult:
    gt = rep.ground_truth
    if not gt:
        return _na("no ground_truth.jsonl")
    a = rep.assignments
    members = {}
    for cid, aid in zip(a["cut_id"], a["audio_cluster_id"]):
        if aid is not None and aid >= 0 and cid in gt:
            members.setdefault(aid, []).append(cid)
    multi = pure_c = pure_cuts = total_cuts = 0
    for aid, cids in members.items():
        if len(cids) < 2:
            continue
        multi += 1; total_cuts += len(cids)
        origins = {(gt[c].get("original_id") or c) for c in cids}
        if len(origins) == 1:
            pure_c += 1; pure_cuts += len(cids)
    if not multi:
        return BlockResult(markdown="No multi-member audio clusters.")
    s = (f"**{100*pure_c/multi:.2f}%** of multi-member audio clusters are pure "
         f"({pure_c}/{multi}); **{100*pure_cuts/total_cuts:.2f}%** of clustered cuts "
         f"sit in a pure cluster — a precision/false-merge proxy.")
    return BlockResult(markdown=s)


# --------------------------------------------------------------------------- #
# 5. QUALITY (Stage Q / retention gate)
# --------------------------------------------------------------------------- #

_QMETRICS = ["utmos", "dnsmos_nisqa", "audiobox_OVL", "pesq", "stoi", "si_sdr"]


@block("quality_distributions", "Quality-metric distributions", "quality", "figure",
       desc="Small-multiples histograms of the MOS/quality metrics.")
def quality_distributions(rep: PipelineReport) -> BlockResult:
    q = rep.quality
    if q is None:
        return _na("quality/merged.parquet missing")
    cols = [c for c in _QMETRICS if c in q.columns]
    if not cols:
        return _na("no known quality columns")
    ncol = 3; nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        ax.hist(q[c].dropna(), bins=50, color="#4c72b0")
        ax.set_title(c)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return BlockResult(markdown=f"Distributions of {len(cols)} quality metrics "
                       f"over {len(q):,} cuts.", fig=fig)


@block("quality_correlation", "Quality-metric correlation", "quality", "figure",
       desc="Correlation heatmap among quality metrics (redundancy check).")
def quality_correlation(rep: PipelineReport) -> BlockResult:
    q = rep.quality
    if q is None:
        return _na("quality/merged.parquet missing")
    cols = [c for c in _QMETRICS if c in q.columns]
    if len(cols) < 2:
        return _na("need >=2 quality columns")
    corr = q[cols].corr()
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 2, 1.0 * len(cols) + 2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Quality metric correlation")
    fig.tight_layout()
    return BlockResult(markdown="Pearson correlation among quality metrics.", fig=fig)


@block("quality_gate_summary", "Quality-gate summary", "quality", "table",
       desc="How many cuts the gate flags low-quality, dropped vs kept-but-flagged.")
def quality_gate_summary(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    n = len(a)
    lowq = int(a["low_quality"].sum())
    kept_flag = int((a["low_quality"] & a["is_kept"]).sum())
    dropped_flag = lowq - kept_flag
    tbl = pd.DataFrame([
        ["cuts", f"{n:,}"],
        ["flagged low_quality", f"{lowq:,} ({100*lowq/n:.1f}%)"],
        ["  dropped by gate", f"{dropped_flag:,}"],
        ["  kept but flagged", f"{kept_flag:,}"],
    ], columns=["metric", "value"])
    s = (f"The quality gate flags **{lowq:,} cuts ({100*lowq/n:.1f}%)** as "
         f"low-quality ({dropped_flag:,} dropped, {kept_flag:,} kept-but-flagged).")
    return BlockResult(markdown=s + "\n\n" + df_to_md(tbl))


# --------------------------------------------------------------------------- #
# 6. RETENTION ACCURACY (ground-truth, two report cases)
# --------------------------------------------------------------------------- #

@block("retention_keepbest", "Case 1 — keep-best within duplicates", "retention_accuracy",
       "table", needs_gt=True,
       desc="Among audio clusters with a clean+low-quality contrast, is the kept cut clean?")
def retention_keepbest(rep: PipelineReport) -> BlockResult:
    gt = rep.ground_truth
    if not gt:
        return _na("no ground_truth.jsonl")
    a = rep.assignments[rep.assignments["cut_id"].isin(gt)]

    def is_clean(cid):
        me = gt[cid].get("mos_eval")
        return (me["family"] == "clean_control") if me else \
            gt[cid].get("dup_type") in ("original", "exact")

    def is_lowq(cid):
        me = gt[cid].get("mos_eval")
        return bool(me and me.get("expect_low_quality"))

    clusters = {}
    for r in a.itertuples():
        if r.audio_cluster_id is not None and r.audio_cluster_id >= 0:
            clusters.setdefault(r.audio_cluster_id, []).append((r.cut_id, r.is_kept))
    contrast = kept_clean = kept_lowq = 0
    for members in clusters.values():
        cids = [c for c, _ in members]
        if not (any(is_clean(c) for c in cids) and any(is_lowq(c) for c in cids)):
            continue
        contrast += 1
        kept = [c for c, k in members if k]
        if kept:
            kept_clean += is_clean(kept[0]); kept_lowq += is_lowq(kept[0])
    if not contrast:
        return BlockResult(markdown="No clusters with a clean+low-quality contrast.")
    tbl = pd.DataFrame([
        ["contrastive clusters", f"{contrast}"],
        ["kept cut is CLEAN", f"{kept_clean}/{contrast} = {100*kept_clean/contrast:.1f}%"],
        ["kept cut is LOW-QUALITY", f"{kept_lowq}/{contrast} = {100*kept_lowq/contrast:.1f}%"],
    ], columns=["metric", "value"])
    s = (f"In duplicate clusters with a quality contrast, retention keeps the "
         f"**clean** copy **{100*kept_clean/contrast:.1f}%** of the time "
         f"({kept_clean}/{contrast}).")
    return BlockResult(markdown=s + "\n\n" + df_to_md(tbl))


@block("retention_gate_prf", "Case 2 — quality-gate precision/recall/F1",
       "retention_accuracy", "table", needs_gt=True,
       desc="On labelled unique cuts: does low_quality match expect_low_quality (+per-family).")
def retention_gate_prf(rep: PipelineReport) -> BlockResult:
    gt = rep.ground_truth
    if not gt:
        return _na("no ground_truth.jsonl")
    a = rep.assignments[rep.assignments["cut_id"].isin(gt)]

    def is_lowq(cid):
        me = gt[cid].get("mos_eval")
        return bool(me and me.get("expect_low_quality"))

    def prf(pop):
        tp = sum(1 for r in pop if is_lowq(r.cut_id) and r.low_quality)
        fn = sum(1 for r in pop if is_lowq(r.cut_id) and not r.low_quality)
        fp = sum(1 for r in pop if not is_lowq(r.cut_id) and r.low_quality)
        tn = sum(1 for r in pop if not is_lowq(r.cut_id) and not r.low_quality)
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f = 2 * p * r / (p + r) if p + r else float("nan")
        return p, r, f, (tp, fp, fn, tn)

    uniq = [r for r in a.itertuples()
            if not r.is_duplicate and gt[r.cut_id].get("mos_eval")]
    allm = [r for r in a.itertuples() if gt[r.cut_id].get("mos_eval")]
    if not allm:
        return _na("no labelled (mos_eval) cuts — gate eval needs the MOS block")
    clean_orig = [r for r in a.itertuples() if gt[r.cut_id].get("dup_type") == "original"]
    fpr = (sum(1 for r in clean_orig if r.low_quality) / len(clean_orig)
           if clean_orig else float("nan"))
    rows = []
    for name, pop in (("UNIQUE (acoustically-alone)", uniq), ("ALL MOS cuts", allm)):
        if not pop:
            continue
        p, r, f, (tp, fpc, fn, tn) = prf(pop)
        rows.append([name, f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", f"{tp}/{fpc}/{fn}/{tn}"])
    df = pd.DataFrame(rows, columns=["population", "precision", "recall", "F1", "TP/FP/FN/TN"])
    up, ur, uf, _ = prf(uniq) if uniq else prf(allm)
    s = (f"On the canonical **unique** population the gate scores **F1={uf:.3f}** "
         f"(P={up:.3f}, R={ur:.3f}); false-positive rate on clean originals "
         f"**{100*fpr:.1f}%** (target ≤ gate_percentile).")
    return BlockResult(markdown=s + "\n\n" + df_to_md(df))


# --------------------------------------------------------------------------- #
# 7. RETENTION OUTCOME (label-free)
# --------------------------------------------------------------------------- #

@block("removed_by_reason", "Removed cuts by reason", "retention_outcome", "figure+table",
       desc="Breakdown of why cuts were dropped (retention_reason).")
def removed_by_reason(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    drop = a[~a["is_kept"]]
    if drop.empty:
        return BlockResult(markdown="No cuts removed.")
    g = drop.groupby("retention_reason").agg(
        cuts=("cut_id", "size"),
        hours=("duration_secs", lambda s: s.sum() / 3600.0)).reset_index()
    g = g.sort_values("cuts", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(g["retention_reason"], g["cuts"], color="#c44e52")
    ax.set_xlabel("cuts removed"); ax.set_title("Removed-by-reason")
    ax.invert_yaxis(); fig.tight_layout()
    g["hours"] = g["hours"].round(1)
    return BlockResult(markdown=df_to_md(g), fig=fig)


@block("kept_vs_removed", "Kept vs removed (cuts & hours)", "retention_outcome", "figure",
       desc="Side-by-side kept/removed totals in cuts and hours.")
def kept_vs_removed(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    kept = a["is_kept"]
    data = {
        "cuts": [int(kept.sum()), int((~kept).sum())],
        "hours": [_hours(a, kept), _hours(a, ~kept)],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (k, v) in zip(axes, data.items()):
        ax.bar(["kept", "removed"], v, color=["#3b6ea5", "#c44e52"])
        ax.set_title(k)
    fig.tight_layout()
    return BlockResult(markdown=f"Kept **{data['hours'][0]:.0f} h** vs removed "
                       f"**{data['hours'][1]:.0f} h**.", fig=fig)


# --------------------------------------------------------------------------- #
# 8. ASR / ROVER / ENHANCED TRANSCRIPT
# --------------------------------------------------------------------------- #

def _wer(ref, hyp):
    rw = norm_words(ref)
    return (_word_ed(rw, norm_words(hyp)) / len(rw)) if rw else np.nan


@block("asr_model_wer", "ASR model WER + consensus + enhanced", "asr", "figure+table",
       desc="Per-model WER vs the original transcript, plus rover & enhanced consensus.")
def asr_model_wer(rep: PipelineReport) -> BlockResult:
    r = rep.rover
    if r is None:
        return _na("rover/merged.jsonl missing")
    hyp_cols = [c for c in r.columns if c.startswith("hyp_")]
    series = {c.replace("hyp_", ""): c for c in hyp_cols}
    if "rover_text" in r.columns:
        series["consensus"] = "rover_text"
    if "text_enhanced" in r.columns:
        series["enhanced"] = "text_enhanced"
    rows = []
    for name, col in series.items():
        wers = [_wer(ref, hyp) for ref, hyp in zip(r["ref_text"], r[col])]
        wers = [w for w in wers if not np.isnan(w)]
        rows.append([name, 100 * np.mean(wers)])
    df = pd.DataFrame(rows, columns=["system", "WER_%"]).sort_values("WER_%")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["system"], df["WER_%"], color="#4c72b0")
    ax.set_ylabel("mean WER (%)"); ax.set_title("WER vs original transcript")
    ax.tick_params(axis="x", rotation=30); fig.tight_layout()
    best = df.iloc[0]
    return BlockResult(markdown=f"Lowest WER: **{best['system']}** "
                       f"({best['WER_%']:.2f}%).\n\n" + df_to_md(df), fig=fig)


@block("enhanced_change", "Enhanced-transcript change rate", "asr", "figure",
       desc="How much the ref-anchored enhanced transcript changes the original.")
def enhanced_change(rep: PipelineReport) -> BlockResult:
    r = rep.rover
    if r is None or "text_enhanced" not in r.columns:
        return _na("no text_enhanced in rover data")
    ed = ref_w = real = casing = same = 0
    for ref, enh in zip(r["ref_text"], r["text_enhanced"]):
        rw, ew = norm_words(ref), norm_words(enh)
        ref_w += len(rw); ed += _word_ed(rw, ew)
        if rw != ew:
            real += 1
        elif (ref or "").strip() != (enh or "").strip():
            casing += 1
        else:
            same += 1
    n = len(r)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["real word change", "casing/punct only", "identical"],
           [100*real/n, 100*casing/n, 100*same/n], color=["#c44e52", "#dd8452", "#55a868"])
    ax.set_ylabel("% of cuts"); ax.set_title("Enhanced vs original transcript")
    fig.tight_layout()
    s = (f"The enhanced transcript changes a real word in **{100*real/n:.1f}%** of cuts "
         f"(**{100*ed/max(ref_w,1):.2f}%** of words, edit-distance) and adds "
         f"casing/punctuation only in **{100*casing/n:.1f}%**.")
    return BlockResult(markdown=s, fig=fig)


@block("asr_examples", "Enhanced-transcript examples", "asr", "table",
       desc="A few before/after rows: original transcript -> ROVER-enhanced.")
def asr_examples(rep: PipelineReport) -> BlockResult:
    r = rep.rover
    if r is None or "text_enhanced" not in r.columns:
        return _na("no text_enhanced in rover data")
    rows = []
    for ref, enh in zip(r["ref_text"], r["text_enhanced"]):
        rw, ew = norm_words(ref), norm_words(enh)
        if rw != ew and 0 < _word_ed(rw, ew) <= 3 and rw:
            rows.append([(ref or "")[:90], (enh or "")[:90]])
        if len(rows) >= 5:
            break
    if not rows:
        return BlockResult(markdown="No small-edit examples found.")
    df = pd.DataFrame(rows, columns=["original", "enhanced"])
    return BlockResult(markdown=df_to_md(df))


# --------------------------------------------------------------------------- #
# 9. MULTILINGUAL
# --------------------------------------------------------------------------- #

@block("per_language_overview", "Per-language overview", "multilingual", "figure+table",
       multilingual=True,
       desc="Cuts, hours, dedup-removal % and low-quality % per dataset/language.")
def per_language_overview(rep: PipelineReport) -> BlockResult:
    a = rep.assignments
    g = a.groupby("dataset").apply(lambda d: pd.Series({
        "cuts": len(d),
        "hours": d["duration_secs"].sum() / 3600.0,
        "removed_%": 100 * (~d["is_kept"]).mean(),
        "low_quality_%": 100 * d["low_quality"].mean(),
    }), include_groups=False).reset_index()
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(g) + 3), 4))
    x = np.arange(len(g)); w = 0.4
    ax.bar(x - w/2, g["removed_%"], w, label="removed %", color="#c44e52")
    ax.bar(x + w/2, g["low_quality_%"], w, label="low-quality %", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(g["dataset"], rotation=45, ha="right")
    ax.set_ylabel("%"); ax.legend(); ax.set_title("Per-language dedup & quality")
    fig.tight_layout()
    for c in ("hours", "removed_%", "low_quality_%"):
        g[c] = g[c].round(1)
    return BlockResult(markdown=df_to_md(g), fig=fig)


@block("per_language_asr_change", "Per-language enhanced-change rate", "multilingual",
       "figure", multilingual=True,
       desc="Enhanced-transcript word-change rate per dataset/language.")
def per_language_asr_change(rep: PipelineReport) -> BlockResult:
    r = rep.rover
    if r is None or "text_enhanced" not in r.columns:
        return _na("no text_enhanced in rover data")
    key = "dataset" if r["dataset"].notna().any() else "language_hint"
    out = []
    for g, d in r.groupby(key):
        ed = rw = 0
        for ref, enh in zip(d["ref_text"], d["text_enhanced"]):
            w = norm_words(ref); rw += len(w); ed += _word_ed(w, norm_words(enh))
        out.append([g, 100 * ed / max(rw, 1)])
    df = pd.DataFrame(out, columns=[key, "word_change_%"]).sort_values("word_change_%")
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(df) + 3), 4))
    ax.bar(df[key].astype(str), df["word_change_%"], color="#8172b3")
    ax.set_ylabel("word change (%)"); ax.tick_params(axis="x", rotation=45)
    ax.set_title("Enhanced word-change per language"); fig.tight_layout()
    return BlockResult(markdown=df_to_md(df.round(2)), fig=fig)


@block("language_consistency", "Language-consistency check", "multilingual",
       "sentence",
       desc="Share of cuts where all ASRs agree on the detected language (rover).")
def language_consistency(rep: PipelineReport) -> BlockResult:
    r = rep.rover
    if r is None or "lang_consistent" not in r.columns:
        return _na("no language_consistency in rover data")
    lc = r["lang_consistent"].dropna()
    if lc.empty:
        return _na("language_consistency unpopulated")
    s = (f"ASRs agree on the spoken language in **{100*lc.mean():.1f}%** of cuts "
         f"(n={len(lc):,}); the remainder are candidate language-mismatch / "
         f"code-switch cases worth review in a multilingual corpus.")
    return BlockResult(markdown=s)


# --------------------------------------------------------------------------- #
# 10. METHODOLOGY + LIMITATIONS (narrative)
# --------------------------------------------------------------------------- #

# family -> (category, what it simulates, what it should make the pipeline do)
_FAMILY_GLOSSARY = {
    "original": ("base", "An unmodified source cut.",
                 "Unique unless a genuine duplicate exists in the corpus."),
    "exact": ("dedup +", "Verbatim audio + text copy of a source cut.",
              "Must join its source's TEXT and AUDIO cluster."),
    "text_slight_change": ("dedup +", "Same audio, 1-2 words edited in the transcript.",
                           "Must AUDIO-cluster with its source; text is only a near-match."),
    "degraded_quality": ("dedup +", "Realistic re-encode / resample / light noise (a cross-upload).",
                         "Must AUDIO-cluster (near-dup) with its source; text identical."),
    "same_text_diff_speaker": ("dedup -", "Pitch/tempo-shifted into a different 'voice', same words.",
                               "Must TEXT-cluster but must NOT audio-merge (different speaker)."),
    "partial_crop": ("dedup +", "Contiguous 40-70% crop of the source audio.",
                     "Must AUDIO-cluster with its source from the overlapping span."),
    "wrong_pairing": ("dedup +", "Speaker A's audio paired with speaker B's transcript.",
                      "Audio-cluster with A, text-cluster with B (cross-keyed)."),
    "hard_negative": ("dedup -", "Voice-shift + heavy text edit; derived from nothing.",
                      "Must NOT merge into any text or audio cluster (false-merge trap)."),
    "clean_control": ("control", "Pristine (FLAC round-trip only).",
                      "Quality gate must NOT flag it (false-positive check)."),
    "awgn": ("quality", "Synthetic white noise across an SNR sweep.",
             "Quality gate should flag at low SNR (DNSMOS axis)."),
    "real_noise": ("quality", "Real MUSAN noise across an SNR sweep.",
                   "Quality gate should flag at low SNR (DNSMOS)."),
    "music": ("quality", "MUSAN music mixed in across an SNR sweep.",
              "Quality gate should flag (AudioBox PC/CU)."),
    "reverb": ("quality", "Real room impulse response (RIR).",
               "Quality gate should flag heavy reverb (UTMOS)."),
    "crosstalk": ("quality", "A second speaker mixed in across an SIR sweep.",
                  "Quality gate should flag (STOI / SI-SDR)."),
    "clipping": ("quality", "Hard clipping across a threshold sweep.",
                 "Quality gate should flag (DNSMOS)."),
    "telephony": ("quality", "8 kHz band-limit + mu-law companding.",
                  "Quality gate should flag band-limited audio (SI-SDR / DNSMOS)."),
    "naturalness": ("quality", "Griffin-Lim resynthesis (clean SNR, unnatural timbre).",
                    "Quality gate should flag low naturalness (UTMOS) despite a clean SNR."),
    "wer": ("asr", "Clean audio with the REFERENCE corrupted to a known WER.",
            "ASR branch should flag a high WER vs its consensus."),
    "silence": ("asr", "Injected lead/trail/gap silence (or full silence).",
                "VAD / hallucination-on-silence handling."),
    "lang": ("asr", "Correct audio + text but a WRONG language tag.",
             "Language-consistency / routing check."),
}


@block("dataset_construction", "How the test dataset was built", "methodology",
       "sentence", desc="Methodology: the synthetic eval blocks and their provenance labels.")
def dataset_construction(rep: PipelineReport) -> BlockResult:
    prose = (
        "**How we created the test dataset.** The evaluation set is built by "
        "`make_eval_dataset.py` from a prepared LibriSpeech Shar. It **keeps every "
        "original cut** and appends synthetic cuts whose ground-truth provenance "
        "(source cut, edit type, and the expected match) is stored in "
        "`cut.custom[\"dedup_eval\"] / mos_eval / asr_eval` and mirrored in "
        "`ground_truth.jsonl`. Every operation is seeded from `(seed, cut_id)`, so "
        "the build is fully reproducible. Four blocks each exercise a different "
        "part of the pipeline:\n\n"
        "1. **Dedup block (+30%)** — copies of originals altered in *known* ways "
        "(exact, text_slight_change, degraded_quality, same_text_diff_speaker, "
        "partial_crop, wrong_pairing) plus hard-negatives, to test text + audio "
        "duplicate detection.\n"
        "2. **MOS-quality block** — degraded copies of distinct clean originals "
        "across 9 degradation families at sampled severities, to test the quality "
        "gate per acoustic axis (with a `clean_control` positive control).\n"
        "3. **Standalone low-quality block** — severely degraded *and* text-altered "
        "cuts with no twin, to test the unique-sample quality gate (drop path).\n"
        "4. **ASR block** — corrupted references / injected silence / wrong language "
        "tags, to test the ASR-ROVER branch.\n\n"
        "Because the edits are synthetic, the exact expected outcome of every cut "
        "is known, which is what makes per-family recall / precision measurable.")
    gt = rep.ground_truth
    if not gt:
        return BlockResult(markdown=prose)
    n = len(gt)
    syn = sum(1 for g in gt.values() if g.get("is_synthetic"))
    n_mos = sum(1 for g in gt.values() if g.get("mos_eval"))
    n_asr = sum(1 for g in gt.values() if g.get("asr_eval"))
    add = (f"\n\nIn this build: **{n:,} labelled cuts** — {n-syn:,} originals + "
           f"{syn:,} synthetic ({n_mos:,} carry a MOS-quality label, {n_asr:,} an "
           f"ASR label).")
    return BlockResult(markdown=prose + add)


@block("family_glossary", "Per-family glossary — what each flag means", "methodology",
       "table", desc="Each ground-truth family: what it simulates and what the pipeline should do.")
def family_glossary(rep: PipelineReport) -> BlockResult:
    gt = rep.ground_truth
    present, counts = sorted(_FAMILY_GLOSSARY), {}
    if gt:
        for cid in gt:
            f = _gt_family(gt, cid)
            counts[f] = counts.get(f, 0) + 1
        present = sorted(set(counts) | set(_FAMILY_GLOSSARY),
                         key=lambda f: (-counts.get(f, 0), f))
    rows = []
    for f in present:
        cat, what, expect = _FAMILY_GLOSSARY.get(f, ("?", "—", "—"))
        row = [f, cat, what, expect]
        if gt:
            row.insert(1, f"{counts.get(f, 0):,}")
        rows.append(row)
    cols = (["family", "count", "category", "simulates", "expected behavior"] if gt
            else ["family", "category", "simulates", "expected behavior"])
    intro = ("Each synthetic cut belongs to a **family** that targets one pipeline "
             "behavior. `dedup +` = should be merged with its source; `dedup -` = a "
             "trap that must NOT merge; `quality` = a degradation the gate should "
             "flag; `control` = clean (must not be flagged); `asr` = an ASR-branch "
             "probe.\n")
    return BlockResult(markdown=intro + "\n" + df_to_md(pd.DataFrame(rows, columns=cols)))


@block("method_limitations", "Limitations of the current method", "limitations",
       "sentence", desc="Where the current dedup/quality/ASR method is weakest (live numbers).")
def method_limitations(rep: PipelineReport) -> BlockResult:
    parts = ["**Limitations of the current method.**\n"]
    if rep.has_gt:
        aby, _ = _recall_by_family(rep, "expect_audio_dup_of", "audio_cluster_id")
        tby, _ = _recall_by_family(rep, "expect_text_dup_of", "text_cluster_id")

        def rec(by, f):
            v = by.get(f)
            return (100 * v[0] / v[1]) if v and v[1] else float("nan")

        parts.append(
            "- **Audio dedup is gated by text dedup.** Stage E only compares "
            "fingerprints *within* a text cluster, so a cut that fails to text-cluster "
            "can never audio-dedup — even with byte-identical audio. This is the "
            f"dominant failure mode: `text_slight_change` (same audio, 1-2 words "
            f"edited) gets only {rec(aby,'text_slight_change'):.0f}% audio recall — the "
            f"same as its {rec(tby,'text_slight_change'):.0f}% text recall — and "
            f"`wrong_pairing` (A's audio under B's transcript) gets "
            f"{rec(aby,'wrong_pairing'):.0f}%, because it lands in B's text cluster and "
            "its audio is never compared against A. (For reference, `exact` — identical "
            f"audio that DOES text-cluster — reaches {rec(aby,'exact'):.0f}%.)")
        parts.append(
            f"- **Fingerprint robustness.** Even inside the correct text cluster the "
            f"acoustic match degrades on `reverb` ({rec(aby,'reverb'):.0f}%) — room "
            f"impulse responses smear the constellation peaks — and `partial_crop` "
            f"({rec(aby,'partial_crop'):.0f}%) — a global energy-normalised peak gate "
            "plus a min-keypoints floor loses anchors on short fragments (a crop-"
            "invariant local gate was tried and deferred).")
        parts.append(
            "- **Text-dedup threshold (the upstream cause).** A 1-2 word edit drops a "
            "pair below the MinHash/LSH band threshold. The recall lever is LSH band "
            "geometry, not the Jaccard threshold — and because audio is gated by text, "
            "fixing this also recovers the cascaded audio misses above (e.g. dedup on "
            "a reference-independent ASR key, which is identical for identical audio).")
    else:
        parts.append(
            "- **Audio dedup is gated by text dedup** — audio is compared only within "
            "text clusters, so cuts that fail text clustering can't audio-dedup even "
            "with identical audio. The fingerprint is also weak on reverberation and "
            "partial crops (peak smearing / lost keypoints).")
    parts.append(
        "- **Quality gate** is *percentile-relative* — it flags the bottom "
        "`gate_percentile`% by a combined z-score, so the flag set shifts with the "
        "corpus mix rather than an absolute MOS floor. No single MOS metric covers "
        "every degradation (telephony/band-limit is under-flagged), which is why a "
        "`mean(z)+min(z)` combiner is used; it is still a proxy, not human MOS.")
    parts.append(
        "- **Enhanced transcript** is reference-anchored: on a heavily-corrupted "
        "original the correction is partial (it preserves the original's structure "
        "where alignment fails), and on ALL-CAPS references the casing is inherited "
        "unevenly. The LLM-ITN refinement that normalises this is currently OFF.")
    return BlockResult(markdown="\n".join(parts))


@block("coverage_gaps", "Out of scope — what we don't cover", "limitations",
       "sentence", desc="Capabilities deliberately or not-yet covered by this evaluation.")
def coverage_gaps(rep: PipelineReport) -> BlockResult:
    multi = rep.is_multilingual
    lang_line = ("- **Languages beyond this run.** "
                 + ("Multiple datasets/languages are present, but per-language dedup "
                    "and quality behaviour is only as good as each language's ASR/MOS "
                    "models." if multi else
                    "Only **English** is evaluated here; multilingual dedup, quality "
                    "and ASR are the next step and are not yet measured."))
    parts = [
        "**What we don't cover (out of scope / not yet measured).**\n",
        lang_line,
        "- **Semantic / paraphrase duplicates.** We detect near-exact text and "
        "acoustic near-duplicates only — paraphrases, translations, or the same "
        "content spoken with different words are NOT flagged as duplicates.",
        "- **Real-world distribution shift.** Degradations are *synthetic* "
        "approximations of real re-uploads, from finite MUSAN/RIR pools; genuine "
        "web-scraped duplicates and codecs in the wild may behave differently.",
        "- **No human MOS.** Quality 'ground truth' is degradation-family proxy "
        "labels, not listener ratings, so absolute MOS calibration is unverified.",
        "- **Scale.** Validated at ~190k cuts / ~658 h; large-scale behaviour "
        "(multi-GPU connected-components, memory, very large clusters which are "
        "currently capped) is not yet stress-tested.",
        "- **Number / ITN formatting** is not exercised by audio (LibriSpeech "
        "carries almost no digits) — it is covered separately by a unit test.",
        "- **Speaker identity / PII, copyright, and toxicity** filtering are out of "
        "scope for this pipeline.",
    ]
    return BlockResult(markdown="\n".join(parts))


BLOCKS = {b.id: b for b in CATALOG}
