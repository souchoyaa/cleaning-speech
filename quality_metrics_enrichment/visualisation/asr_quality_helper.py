"""Helpers for ASR quality visualisation.

Loads ASR fused output rows joined with SHAR ``cut.custom`` into a single
``pandas.DataFrame``, then provides per-language plots:

* split distribution (validated / other / invalidated)
* WER / CER for ROVER vs ``ref_text``
* upvote / downvote pair distribution (auto-detects the column names in
  ``cut.custom`` so it works whatever your SHAR pipeline named them)

Only stdlib + pandas + matplotlib + jiwer (the last three already in
``~/.venv-tools`` on clariden after the install run earlier in this
repo's setup).

Typical use from a notebook
---------------------------
    from asr_quality_helper import (
        load_asr_dataframe, wer_cer_per_language,
        plot_split_distribution, plot_vote_distribution, plot_wer_cer,
    )

    df = load_asr_dataframe(asr_root, shar_root, workers=8)
    plot_split_distribution(df)
    stats = wer_cer_per_language(df)
    plot_wer_cer(stats)
    plot_vote_distribution(df)
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Same shadow-fix as our other tools: this module sits next to a
# stdlib-shadowing ``logging.py`` in some sibling dirs; defensively scrub
# our own dir before stdlib imports it transitively.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SELF_DIR]

import pandas as pd

SHAR_INDEX_FILENAME = "shar_index.json"

# Stable colors + display order so the same split always renders the same way
# across plots. Both naming conventions (validation / validated) are mapped so
# the helper stays robust to whichever the SHAR uses.
SPLIT_COLORS = {
    "validation":  "#2ca02c",   # green  — human-verified good
    "validated":   "#2ca02c",
    "other":       "#1f77b4",   # blue   — unannotated / pending
    "invalidated": "#d62728",   # red    — human-verified bad
}
_SPLIT_RANK = {"validation": 0, "validated": 0, "other": 1, "invalidated": 2}


def _sorted_splits(splits) -> list[str]:
    """Stable sort: validated → other → invalidated → unknown (alphabetical)."""
    known = sorted([s for s in splits if s in _SPLIT_RANK], key=_SPLIT_RANK.get)
    unknown = sorted([s for s in splits if s not in _SPLIT_RANK])
    return known + unknown


def _color_for(split: str) -> str:
    return SPLIT_COLORS.get(split, "#7f7f7f")  # neutral gray fallback


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------


def _find_asr_split_dirs(root: Path) -> list[Path]:
    out: list[Path] = []

    def walk(d: Path) -> None:
        try:
            children = list(d.iterdir())
        except (PermissionError, OSError):
            return
        if any(c.is_file() and c.name.startswith("fused_rank_")
               and c.name.endswith(".jsonl") for c in children):
            out.append(d)
            return
        for c in sorted(children):
            if c.is_dir():
                walk(c)

    walk(root)
    return out


def _build_shar_lookup(shar_dir: Path) -> dict[str, dict]:
    """cut_id -> {'ref_text': str, 'custom': dict}.

    Reads every ``cuts.*.jsonl.gz`` once (no audio decode). ``custom`` is
    the raw cut.custom dict from the shar; downstream code can look for
    upvote/downvote keys in there.
    """
    idx_path = shar_dir / SHAR_INDEX_FILENAME
    with open(idx_path) as f:
        cuts_rel = json.load(f).get("fields", {}).get("cuts", [])
    out: dict[str, dict] = {}
    for rel in cuts_rel:
        full = shar_dir / rel
        with gzip.open(full, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = d.get("id")
                if not cid:
                    continue
                sup0 = (d.get("supervisions") or [{}])[0]
                out.setdefault(cid, {
                    "ref_text": (sup0.get("text") or ""),
                    "custom": d.get("custom") or {},
                })
    return out


def _process_split(args: tuple) -> list[dict]:
    """Per-split worker: build shar lookup, stream all rank files."""
    asr_split_s, shar_dir_s, lang, split = args
    asr_split = Path(asr_split_s)
    shar_dir = Path(shar_dir_s)

    lookup = _build_shar_lookup(shar_dir)
    rows: list[dict] = []
    for jsonl in sorted(asr_split.glob("fused_rank_*.jsonl")):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = r.get("cut_id") or ""
                extra = lookup.get(cid, {})
                custom = extra.get("custom", {})
                row = {
                    "lang": lang,
                    "split": split,
                    "cut_id": cid,
                    "duration": r.get("duration"),
                    # Prefer ASR's ref_text; fall back to shar supervision.
                    "ref_text": (r.get("ref_text") or "").strip()
                                or extra.get("ref_text", "").strip(),
                    "rover_text": ((r.get("rover") or {}).get("text") or ""),
                    "filtered_reason": r.get("filtered_reason"),
                    "primary_present": ((r.get("rover") or {})
                                        .get("primary_present")),
                }
                # Fold custom fields verbatim (rms_db, source_id,
                # up_votes/down_votes if present, etc.).
                for k, v in custom.items():
                    if k not in row:
                        row[k] = v
                rows.append(row)
    return rows


def load_asr_dataframe(
    asr_root: str | Path,
    shar_root: str | Path,
    workers: int = 1,
    lang_filter: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Walk ``asr_root``, join with ``shar_root`` per split, return one
    DataFrame with one row per ASR cut.

    ``lang_filter`` (e.g. ``["de","fr"]``) restricts to the listed langs.
    """
    asr_root = Path(asr_root).resolve()
    shar_root = Path(shar_root).resolve()
    splits = _find_asr_split_dirs(asr_root)
    if not splits:
        raise FileNotFoundError(f"no ASR splits under {asr_root}")

    jobs: list[tuple] = []
    for asr_split in splits:
        rel = asr_split.relative_to(asr_root)
        parts = rel.parts
        if len(parts) < 2:
            continue  # expect <lang>/<split>
        lang, split = parts[0], parts[1]
        if lang_filter and lang not in set(lang_filter):
            continue
        shar_dir = shar_root / rel
        if not (shar_dir / SHAR_INDEX_FILENAME).is_file():
            print(f"[skip] no shar at {shar_dir}", file=sys.stderr)
            continue
        jobs.append((str(asr_split), str(shar_dir), lang, split))

    rows: list[dict] = []
    if workers <= 1:
        for j in jobs:
            rows.extend(_process_split(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_split, j) for j in jobs]
            for fut in as_completed(futs):
                rows.extend(fut.result())

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Predictable column order for the common ones.
    front = [c for c in ["lang", "split", "cut_id", "duration",
                          "ref_text", "rover_text", "filtered_reason",
                          "primary_present"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------


def wer_cer_per_language(
    df: pd.DataFrame,
    hyp_col: str = "rover_text",
    ref_col: str = "ref_text",
    exclude_filtered: bool = True,
) -> pd.DataFrame:
    """Per-language WER and CER for the given hypothesis column.

    Drops rows where ref or hyp is empty (jiwer raises on those). When
    ``exclude_filtered`` is True (default), also drops rows with a
    non-null ``filtered_reason``.
    """
    if df.empty:
        return pd.DataFrame(columns=["lang", "n_rows", "n_skipped", "wer", "cer"])

    import jiwer

    sub = df.copy()
    sub[ref_col] = sub[ref_col].fillna("").astype(str).str.strip()
    sub[hyp_col] = sub[hyp_col].fillna("").astype(str)

    mask = (sub[ref_col] != "") & (sub[hyp_col] != "")
    if exclude_filtered and "filtered_reason" in sub.columns:
        mask &= sub["filtered_reason"].isna() | (sub["filtered_reason"] == None)  # noqa: E711
    sub_use = sub[mask]
    skipped = (~mask).groupby(sub["lang"]).sum()

    out = []
    for lang, g in sub_use.groupby("lang"):
        if g.empty:
            out.append({"lang": lang, "n_rows": 0,
                        "n_skipped": int(skipped.get(lang, 0)),
                        "wer": None, "cer": None})
            continue
        refs = list(g[ref_col])
        hyps = list(g[hyp_col])
        try:
            wer = float(jiwer.wer(refs, hyps))
        except Exception:
            wer = None
        try:
            cer = float(jiwer.cer(refs, hyps))
        except Exception:
            cer = None
        out.append({"lang": lang, "n_rows": int(len(g)),
                    "n_skipped": int(skipped.get(lang, 0)),
                    "wer": wer, "cer": cer})
    res = pd.DataFrame(out).sort_values("lang").reset_index(drop=True)
    return res


# ---------------------------------------------------------------------------
# Vote-column auto-detection
# ---------------------------------------------------------------------------


def detect_vote_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Best-guess up_votes / down_votes column names from cut.custom keys."""
    up = down = None
    for c in df.columns:
        cl = c.lower().replace("-", "_")
        if up is None and "up" in cl and "vote" in cl:
            up = c
        if down is None and "down" in cl and "vote" in cl:
            down = c
    return up, down


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_split_distribution(df: pd.DataFrame, ax=None, figsize=(14, 6)):
    """Stacked bar: cuts per (language, split). Validated=green, other=blue,
    invalidated=red so the human-quality axis reads visually."""
    import matplotlib.pyplot as plt
    if df.empty:
        print("empty dataframe — nothing to plot")
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    pivot = df.groupby(["lang", "split"]).size().unstack(fill_value=0)
    splits = _sorted_splits(pivot.columns)
    pivot = pivot[splits].sort_index()
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[_color_for(s) for s in splits])
    ax.set_ylabel("# cuts")
    ax.set_xlabel("language")
    ax.set_title("ASR rows per language × split")
    ax.legend(title="split", loc="upper right")
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    return ax


def plot_wer_cer(stats: pd.DataFrame, axes=None):
    """Two side-by-side bars: WER and CER per language."""
    import matplotlib.pyplot as plt
    if stats is None or stats.empty or stats["wer"].dropna().empty:
        print("no WER/CER values to plot (no usable refs?)")
        return None
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    s = stats.dropna(subset=["wer"]).copy()
    s.plot.bar(x="lang", y="wer", ax=axes[0], legend=False, color="steelblue")
    axes[0].set_title("WER per language (rover)")
    axes[0].set_ylabel("WER")
    s.plot.bar(x="lang", y="cer", ax=axes[1], legend=False, color="darkorange")
    axes[1].set_title("CER per language (rover)")
    axes[1].set_ylabel("CER")
    for ax in axes:
        ax.set_xlabel("language")
        for label in ax.get_xticklabels():
            label.set_rotation(0)
    return axes


def _wer_cer_pair(pair):
    """Module-level so ProcessPoolExecutor can pickle it."""
    import jiwer  # local import keeps base helper light
    r, h = pair
    try:
        return jiwer.wer(r, h), jiwer.cer(r, h)
    except Exception:
        return float("nan"), float("nan")


def add_per_cut_wer_cer(
    df: pd.DataFrame,
    hyp_col: str = "rover_text",
    ref_col: str = "ref_text",
    exclude_filtered: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    """Return ``df`` with extra ``wer`` and ``cer`` float columns.

    Rows where ref or hyp is empty (or the row was filtered, when
    ``exclude_filtered=True``) get NaN. Per-row jiwer is microseconds —
    49K rows ≈ 5s, 2M rows ≈ a few minutes. Pass ``workers > 1`` to
    parallelize across processes.
    """
    if df.empty:
        out = df.copy()
        out["wer"] = pd.Series(dtype="float64")
        out["cer"] = pd.Series(dtype="float64")
        return out

    import jiwer

    out = df.copy()
    out[ref_col] = out[ref_col].fillna("").astype(str).str.strip()
    out[hyp_col] = out[hyp_col].fillna("").astype(str)
    eligible = (out[ref_col] != "") & (out[hyp_col] != "")
    if exclude_filtered and "filtered_reason" in out.columns:
        eligible &= out["filtered_reason"].isna()

    refs = out.loc[eligible, ref_col].tolist()
    hyps = out.loc[eligible, hyp_col].tolist()

    if workers <= 1:
        scores = [_wer_cer_pair(p) for p in zip(refs, hyps)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            scores = list(ex.map(_wer_cer_pair, list(zip(refs, hyps)), chunksize=1024))

    wer_arr = pd.Series(float("nan"), index=out.index)
    cer_arr = pd.Series(float("nan"), index=out.index)
    if scores:
        idxs = out.index[eligible]
        wer_arr.loc[idxs] = [s[0] for s in scores]
        cer_arr.loc[idxs] = [s[1] for s in scores]
    out["wer"] = wer_arr
    out["cer"] = cer_arr
    return out


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------


def plot_metric_distribution_by_split(
    df: pd.DataFrame,
    metric: str = "wer",
    bins: int = 50,
    clip_max: float | None = 1.5,
    cols: int = 3,
    panel_w: float = 6.0,
    panel_h: float = 4.0,
):
    """Per language: overlapping histograms of ``metric`` colored by split.

    Splits are rendered in stable order (validated → other → invalidated)
    using SPLIT_COLORS so the same color always means the same human label.
    ``clip_max`` caps the x-axis (WER can exceed 1.0 — long right tails
    drown the rest of the distribution if you don't clip).
    """
    import matplotlib.pyplot as plt
    if metric not in df.columns:
        print(f"column {metric!r} not in dataframe — call add_per_cut_wer_cer() first")
        return None
    sub = df[df[metric].notna()].copy()
    if sub.empty:
        print(f"no non-NaN values in column {metric!r}")
        return None
    if clip_max is not None:
        sub = sub[sub[metric] <= clip_max]

    langs = sorted(sub["lang"].unique())
    n = len(langs); ncols = min(cols, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_w * ncols, panel_h * nrows),
                              sharex=True, squeeze=False)
    splits = _sorted_splits(sub["split"].unique())

    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        gl = sub[sub["lang"] == lang]
        for s in splits:
            vals = gl.loc[gl["split"] == s, metric].values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.55, label=f"{s} (n={len(vals):,})",
                    color=_color_for(s), density=True, edgecolor="white",
                    linewidth=0.3)
        ax.set_title(f"{lang}  (total n={len(gl):,})")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("density")
        ax.legend(fontsize="small", loc="upper right")
        ax.grid(True, axis="y", alpha=0.25)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{metric.upper()} distribution per language, by split",
                 y=1.00, fontsize=14)
    plt.tight_layout()
    return axes


def plot_metric_vs_votes(
    df: pd.DataFrame,
    metric: str = "wer",
    vote_col: str | None = None,
    cols: int = 3,
    clip_max: float | None = 1.5,
    max_vote_value: int = 6,
    panel_w: float = 6.0,
    panel_h: float = 4.0,
):
    """Per language: boxplot of ``metric`` per integer vote bucket.

    ``vote_col`` defaults to the auto-detected ``up_votes`` (or
    ``down_votes``); pass an explicit column to switch. Vote values
    above ``max_vote_value`` are bucketed into a single "≥N" group so a
    rare 25-upvote outlier doesn't blow out the x-axis.
    """
    import matplotlib.pyplot as plt
    if metric not in df.columns:
        print(f"column {metric!r} not in dataframe — call add_per_cut_wer_cer() first")
        return None
    if vote_col is None:
        up, down = detect_vote_columns(df)
        vote_col = up or down
    if vote_col is None or vote_col not in df.columns:
        print("no vote column found")
        return None

    sub = df[df[metric].notna()].copy()
    if clip_max is not None:
        sub = sub[sub[metric] <= clip_max]
    sub[vote_col] = pd.to_numeric(sub[vote_col], errors="coerce")
    sub = sub.dropna(subset=[vote_col])
    if sub.empty:
        print(f"no rows after filtering for {metric!r} + {vote_col!r}")
        return None
    sub[vote_col] = sub[vote_col].astype(int)
    sub["_bucket"] = sub[vote_col].clip(upper=max_vote_value)

    langs = sorted(sub["lang"].unique())
    n = len(langs); ncols = min(cols, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_w * ncols, panel_h * nrows),
                              sharey=True, squeeze=False)
    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        gl = sub[sub["lang"] == lang]
        buckets = sorted(gl["_bucket"].unique())
        data = [gl.loc[gl["_bucket"] == b, metric].values for b in buckets]
        ax.boxplot(data, labels=[
            (f"≥{max_vote_value}" if b >= max_vote_value else str(b))
            for b in buckets
        ], showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="#aec7e8", alpha=0.8),
            medianprops=dict(color="black", linewidth=1.4))
        for j, vals in enumerate(data, start=1):
            ax.text(j, ax.get_ylim()[1] * 0.97, f"n={len(vals):,}",
                    ha="center", va="top", fontsize=8, color="dimgray")
        ax.set_title(f"{lang}", fontsize=12)
        ax.set_xlabel(vote_col)
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.25)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{metric.upper()} vs {vote_col} per language",
                 y=1.00, fontsize=14)
    plt.tight_layout()
    return axes


def plot_vote_distribution(df: pd.DataFrame, ax=None,
                           up_col: str | None = None,
                           down_col: str | None = None):
    """Distribution of (up, down) vote pairs per language.

    If column names not given, auto-detect via ``detect_vote_columns``.
    Returns ``None`` (with a printed message) if no vote columns are
    available — useful when running before SHAR has been re-prepared
    with vote fields.
    """
    import matplotlib.pyplot as plt
    if df.empty:
        print("empty dataframe — nothing to plot")
        return None
    if up_col is None or down_col is None:
        d_up, d_down = detect_vote_columns(df)
        up_col = up_col or d_up
        down_col = down_col or d_down
    if not (up_col and down_col):
        print("no upvote/downvote columns found in cut.custom; "
              "available custom-derived columns: "
              + ", ".join(c for c in df.columns
                          if c not in {"lang","split","cut_id","duration",
                                       "ref_text","rover_text",
                                       "filtered_reason","primary_present"}))
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    sub = df.copy()
    sub[up_col] = pd.to_numeric(sub[up_col], errors="coerce").fillna(0).astype(int)
    sub[down_col] = pd.to_numeric(sub[down_col], errors="coerce").fillna(0).astype(int)
    sub["pair"] = sub[up_col].astype(str) + "↑/" + sub[down_col].astype(str) + "↓"
    counts = (sub.groupby(["lang", "pair"]).size()
              .unstack(fill_value=0).sort_index())
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("# cuts")
    ax.set_xlabel("language")
    ax.set_title(f"Vote-pair distribution per language "
                 f"({up_col} / {down_col})")
    ax.legend(title="↑/↓ pair", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize="small")
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    return ax


# ---------------------------------------------------------------------------
# Treating the (up, down) vote pair as the unit of human judgment
# ---------------------------------------------------------------------------


def add_vote_features(
    df: pd.DataFrame,
    up_col: str | None = None,
    down_col: str | None = None,
) -> pd.DataFrame:
    """Add ``net_score = up - down``, ``vote_total = up + down``, and
    ``vote_ratio = up / total`` (NaN when total == 0).

    These let plots and correlations consume the *pair* (which is what
    actually encodes human judgment) instead of either coordinate alone.
    """
    if up_col is None or down_col is None:
        u, d = detect_vote_columns(df)
        up_col = up_col or u; down_col = down_col or d
    if not (up_col and down_col and up_col in df.columns and down_col in df.columns):
        return df
    out = df.copy()
    out[up_col] = pd.to_numeric(out[up_col], errors="coerce")
    out[down_col] = pd.to_numeric(out[down_col], errors="coerce")
    import numpy as np
    out["net_score"] = out[up_col] - out[down_col]
    out["vote_total"] = out[up_col] + out[down_col]
    # Avoid division by zero — produce NaN where vote_total == 0.
    ratio = out[up_col] / out["vote_total"].where(out["vote_total"] != 0)
    out["vote_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    return out


def plot_metric_heatmap_by_vote_pair(
    df: pd.DataFrame,
    metric: str = "wer",
    up_col: str | None = None,
    down_col: str | None = None,
    max_v: int = 6,
    min_cell: int = 5,
    cols: int = 3,
    panel_w: float = 5.5,
    panel_h: float = 4.5,
    annotate: bool = True,
):
    """Per language: 2D heatmap of mean ``metric`` for each (up, down) pair.

    This is the "pair as a unit" view the user asked for — instead of
    plotting WER vs up_votes alone (or down_votes alone), it shows mean
    WER across the joint integer grid. Cells with fewer than
    ``min_cell`` samples are masked (gray) so noisy single-cut cells
    don't confuse the picture. Counts are shown below the value when
    ``annotate=True``.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if metric not in df.columns:
        print(f"column {metric!r} not in df — call add_per_cut_wer_cer() first")
        return None
    if up_col is None or down_col is None:
        u, d = detect_vote_columns(df)
        up_col = up_col or u; down_col = down_col or d
    if not (up_col and down_col):
        print("no vote columns detected")
        return None

    sub = df[df[metric].notna()].copy()
    sub[up_col] = pd.to_numeric(sub[up_col], errors="coerce")
    sub[down_col] = pd.to_numeric(sub[down_col], errors="coerce")
    sub = sub.dropna(subset=[up_col, down_col])
    sub["_u"] = sub[up_col].clip(upper=max_v).astype(int)
    sub["_d"] = sub[down_col].clip(upper=max_v).astype(int)

    langs = sorted(sub["lang"].unique())
    n = len(langs); ncols = min(cols, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_w * ncols, panel_h * nrows),
                              squeeze=False)
    # Shared color scale across panels for cross-language comparison.
    vmin = sub[metric].quantile(0.05)
    vmax = sub[metric].quantile(0.95)

    last_im = None
    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        gl = sub[sub["lang"] == lang]
        mean = gl.groupby(["_d", "_u"])[metric].mean().unstack()
        count = gl.groupby(["_d", "_u"]).size().unstack(fill_value=0)
        mean = mean.reindex(index=range(max_v + 1), columns=range(max_v + 1))
        count = count.reindex(index=range(max_v + 1), columns=range(max_v + 1),
                              fill_value=0)
        masked = np.where(count.values < min_cell, np.nan, mean.values)
        im = ax.imshow(masked, origin="lower", aspect="equal",
                       cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        last_im = im
        ax.set_title(f"{lang}  (mean {metric.upper()})")
        ax.set_xlabel(f"{up_col} (clipped at {max_v})")
        ax.set_ylabel(f"{down_col} (clipped at {max_v})")
        ax.set_xticks(range(max_v + 1))
        ax.set_yticks(range(max_v + 1))
        ax.set_xticklabels([str(v) if v < max_v else f"≥{max_v}"
                             for v in range(max_v + 1)], fontsize=8)
        ax.set_yticklabels([str(v) if v < max_v else f"≥{max_v}"
                             for v in range(max_v + 1)], fontsize=8)
        if annotate:
            for di in range(max_v + 1):
                for ui in range(max_v + 1):
                    c = int(count.values[di, ui])
                    if c < min_cell:
                        continue
                    val = mean.values[di, ui]
                    ax.text(ui, di, f"{val:.2f}\nn={c:,}",
                            ha="center", va="center", fontsize=7,
                            color="black" if (vmin + vmax) / 2 > val
                            else "white")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                            shrink=0.7, pad=0.02)
        cbar.set_label(f"mean {metric.upper()} (5–95th pct color scale)")
    fig.suptitle(
        f"{metric.upper()} over the (up, down) vote pair, per language\n"
        f"cells with < {min_cell} samples are masked",
        y=1.01, fontsize=14,
    )
    return axes


def plot_metric_vs_net_score(
    df: pd.DataFrame,
    metric: str = "wer",
    cols: int = 3,
    clip: int = 5,
    panel_w: float = 6.0,
    panel_h: float = 4.0,
):
    """Per language: boxplot of ``metric`` per ``net_score = up - down``
    bucket. Net score is the simplest scalar that combines the pair into
    a single quality axis. Buckets outside ``[-clip, clip]`` collapse
    into the edge buckets so rare extremes don't dominate."""
    import matplotlib.pyplot as plt
    if metric not in df.columns:
        print(f"column {metric!r} not in df — call add_per_cut_wer_cer() first")
        return None
    if "net_score" not in df.columns:
        df = add_vote_features(df)
        if "net_score" not in df.columns:
            print("could not derive net_score — vote columns missing")
            return None

    sub = df[df[metric].notna() & df["net_score"].notna()].copy()
    sub["_b"] = sub["net_score"].clip(lower=-clip, upper=clip).astype(int)

    langs = sorted(sub["lang"].unique())
    n = len(langs); ncols = min(cols, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_w * ncols, panel_h * nrows),
                              sharey=True, squeeze=False)
    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        gl = sub[sub["lang"] == lang]
        buckets = sorted(gl["_b"].unique())
        data = [gl.loc[gl["_b"] == b, metric].values for b in buckets]
        labels = [
            (f"≤{-clip}" if b == -clip else
             f"≥{clip}" if b == clip else str(b))
            for b in buckets
        ]
        ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#c5b0d5", alpha=0.8),
                   medianprops=dict(color="black", linewidth=1.4))
        for j, vals in enumerate(data, start=1):
            ax.text(j, ax.get_ylim()[1] * 0.97, f"n={len(vals):,}",
                    ha="center", va="top", fontsize=8, color="dimgray")
        ax.set_title(f"{lang}", fontsize=12)
        ax.set_xlabel("net_score = up - down")
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.25)
        if 0 in buckets:
            ax.axvline(buckets.index(0) + 1, color="gray",
                       linestyle=":", linewidth=0.8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{metric.upper()} vs net_score (up - down) per language",
                 y=1.00, fontsize=14)
    plt.tight_layout()
    return axes


def compute_quality_correlations(
    df: pd.DataFrame,
    metric: str = "wer",
    up_col: str | None = None,
    down_col: str | None = None,
) -> pd.DataFrame:
    """Per language, Spearman + Pearson of ``metric`` against vote-derived
    quality signals (``up_votes``, ``-down_votes``, ``net_score``,
    ``vote_ratio``). All correlations are negated where appropriate so
    the "expected" sign is positive — a positive value means "higher
    quality signal ↔ lower WER".

    Spearman is the right metric here: votes are integer counts and WER
    is bounded in [0, ∞), so neither is normally distributed.
    """
    if metric not in df.columns:
        print(f"column {metric!r} not in df")
        return pd.DataFrame()
    work = df.copy()
    if "net_score" not in work.columns or "vote_ratio" not in work.columns:
        work = add_vote_features(work, up_col=up_col, down_col=down_col)
    if up_col is None or down_col is None:
        up_col, down_col = detect_vote_columns(work)

    rows = []
    for lang, g in work.groupby("lang"):
        g = g.dropna(subset=[metric])
        n = len(g)
        if n == 0:
            continue
        rec = {"lang": lang, "n": n}
        for sig_name, series in [
            (up_col,            g.get(up_col)),
            (down_col,          g.get(down_col)),
            ("net_score",       g.get("net_score")),
            ("vote_ratio",      g.get("vote_ratio")),
        ]:
            if series is None:
                rec[f"sp_{sig_name}"] = None
                continue
            s = pd.to_numeric(series, errors="coerce")
            mask = s.notna()
            if mask.sum() < 5:
                rec[f"sp_{sig_name}"] = None
                continue
            corr = g.loc[mask, metric].corr(s[mask], method="spearman")
            if pd.isna(corr):
                rec[f"sp_{sig_name}"] = None
                continue
            # Sign convention: higher quality should correlate with lower WER,
            # so flip sign on `up_*`, `net_score`, `vote_ratio`.
            if sig_name in (up_col, "net_score", "vote_ratio"):
                corr = -corr
            rec[f"sp_{sig_name}"] = float(corr)
        rows.append(rec)
    res = pd.DataFrame(rows).sort_values("lang").reset_index(drop=True)
    return res


def plot_quality_roc(
    df: pd.DataFrame,
    metric: str = "wer",
    positive_split: str = "validation",
    negative_split: str = "invalidated",
    cols: int = 3,
    panel_w: float = 5.5,
    panel_h: float = 5.0,
):
    """Per language, ROC of ``-metric`` discriminating ``positive_split``
    vs ``negative_split``. AUC summarizes how well WER alone recovers
    the human good/bad label.

    Why this matters for the user's stated goal: once they apply WER on
    a dataset that *doesn't* have human labels, they can use the AUC
    here as a confidence score for "WER is a useful quality proxy" —
    AUC=0.5 means useless, AUC=1.0 means perfect.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if metric not in df.columns:
        print(f"column {metric!r} not in df — call add_per_cut_wer_cer() first")
        return None

    accept = {positive_split, negative_split, "validated"}  # alias
    sub = df[df[metric].notna() & df["split"].isin(accept)].copy()
    if sub.empty:
        print(f"no rows with split in {sorted(accept)}")
        return None
    pos_set = {positive_split, "validated"} if positive_split == "validation" else {positive_split}
    sub["_y"] = sub["split"].isin(pos_set).astype(int)

    langs = sorted(sub["lang"].unique())
    aucs = {}
    n = len(langs); ncols = min(cols, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_w * ncols, panel_h * nrows),
                              squeeze=False)
    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        gl = sub[sub["lang"] == lang]
        if gl["_y"].nunique() < 2:
            ax.set_title(f"{lang} — only one class present")
            ax.axis("off"); continue
        scores = -gl[metric].values  # higher score = predicted positive
        labels = gl["_y"].values
        fpr, tpr, auc = _roc_and_auc(labels, scores)
        aucs[lang] = auc
        ax.plot(fpr, tpr, color="steelblue", linewidth=2,
                label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
        ax.fill_between(fpr, tpr, alpha=0.15, color="steelblue")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{lang}  ({len(gl):,} cuts: "
                     f"{int(gl['_y'].sum()):,} positive)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.25)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        f"ROC of -{metric.upper()} discriminating "
        f"split={positive_split!r} (positive) vs split={negative_split!r}",
        y=1.00, fontsize=13,
    )
    plt.tight_layout()
    return aucs, axes


def _roc_and_auc(y_true, scores):
    """Plain-numpy ROC: returns (fpr, tpr, auc). Avoids sklearn dep."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]
    P = int(y.sum())
    N = int(len(y) - P)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    # Collapse threshold ties
    distinct = np.where(np.diff(s))[0]
    end = np.r_[distinct, len(y) - 1]
    tps = tps[end]
    fps = fps[end]
    tpr = np.r_[0, tps / P]
    fpr = np.r_[0, fps / N]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def plot_quality_threshold_curves(
    df: pd.DataFrame,
    metric: str = "wer",
    positive_split: str = "validation",
    negative_split: str = "invalidated",
    cols: int = 3,
    panel_w: float = 6.0,
    panel_h: float = 4.0,
    n_thresholds: int = 200,
):
    """Per language, plot precision and recall of "validated" as a
    function of the WER threshold. Tells you operationally: "if I keep
    only cuts with WER < T, what fraction of those are validated, and
    what fraction of validated cuts do I keep?"
    """
    import matplotlib.pyplot as plt
    import numpy as np

    accept = {positive_split, negative_split, "validated"}
    sub = df[df[metric].notna() & df["split"].isin(accept)].copy()
    if sub.empty:
        print(f"no rows with split in {sorted(accept)}")
        return None
    pos_set = {positive_split, "validated"} if positive_split == "validation" else {positive_split}
    sub["_y"] = sub["split"].isin(pos_set).astype(int)

    langs = sorted(sub["lang"].unique())
    n = len(langs); ncols = min(cols, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_w * ncols, panel_h * nrows),
                              squeeze=False, sharex=True)
    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        gl = sub[sub["lang"] == lang]
        if gl["_y"].nunique() < 2:
            ax.set_title(f"{lang} — only one class present")
            ax.axis("off"); continue
        m_lo = float(gl[metric].quantile(0.005))
        m_hi = float(gl[metric].quantile(0.995))
        thresholds = np.linspace(m_lo, m_hi, n_thresholds)
        prec, rec, kept = [], [], []
        P = int(gl["_y"].sum())
        total = len(gl)
        for t in thresholds:
            keep = gl[metric] <= t
            n_keep = int(keep.sum())
            tp = int(((gl["_y"] == 1) & keep).sum())
            prec.append(tp / n_keep if n_keep else float("nan"))
            rec.append(tp / P if P else float("nan"))
            kept.append(n_keep / total)
        ax.plot(thresholds, prec, label="precision (P(validated | WER≤t))",
                color="#2ca02c", linewidth=2)
        ax.plot(thresholds, rec, label="recall (frac validated kept)",
                color="#1f77b4", linewidth=2)
        ax.plot(thresholds, kept, label="frac of all cuts kept",
                color="#7f7f7f", linewidth=1.2, linestyle="--")
        ax.set_xlabel(f"{metric.upper()} threshold (keep if ≤ t)")
        ax.set_ylabel("rate")
        ax.set_title(f"{lang}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize="small")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        f"{metric.upper()} threshold operating points "
        f"(positive={positive_split!r}, negative={negative_split!r})",
        y=1.00, fontsize=13,
    )
    plt.tight_layout()
    return axes


# ---------------------------------------------------------------------------
# Audio inspector slider (mirrors mos_inspector_helper.score_slider_inspector)
# ---------------------------------------------------------------------------


def build_cut_lookup(
    shar_root: str | Path,
    keep_ids: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    target_sample_rate: int = 16000,
    cache_path: str | Path | None = None,
):
    """Build cut_id -> Lhotse Cut for one SHAR (single- or multi-language).

    Mirrors ``quality_assesment/notebook/mos_inspector_helper.py:build_cut_lookup``
    but trimmed for the ASR notebook. Loads via ``build_cutset`` from the
    project's lhotse pipeline package, which handles single-shar and the
    nested ``<lang>/<split>/`` layout. Pass ``keep_ids`` (a set of
    cut_ids) to only retain cuts you care about — e.g. ``set(df["cut_id"])``.

    ``cache_path``: optional pickle path to memoize across kernel restarts.
    Delete the file to force a rebuild after the SHAR changes.
    """
    import pickle

    shar_root = Path(shar_root)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                lookup = pickle.load(fh)
            print(f"loaded cut lookup from cache: {cache_path} ({len(lookup):,} cuts)")
            return lookup

    import sys
    _PKG_ROOT = Path(__file__).resolve().parents[5]
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))
    from audio_tokenization.pipelines.lhotse.data import build_cutset

    keep = set(keep_ids) if keep_ids is not None else None

    def _load_one(d: Path) -> dict:
        cfg = {
            "shar_dir": str(d),
            "target_sample_rate": int(target_sample_rate),
            "mode": "audio_only",
        }
        out = {}
        for cut in build_cutset(cfg, rank=0, world_size=1):
            if keep is None or cut.id in keep:
                out[cut.id] = cut
        return out

    if (shar_root / SHAR_INDEX_FILENAME).is_file():
        lookup = _load_one(shar_root)
    else:
        # Walk every leaf shar dir (any subdir holding shar_index.json).
        leaves: list[Path] = []
        def walk(d: Path) -> None:
            if (d / SHAR_INDEX_FILENAME).is_file():
                leaves.append(d); return
            for c in sorted(d.iterdir()):
                if c.is_dir():
                    walk(c)
        walk(shar_root)
        if languages is not None:
            lang_set = set(languages)
            leaves = [d for d in leaves if d.relative_to(shar_root).parts[0] in lang_set]
        lookup = {}
        for d in leaves:
            lookup.update(_load_one(d))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(lookup, fh)
        print(f"cut lookup cached to: {cache_path} ({len(lookup):,} cuts)")
    return lookup


def metric_slider_inspector(
    df: pd.DataFrame,
    cut_lookup: dict,
    metric: str = "wer",
    n_samples: int = 3,
    n_bins: int = 50,
    max_seconds: float = 10.0,
    seed: int | None = None,
):
    """Slider over ``metric`` values; plays random samples from each bin.

    Bins are quantile-based (each holds ~equal sample count). Duplicate
    edges collapse, so heavy mass at one value (e.g. WER==0) becomes a
    single bin and the tail gets fine-grained resolution. Slider is an
    IntSlider over bin index since edges are irregular.

    ``df`` must already have a column named ``metric`` — call
    ``add_per_cut_wer_cer(df)`` first to populate ``wer`` and ``cer``.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import Audio, HTML, clear_output, display
    except ImportError:
        print("ipywidgets is required: ~/.venv-tools/bin/pip install ipywidgets")
        return

    import html as _html
    import numpy as np

    if metric not in df.columns:
        print(f"column {metric!r} not in df — run add_per_cut_wer_cer() first")
        return
    valid = df[df[metric].notna()].reset_index(drop=True).copy()
    if valid.empty:
        print(f"no non-NaN values for {metric!r}")
        return

    arr_m = valid[metric].to_numpy()
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(arr_m, quantiles)).astype(float)
    if edges.size < 2:
        edges = np.array([float(arr_m.min()), float(arr_m.min()) + 1e-9])
    edges[-1] = np.nextafter(edges[-1], np.inf)
    n_bins_actual = edges.size - 1

    out = widgets.Output()
    up_col, down_col = detect_vote_columns(valid)

    def _show_one(row, max_seconds: float = max_seconds):
        cid = row["cut_id"]
        parts = [
            f"<code style='color:#444'>{_html.escape(str(cid))}</code>",
            f"<b>{metric.upper()}</b>={row[metric]:.3f}",
            f"split=<b>{_html.escape(str(row['split']))}</b>",
            f"lang=<b>{_html.escape(str(row['lang']))}</b>",
            f"dur={row.get('duration', float('nan')):.2f}s",
        ]
        if up_col and pd.notna(row.get(up_col)):
            parts.append(f"↑={int(row[up_col])}")
        if down_col and pd.notna(row.get(down_col)):
            parts.append(f"↓={int(row[down_col])}")
        if row.get("filtered_reason"):
            parts.append(f"filtered={_html.escape(str(row['filtered_reason']))}")
        display(HTML(" &nbsp;|&nbsp; ".join(parts)))

        ref = _html.escape((row.get("ref_text") or "").strip()[:300])
        hyp = _html.escape((row.get("rover_text") or "").strip()[:300])
        display(HTML(
            "<ul style='margin:2px 0'>"
            f"<li><b>ref</b>: <span style='color:#111'>{ref}</span></li>"
            f"<li><b>rover</b>: <span style='color:#111'>{hyp}</span></li>"
            "</ul>"
        ))

        cut = cut_lookup.get(cid)
        if cut is None:
            display(HTML("<i>audio: cut not in cut_lookup — skipping</i>"))
            return
        try:
            audio = cut.load_audio()
            sr = int(getattr(cut, "sampling_rate"))
            data = np.asarray(audio)
            data = data[0] if data.ndim == 2 else np.squeeze(data)
            if max_seconds is not None and len(data) > int(max_seconds * sr):
                data = data[: int(max_seconds * sr)]
            display(Audio(data, rate=sr))
        except Exception as e:
            display(HTML(f"<i>audio load failed: {_html.escape(str(e))}</i>"))

    def _refresh(bin_idx: int) -> None:
        bin_idx = max(0, min(int(bin_idx), n_bins_actual - 1))
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        in_bin = valid[(valid[metric] >= lo) & (valid[metric] < hi)]
        if in_bin.empty:
            dists = (valid[metric] - lo).abs()
            in_bin = valid.loc[dists.nsmallest(n_samples).index]
        drawn = in_bin.sample(
            n=min(n_samples, len(in_bin)),
            random_state=seed,
            replace=False,
        )
        with out:
            clear_output(wait=True)
            display(HTML(
                f"<h4 style='margin:4px 0'>{metric.upper()} "
                f"bin {bin_idx + 1}/{n_bins_actual} &nbsp;"
                f"[{lo:.3f}, {hi:.3f}) &nbsp;— "
                f"{len(in_bin):,} samples in bin, showing {len(drawn)}</h4>"
            ))
            for _, row in drawn.iterrows():
                _show_one(row)
                display(HTML("<hr style='border:none;border-top:1px solid #ddd'>"))
            display(reload_btn)

    slider = widgets.IntSlider(
        value=n_bins_actual // 2,
        min=0,
        max=max(n_bins_actual - 1, 0),
        step=1,
        description=f"{metric.upper()} bin",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
        continuous_update=False,
    )
    reload_btn = widgets.Button(
        description="Reload",
        button_style="info",
        icon="refresh",
        layout=widgets.Layout(width="120px"),
        tooltip="Draw a new random sample from the current bin",
    )
    slider.observe(lambda chg: _refresh(int(chg["new"])), names="value")
    reload_btn.on_click(lambda _btn: _refresh(int(slider.value)))

    display(widgets.VBox([slider, out]))
    _refresh(slider.value)
