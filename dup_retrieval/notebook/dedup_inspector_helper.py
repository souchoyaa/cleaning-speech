"""Notebook helper: inspect dedup clusters with audio + transcripts.

Mirrors ``mos_inspector_helper.py`` style.  Workflow in a Jupyter cell:

    from audio_tokenization.utils.data_selection.dup_retrieval.notebook \
        import dedup_inspector_helper as dih

    paths = dih.paths_for_run(output_dir="/scratch/dedup_run")
    df    = dih.load_final(paths)              # all cuts + dedup decisions
    cuts  = dih.build_cut_lookup(paths, shar_dir="/path/to/shar")
    dih.show_cluster(df, cuts, cluster_id=42)  # play audio for each member
    dih.show_random_clusters(df, cuts, n=20)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def paths_for_run(output_dir: str) -> dict:
    out = Path(output_dir)
    return {
        "output_dir":   out,
        "manifest":     out / "manifest" / "part_*.parquet",
        "text_clusters": out / "text_dedup" / "clusters.parquet",
        "fingerprints": out / "fingerprint" / "part_*.parquet",
        "audio_clusters": out / "audio_match" / "clusters.parquet",
        "final_dedup":  out / "retention" / "assignments.parquet",
        "huge_clusters": out / "audio_match" / "huge_clusters.parquet",
    }


def load_final(paths: dict) -> pd.DataFrame:
    """Read ``retention/assignments.parquet`` as a DataFrame."""
    t = pq.read_table(paths["final_dedup"])
    return t.to_pandas()


def load_huge_clusters(paths: dict) -> Optional[pd.DataFrame]:
    p = paths["huge_clusters"]
    if not p.exists():
        return None
    return pq.read_table(p).to_pandas()


def build_cut_lookup(paths: dict, shar_dir: str,
                     target_sample_rate: int = 16000) -> Dict[Tuple[str, str], "lhotse.Cut"]:
    """Return ``{(dataset, cut_id): Cut}`` for the shar.

    Loads the entire shar via ``CutSet.from_shar`` (lhotse handles streaming).
    """
    from audio_tokenization.pipelines.lhotse.data import build_cutset  # type: ignore

    cfg = {"shar_dir": shar_dir, "mode": "audio_only",
           "target_sample_rate": target_sample_rate}
    cuts = build_cutset(cfg, rank=0, world_size=1)

    from audio_tokenization.utils.data_selection.dup_retrieval.core.manifest \
        import _derive_dataset_name  # type: ignore
    ds_tag = _derive_dataset_name(shar_dir)

    out: Dict[Tuple[str, str], object] = {}
    for cut in cuts:
        cid = cut.id[2:] if cut.id.startswith("./") else cut.id
        out[(ds_tag, cid)] = cut
    return out


def show_cluster(df: pd.DataFrame,
                 cut_lookup: Dict[Tuple[str, str], object],
                 cluster_id: int,
                 max_members: int = 10) -> None:
    """Display every cut in an audio_cluster with its text + audio player."""
    from IPython.display import Audio, Markdown, display

    rows = df[df["audio_cluster_id"] == cluster_id].head(max_members)
    if rows.empty:
        display(Markdown(f"**No cluster {cluster_id} in assignments.parquet.**"))
        return

    display(Markdown(f"## Cluster {cluster_id} (size = {len(rows)} shown / "
                     f"{int(rows['cluster_size'].iloc[0])} total)"))
    for _, r in rows.iterrows():
        cut = cut_lookup.get((r["dataset"], r["cut_id"]))
        text = ""
        if cut is not None and cut.supervisions:
            text = cut.supervisions[0].text or ""
        kept = "**KEPT**" if r["is_kept"] else "dropped"
        line = (f"- *{r['dataset']}* `{r['cut_id']}` {kept} — "
                f"q={r['quality_score']:.3f} | reason={r['retention_reason']} | "
                f"dur={r['duration_secs']:.1f}s | text: \"{text[:140]}\"")
        display(Markdown(line))
        if cut is not None:
            try:
                audio = cut.load_audio()
                if audio.ndim > 1:
                    audio = audio[0]
                display(Audio(audio, rate=cut.sampling_rate))
            except Exception as e:
                display(Markdown(f"  *(audio load failed: {e})*"))


def show_random_clusters(df: pd.DataFrame,
                         cut_lookup: Dict[Tuple[str, str], object],
                         n: int = 20,
                         min_size: int = 2,
                         seed: int = 0) -> None:
    """Pick ``n`` random non-singleton clusters and show each."""
    rng = random.Random(seed)
    multi = df[df["cluster_size"] >= max(min_size, 2)]
    cl_ids = multi["audio_cluster_id"].dropna().unique().tolist()
    rng.shuffle(cl_ids)
    for cid in cl_ids[:n]:
        show_cluster(df, cut_lookup, int(cid))


def cluster_size_histogram(df: pd.DataFrame, max_size: int = 30):
    """Return (sizes, counts) for a histogram of cluster sizes (>1)."""
    multi = df[df["cluster_size"] > 1]
    sizes = multi.groupby("audio_cluster_id")["cluster_size"].first().values
    sizes = sizes.clip(max=max_size)
    return np.unique(sizes, return_counts=True)


def retention_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset summary: total / kept / dropped / cluster sizes."""
    summary = []
    for ds, sub in df.groupby("dataset"):
        n_total = len(sub)
        n_kept  = int(sub["is_kept"].sum())
        n_drop  = n_total - n_kept
        n_in_clusters = int((sub["cluster_size"] > 1).sum())
        n_clusters    = int(sub.loc[sub["cluster_size"] > 1, "audio_cluster_id"].nunique())
        summary.append({
            "dataset":       ds,
            "n_total":       n_total,
            "n_kept":        n_kept,
            "n_dropped":     n_drop,
            "n_in_clusters": n_in_clusters,
            "n_clusters":    n_clusters,
            "drop_rate":     round(n_drop / max(n_total, 1), 4),
        })
    return pd.DataFrame(summary).sort_values("dataset").reset_index(drop=True)
