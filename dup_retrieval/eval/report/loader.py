"""PipelineReport — file-driven loader for the dedup+quality pipeline outputs.

Point it at ONE path (a run dir, a config yaml, run_manifest.json, or
retention/assignments.parquet) and it lazily resolves + loads every
intermediate artifact a report block might need:

    manifest, assignments, quality (numeric metrics), text/audio clusters,
    rover (ASR transcripts incl. text_enhanced), and ground_truth (optional).

Report blocks (blocks.py) consume these as pandas DataFrames; the driver
(build.py) turns each block into a figure / table / sentence.  Nothing here
knows about plotting — this is pure I/O + light shaping.
"""
from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import orjson as _orjson
    def _loads(b):
        return _orjson.loads(b)
except Exception:                              # pragma: no cover
    def _loads(b):
        return json.loads(b)


# --------------------------------------------------------------------------- #
# small shared text/format helpers (used by blocks too)
# --------------------------------------------------------------------------- #

_PUNC = re.compile(r"[^a-z0-9'\s]")


def norm_words(s: Optional[str]) -> List[str]:
    """Lowercase + strip punctuation -> word list (for case/punct-insensitive WER)."""
    return _PUNC.sub(" ", (s or "").lower()).split()


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-markdown table with no external deps."""
    cols = list(df.columns)
    head = (["", *cols] if index else cols)
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join("---" for _ in head) + " |"]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + [fmt(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #

class PipelineReport:
    """Lazy accessor for one pipeline run's artifacts."""

    def __init__(self, source, ground_truth: Optional[str] = None,
                 rover: Optional[str] = None, rover_limit: Optional[int] = None):
        self.run_dir = self._resolve_run_dir(Path(source))
        self.rover_limit = rover_limit
        self._gt_path = (Path(ground_truth) if ground_truth
                         else self._find_ground_truth())
        self._rover_path = Path(rover) if rover else self._find_rover()

    # ---- path resolution ------------------------------------------------- #

    @staticmethod
    def _resolve_run_dir(p: Path) -> Path:
        if p.is_dir():
            if (p / "retention").is_dir() or (p / "run_manifest.json").is_file():
                return p
            if (p / "assignments.parquet").is_file():       # given retention/
                return p.parent
            return p
        if p.suffix in (".yaml", ".yml"):
            import yaml
            cfg = yaml.safe_load(p.read_text())
            return Path(cfg["output_dir"])
        if p.name == "run_manifest.json":
            return p.parent
        if p.suffix == ".parquet":                          # .../retention/assignments.parquet
            return p.parent.parent
        raise ValueError(f"Cannot resolve a run dir from: {p}")

    def _find_ground_truth(self) -> Optional[Path]:
        """ground_truth.jsonl lives with the synthetic dataset, not in the run
        dir.  Search the run dir's siblings one level up (cheap, bounded)."""
        for cand in sorted(self.run_dir.parent.glob("*/ground_truth.jsonl")):
            return cand
        gt = self.run_dir / "ground_truth.jsonl"
        return gt if gt.is_file() else None

    def _find_rover(self) -> Optional[Path]:
        """rover/merged.jsonl is written under the quality_search_paths, which
        are usually a SIBLING of the run dir (not inside it).  Check the run dir
        first, then the run dir's siblings one level up."""
        inside = self._p("quality", "quality_asr", "rover", "merged.jsonl")
        if inside.is_file():
            return inside
        for cand in sorted(self.run_dir.parent.glob("*/quality_asr/rover/merged.jsonl")):
            return cand
        return None

    def _p(self, *parts) -> Path:
        return self.run_dir.joinpath(*parts)

    # ---- artifacts (lazy) ------------------------------------------------ #

    @cached_property
    def manifest(self) -> dict:
        f = self._p("run_manifest.json")
        return json.loads(f.read_text()) if f.is_file() else {}

    @cached_property
    def assignments(self) -> pd.DataFrame:
        return pd.read_parquet(self._p("retention", "assignments.parquet"))

    @cached_property
    def quality(self) -> Optional[pd.DataFrame]:
        f = self._p("quality", "merged.parquet")
        return pd.read_parquet(f) if f.is_file() else None

    @cached_property
    def text_clusters(self) -> Optional[pd.DataFrame]:
        f = self._p("text_dedup", "clusters.parquet")
        return pd.read_parquet(f) if f.is_file() else None

    @cached_property
    def audio_clusters(self) -> Optional[pd.DataFrame]:
        f = self._p("audio_match", "clusters.parquet")
        return pd.read_parquet(f) if f.is_file() else None

    @cached_property
    def rover(self) -> Optional[pd.DataFrame]:
        """Parse rover/merged.jsonl into a flat DataFrame (drops word_timestamps).

        Columns: cut_id, dataset, language_hint, ref_text, rover_text,
        text_enhanced, text_enhanced_itn, filtered_reason, lang_consistent,
        n_ambiguous, primary_fallbacks, hyp_<slot>.
        """
        f = self._rover_path
        if not f or not f.is_file():
            return None
        rows = []
        with open(f, "rb") as fh:
            for i, line in enumerate(fh):
                if self.rover_limit and i >= self.rover_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                d = _loads(line)
                rv = d.get("rover") or {}
                lc = d.get("language_consistency") or {}
                hyps = d.get("hypotheses") or {}
                row = {
                    "cut_id": d.get("cut_id"),
                    "dataset": d.get("dataset"),
                    "language_hint": d.get("language_hint"),
                    "ref_text": d.get("ref_text"),
                    "rover_text": rv.get("text"),
                    "text_enhanced": rv.get("text_enhanced"),
                    "text_enhanced_itn": rv.get("text_enhanced_itn"),
                    "filtered_reason": d.get("filtered_reason"),
                    "lang_consistent": lc.get("all_consistent"),
                    "n_ambiguous": len(rv.get("ambiguous_words_enhanced") or []),
                    "primary_fallbacks": rv.get("primary_fallbacks"),
                }
                for slot, h in hyps.items():
                    row[f"hyp_{slot}"] = (h or {}).get("text")
                rows.append(row)
        return pd.DataFrame(rows)

    @cached_property
    def ground_truth(self) -> Optional[Dict[str, dict]]:
        if not self._gt_path or not self._gt_path.is_file():
            return None
        gt = {}
        with open(self._gt_path, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = _loads(line)
                gt[d["cut_id"]] = d
        return gt

    # ---- convenience ----------------------------------------------------- #

    @property
    def has_gt(self) -> bool:
        return self.ground_truth is not None

    @property
    def has_rover(self) -> bool:
        return self._rover_path is not None and self._rover_path.is_file()

    @property
    def datasets(self) -> List[str]:
        return sorted(self.assignments["dataset"].dropna().unique().tolist())

    @property
    def is_multilingual(self) -> bool:
        return len(self.datasets) > 1

    def describe(self) -> str:
        a = self.assignments
        hrs = a["duration_secs"].sum() / 3600.0
        return (f"run_dir={self.run_dir}\n"
                f"cuts={len(a):,}  hours={hrs:.1f}  datasets={self.datasets}\n"
                f"ground_truth={'yes' if self.has_gt else 'no'}  "
                f"rover={'yes' if self.has_rover else 'no'}")
