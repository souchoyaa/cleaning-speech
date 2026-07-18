"""Stage G — apply dedup decisions to a clone of the shar.

Default mode: **copy-with-symlinks**.  Writes a thin clone directory where
only ``cuts.*.jsonl.gz`` is materialised fresh (with the unified augmented
metadata injected — ``cut.custom["dedup"]`` + ``["quality"]`` + ``["keep"]`` +
``["asr_transcript"]`` (the ROVER-improved transcript + its rover_wer/rover_cer
vs the original), the diagram's
"metadata augmented + flag keep or not"); every audio / text /
custom-field tar is symlinked (or hardlinked) to the original.  Audio tars are
never written in any mode; the original shar is never touched in clone mode.

Optional ``--in-place`` / ``apply_to_shar.in_place: true`` rewrites the
original ``cuts.*.jsonl.gz`` directly (mirrors
``audio_tokenization/utils/prepare_data/postprocess/add_rms_to_shar.py``).

Inputs  : <output_dir>/final/assignments.parquet
          shar_dirs (originals)
Outputs : <output_shar_dir>/<each shar layout cloned>
          (or, if in_place: in-place modifications of the originals)
"""

import argparse
import gzip
import json
import logging
import os
import re
import shutil
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Tuple

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SHAR_INDEX_FILENAME = "shar_index.json"


def _link(src: Path, dst: Path, mode: str) -> None:
    """Create a link (or copy) from dst -> src per ``mode``.

    src must exist; dst must not exist.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(os.fspath(src), os.fspath(dst))
    elif mode == "hardlink":
        try:
            os.link(os.fspath(src), os.fspath(dst))
        except OSError as e:
            logger.warning("hardlink failed (%s) — falling back to symlink", e)
            os.symlink(os.fspath(src), os.fspath(dst))
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link_mode: {mode!r}")


def _atomic_write_gz_jsonl(out_path: Path, lines: list) -> None:
    """Write a list of dict rows to a gzipped JSONL atomically."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)


def _process_one_shard(args) -> Tuple[int, int, int]:
    """Read one cuts.X.jsonl.gz, inject cut.custom["dedup"] from the lookup,
    write to *out_path*."""
    (src_path, out_path, dataset, dedup_lookup, transcript_lookup,
     in_place_skip_unchanged) = args
    src_path = Path(src_path)
    out_path = Path(out_path)
    if not src_path.exists():
        logger.warning("Missing source shard: %s", src_path)
        return 0, 0, 0

    raw_dicts = []
    with gzip.open(str(src_path), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_dicts.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Bad JSON line in %s — skipping.", src_path)

    n_injected = 0
    n_singleton = 0
    n_unknown = 0

    for d in raw_dicts:
        raw_id = d.get("id") or ""
        cid = raw_id[2:] if raw_id.startswith("./") else raw_id
        info = dedup_lookup.get(cid)
        if info is None:
            n_unknown += 1
            continue
        custom = d.get("custom") or {}
        # Unified augmented metadata (the diagram's grey box): dedup decision +
        # quality + a single keep flag.  Audio/text bytes are untouched.
        custom["dedup"] = {
            "audio_cluster_id": info["audio_cluster_id"],
            "text_cluster_id":  info["text_cluster_id"],
            "cluster_size":     info["cluster_size"],
            "is_duplicate":     info["is_duplicate"],
        }
        quality = {
            "score":       info["quality_score"],
            "low_quality": info["low_quality"],
        }
        if info.get("metrics"):
            quality["metrics"] = info["metrics"]
        custom["quality"] = quality
        # Improved (ROVER-enhanced) ASR transcript — the original transcript
        # corrected by deterministic ASR consensus.  "source" records which
        # form was used (rover_enhanced_itn / rover_enhanced / rover_itn / rover).
        tr = transcript_lookup.get(cid)
        if tr and tr.get("text"):
            at = {"text": tr["text"], "source": tr["source"]}
            if tr.get("rover_wer") is not None:
                # WER/CER of the ROVER consensus vs the original transcript.
                at["rover_wer"] = round(tr["rover_wer"], 4)
                at["rover_cer"] = round(tr["rover_cer"], 4)
            custom["asr_transcript"] = at
        custom["keep"] = {
            "is_kept": info["is_kept"],
            "reason":  info["keep_reason"],       # kept / below_quality_floor / drop reason
            "detail":  info["retention_reason"],  # detailed dedup reason
        }
        d["custom"] = custom
        n_injected += 1
        if info["cluster_size"] == 1:
            n_singleton += 1

    if in_place_skip_unchanged and n_injected == 0:
        logger.info("Skipping %s — no rows touched.", src_path)
        return n_injected, n_singleton, n_unknown

    _atomic_write_gz_jsonl(out_path, raw_dicts)
    return n_injected, n_singleton, n_unknown


def _load_quality_metrics(quality_flat: Path,
                          dataset_filter: Optional[str] = None) -> Dict[str, dict]:
    """``{cut_id: {metric: val}}`` from quality/merged.parquet for a dataset.

    Returns ``{}`` if the file is absent (metrics are optional in the shar).
    """
    if not quality_flat or not Path(quality_flat).exists():
        return {}
    t = pq.read_table(quality_flat)
    metric_names = [n for n in t.column_names if n not in ("dataset", "cut_id")]
    ds  = t.column("dataset").to_pylist()
    cid = t.column("cut_id").to_pylist()
    cols = {m: t.column(m).to_pylist() for m in metric_names}
    out: Dict[str, dict] = {}
    for i in range(len(cid)):
        if dataset_filter is not None and ds[i] != dataset_filter:
            continue
        m = {k: cols[k][i] for k in metric_names if cols[k][i] is not None}
        if m:
            out[cid[i]] = m
    return out


def _build_lookup(dedup_parquet: Path, dataset_filter: Optional[str] = None,
                  quality_flat: Optional[Path] = None) -> Dict[str, dict]:
    """Build ``{cut_id: info}`` from final/assignments.parquet for a given dataset.

    Tolerates an older assignments.parquet without the selection columns
    (``is_duplicate``/``low_quality``/``keep_reason``) by deriving sensible
    defaults.  When ``quality_flat`` is given, raw per-cut metrics are attached
    under ``info["metrics"]``.  Each cut_id is unique within a dataset.
    """
    t = pq.read_table(dedup_parquet)
    names = set(t.column_names)

    def col(name):
        return t.column(name).to_pylist() if name in names else None

    ds  = t.column("dataset").to_pylist()
    cid = t.column("cut_id").to_pylist()
    aid = t.column("audio_cluster_id").to_pylist()
    tid = t.column("text_cluster_id").to_pylist()
    sz  = t.column("cluster_size").to_pylist()
    kept = t.column("is_kept").to_pylist()
    reason = t.column("retention_reason").to_pylist()
    q   = t.column("quality_score").to_pylist()
    is_dup_c = col("is_duplicate")
    low_q_c  = col("low_quality")
    keep_r_c = col("keep_reason")

    metrics = _load_quality_metrics(quality_flat, dataset_filter) if quality_flat else {}

    out: Dict[str, dict] = {}
    for i in range(len(cid)):
        if dataset_filter is not None and ds[i] != dataset_filter:
            continue
        c = cid[i]
        is_kept = bool(kept[i])
        ret = str(reason[i])
        is_dup = bool(is_dup_c[i]) if is_dup_c is not None else int(aid[i]) >= 0
        low_q = bool(low_q_c[i]) if low_q_c is not None else False
        keep_r = (str(keep_r_c[i]) if keep_r_c is not None
                  else ("kept" if is_kept else ret))
        out[c] = {
            "audio_cluster_id": int(aid[i]),
            "text_cluster_id":  int(tid[i]),
            "cluster_size":     int(sz[i]),
            "is_duplicate":     is_dup,
            "is_kept":          is_kept,
            "low_quality":      low_q,
            "keep_reason":      keep_r,
            "retention_reason": ret,
            "quality_score":    float(q[i]) if q[i] is not None else None,
            "metrics":          metrics.get(c),
        }
    return out


# --- WER / CER of the ROVER consensus vs the original transcript ------------- #
# Reference-INDEPENDENT signal: how far the 3-ASR consensus (rover.text) is from
# the cut's provided transcript.  On LibriSpeech (gold ref) this is the true ASR
# error rate; on a noisy source transcript it is the ASR-vs-source disagreement.
# Both sides are normalized (lowercase, strip punctuation) so casing/punctuation
# differences don't inflate the rate.
_ERR_PUNC = re.compile(r"[^a-z0-9'\s]")


def _err_norm(s):
    return _ERR_PUNC.sub(" ", (s or "").lower()).split()


try:
    from rapidfuzz.distance import Levenshtein as _Lev
    def _edit(a, b):
        return _Lev.distance(a, b)
except Exception:                     # pragma: no cover - rapidfuzz is in the container
    def _edit(a, b):
        n, m = len(a), len(b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[m]


def _wer_cer(ref, hyp):
    """(WER, CER) of hyp vs ref over normalized text; (None, None) if ref empty."""
    rw, hw = _err_norm(ref), _err_norm(hyp)
    if not rw:
        return None, None
    wer = _edit(rw, hw) / len(rw)
    rc, hc = " ".join(rw), " ".join(hw)
    cer = (_edit(rc, hc) / len(rc)) if rc else None
    return wer, cer


def _load_rover_transcripts(search_paths) -> Dict[str, dict]:
    """``{cut_id: {"text":..., "source":...}}`` from each ``rover/merged.jsonl``.

    Picks the best improved transcript per cut, preferring the ITN-normalized
    enhanced form:
        text_enhanced_itn > text_enhanced > text_itn > text
    Also attaches ``rover_wer`` / ``rover_cer`` — the WER/CER of the ROVER
    consensus (rover.text) against the original transcript (ref_text).  Empty
    texts are skipped.  Keyed by cut_id alone (content-hash ids are unique across
    datasets).  Returns ``{}`` when no rover files are found.
    """
    out: Dict[str, dict] = {}
    pref = [("text_enhanced_itn", "rover_enhanced_itn"),
            ("text_enhanced",     "rover_enhanced"),
            ("text_itn",          "rover_itn"),
            ("text",              "rover")]
    n_files = 0
    for sp in (search_paths or []):
        for jsonl in sorted(Path(sp).rglob("rover/merged.jsonl")):
            n_files += 1
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rv = d.get("rover") or {}
                    text = src = None
                    for field, label in pref:
                        v = rv.get(field)
                        if v and str(v).strip():
                            text, src = v, label
                            break
                    if not text:
                        continue
                    cid = d.get("cut_id") or ""
                    cid = cid[2:] if cid.startswith("./") else cid
                    wer, cer = _wer_cer(d.get("ref_text"), rv.get("text"))
                    out[cid] = {"text": text, "source": src,
                                "rover_wer": wer, "rover_cer": cer}
    logger.info("Loaded ASR transcripts for %d cuts from %d rover file(s).",
                len(out), n_files)
    return out


def _assert_in_place_writable(shar_dirs) -> None:
    """Fail *before* touching anything if any cuts.*.jsonl.gz is not writable.

    ``in_place`` rewrites the original cuts shards (audio tars are never written
    in any mode).  Refuse loudly on read-only / not-owned inputs so we never
    half-modify a dataset the caller doesn't own.
    """
    bad = []
    for sd in shar_dirs:
        sp = Path(sd).resolve()
        idx_path = sp / SHAR_INDEX_FILENAME
        if not idx_path.is_file():
            continue
        with open(idx_path) as f:
            cuts_rel = json.load(f).get("fields", {}).get("cuts", [])
        for rp in cuts_rel:
            p = sp / rp
            if p.exists() and not os.access(p, os.W_OK):
                bad.append(str(p))
    if bad:
        raise PermissionError(
            "apply_to_shar in_place=true but these cuts shards are not writable "
            "(refusing to modify files you may not own): "
            + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
            + ".  Use a clone instead (set output_shar_dir, in_place=false).")


def run(cfg: dict) -> None:
    cfg_g = cfg.get("apply_to_shar", {})
    if not cfg_g.get("enabled", False):
        logger.info("Stage G is disabled (apply_to_shar.enabled=false).  Sidecar "
                    "parquet at <output_dir>/final/assignments.parquet is the canonical "
                    "output.  Use --enable to override.")
        return

    output_dir = Path(cfg["output_dir"])
    dedup_parquet = output_dir / "retention" / "assignments.parquet"
    if not dedup_parquet.exists():
        raise RuntimeError(f"Stage F output missing: {dedup_parquet}")

    in_place    = bool(cfg_g.get("in_place", False))
    if in_place and not bool(cfg_g.get("allow_in_place", False)):
        # Clone-only by policy: never modify the original tars or cuts.  in_place
        # rewrites the ORIGINAL cuts.*.jsonl.gz, so it requires a deliberate
        # double opt-in (in_place=true AND allow_in_place=true).
        raise ValueError(
            "apply_to_shar.in_place=true modifies the ORIGINAL cuts.*.jsonl.gz. "
            "This pipeline is clone-only: leave in_place=false (default) to write a "
            "new augmented dataset (fresh cuts.*.jsonl.gz + symlinked audio tars, "
            "originals untouched), or set allow_in_place=true to deliberately override.")
    link_mode   = str(cfg_g.get("link_mode", "symlink"))
    workers     = int(cfg_g.get("workers", min(32, os.cpu_count() or 4)))
    include_metrics = bool(cfg_g.get("include_quality_metrics", True))
    qflat = output_dir / "quality" / "merged.parquet"
    quality_flat = qflat if (include_metrics and qflat.exists()) else None
    # Improved ASR transcript (ROVER enhanced) for cut.custom["asr_transcript"].
    # Read straight from the quality JSONLs' rover/merged.jsonl (the transcript
    # text never enters the numeric quality parquet).  On by default.
    add_transcript = bool(cfg_g.get("add_asr_transcript", True))
    transcript_lookup: Dict[str, dict] = {}
    if add_transcript:
        from .quality_ingest import _resolve_search_paths
        transcript_lookup = _load_rover_transcripts(_resolve_search_paths(cfg))
    output_shar = cfg_g.get("output_shar_dir")
    if not in_place and not output_shar:
        raise ValueError("apply_to_shar.output_shar_dir is required when not in_place.")
    if output_shar:
        output_shar = Path(output_shar).resolve()
        # Refuse to clobber any input shar.  This is the single biggest
        # destructive failure mode.
        for sd in (cfg["shar_dirs"] if isinstance(cfg["shar_dirs"], list)
                   else [cfg["shar_dirs"]]):
            if Path(sd).resolve() == output_shar:
                raise ValueError(
                    f"output_shar_dir ({output_shar}) is identical to input "
                    f"shar_dir ({sd}).  Refusing to overwrite.")
            try:
                output_shar.relative_to(Path(sd).resolve())
                raise ValueError(
                    f"output_shar_dir ({output_shar}) is *inside* input "
                    f"shar_dir ({sd}).  Refusing to write under the original.")
            except ValueError as e:
                if "Refusing" in str(e):
                    raise
                # The relative_to threw because it's not a subdir; that's the
                # safe case.
                pass

    shar_dirs = cfg["shar_dirs"]
    if isinstance(shar_dirs, str):
        shar_dirs = [shar_dirs]

    if in_place:
        # Fail before touching anything if any cuts shard is not writable.
        _assert_in_place_writable(shar_dirs)

    overrides    = cfg.get("dataset_overrides") or {}
    dataset_root = cfg.get("dataset_root")
    from .manifest import _derive_dataset_name  # type: ignore

    t0 = time.time()
    total_injected = 0
    total_singleton = 0
    total_unknown = 0

    for sd in shar_dirs:
        sp = Path(sd).resolve()
        if not sp.is_dir():
            raise FileNotFoundError(f"Shar dir missing: {sp}")
        dataset = overrides.get(str(sp), _derive_dataset_name(str(sp), dataset_root))
        idx_path = sp / SHAR_INDEX_FILENAME
        if not idx_path.is_file():
            raise FileNotFoundError(f"Missing shar_index in {sp}")

        with open(idx_path) as f:
            payload = json.load(f)
        fields = payload.get("fields", {})
        cuts_rel = fields.get("cuts", [])
        if not cuts_rel:
            raise ValueError(f"shar_index missing 'cuts': {idx_path}")

        if in_place:
            target_root = sp
        else:
            # Mirror the source shar's relative position under output_shar_dir.
            try:
                rel_to_root = sp.relative_to(Path(dataset_root).resolve()) \
                    if dataset_root else Path(sp.name)
            except (ValueError, TypeError):
                rel_to_root = Path(dataset)
            target_root = output_shar / rel_to_root
            target_root.mkdir(parents=True, exist_ok=True)
            # Refuse to overwrite an input shar.
            if target_root.resolve() == sp.resolve():
                raise ValueError(f"output_shar_dir resolves to input shar: {sp}")

            # Copy shar_index.json verbatim (paths are relative).
            shutil.copy2(idx_path, target_root / SHAR_INDEX_FILENAME)
            # Symlink every non-cuts field tar.
            for field, paths in fields.items():
                if field == "cuts":
                    continue
                for rp in paths:
                    src_path = sp / rp
                    dst_path = target_root / rp
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    _link(src_path, dst_path, link_mode)

        # Build dedup lookup once per shar.
        dedup_lookup = _build_lookup(dedup_parquet, dataset_filter=dataset,
                                     quality_flat=quality_flat)
        if not dedup_lookup:
            logger.warning("No dedup rows for dataset %s; cuts.jsonl.gz will be "
                           "rewritten unchanged.  (Did Stage F see this dataset?)",
                           dataset)

        tasks = []
        for rp in cuts_rel:
            src_path = sp / rp
            if in_place:
                out_path = src_path
            else:
                out_path = target_root / rp
                out_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((str(src_path), str(out_path), dataset, dedup_lookup,
                          transcript_lookup, in_place))

        if workers <= 1 or len(tasks) <= 1:
            for task in tasks:
                inj, sng, unk = _process_one_shard(task)
                total_injected += inj; total_singleton += sng; total_unknown += unk
        else:
            with Pool(workers) as pool:
                for inj, sng, unk in pool.imap_unordered(_process_one_shard, tasks):
                    total_injected += inj; total_singleton += sng; total_unknown += unk

    logger.info("Stage G done in %.1fs: %d cuts injected, %d singletons, %d unknown.",
                time.time() - t0, total_injected, total_singleton, total_unknown)

    if not in_place:
        from . import run_layout
        run_layout.finalize_stage(output_dir, "output_shar", rows=total_injected,
                                  extra={"singletons": total_singleton,
                                         "unknown": total_unknown})


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage G: apply dedup to shar (clone)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--enable", action="store_true",
                        help="Force enable even if cfg has apply_to_shar.enabled=false.")
    parser.add_argument("--in-place", action="store_true",
                        help="Rewrite original cuts.*.jsonl.gz instead of cloning.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = _load_cfg(args.config)
    if args.enable or args.in_place:
        cfg.setdefault("apply_to_shar", {})["enabled"] = True
    if args.in_place:
        cfg["apply_to_shar"]["in_place"] = True
    run(cfg)


if __name__ == "__main__":
    main()
