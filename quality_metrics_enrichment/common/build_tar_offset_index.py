"""Build cut_id → tar member byte-offset index for Lhotse Shar trees.

Standalone CLI. Not a pipeline stage — run by hand against any dataset
when you want random-access into the shar tars without iterating them.

For every shar dataset under ``<shar_root>`` (any dir holding a
``shar_index.json``), write a single JSON file mirroring the layout under
``<output_root>``:

    <output_root>/<dataset_relpath>/cut_offset_index.json

Each entry maps a cut id to the worker subfolder, shard index, and the
byte offset of its 512-byte tar header inside the matching
``recording.NNNNNN.tar``. Lets a caller seek straight to one audio
member without iterating the tar:

    fp = open(f"{shar_root}/{dataset}/{shar}/recording.{shard:06d}.tar", "rb")
    fp.seek(tar_offset)
    hdr = tarfile.TarInfo.frombuf(fp.read(512), "utf-8", "surrogateescape")
    data = fp.read(hdr.size)   # raw audio bytes (flac/opus/wav/…)

Alignment is by position: the Nth line of ``cuts.NNN.jsonl.gz`` pairs
with the Nth member of ``recording.NNN.tar`` (Lhotse Shar invariant).
A length mismatch fails the shard rather than guess.

The shar tree itself is never written to.

Examples
--------
    # preview only
    python build_tar_offset_index.py \\
        /…/SHAR/stage_2/commonvoice22_sidon \\
        /…/results/cut_offset_index/commonvoice22_sidon --dry-run

    # parallel scan, 16 workers
    python build_tar_offset_index.py \\
        /…/SHAR/stage_2/commonvoice22_sidon \\
        /…/results/cut_offset_index/commonvoice22_sidon --workers 16
"""

from __future__ import annotations

import os
import sys

# Strip our own dir from sys.path so the local `logging.py` doesn't shadow
# stdlib (same idiom as shar_du.py).
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SELF_DIR]

import argparse
import gzip
import json
import re
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

SHAR_INDEX_FILENAME = "shar_index.json"
OUT_FILENAME = "cut_offset_index.json"

# recording.000000.tar → 0
_SHARD_RE = re.compile(r"\.(\d+)\.tar$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _find_shar_dirs(root: Path) -> list[Path]:
    """Every dir at or under ``root`` that holds a shar_index.json (no nesting)."""
    out: list[Path] = []

    def walk(d: Path) -> None:
        if (d / SHAR_INDEX_FILENAME).is_file():
            out.append(d)
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except (PermissionError, OSError):
            return
        for c in children:
            walk(c)

    walk(root)
    return out


def _dataset_rel(shar_dir: Path, root: Path) -> str:
    try:
        rel = str(shar_dir.relative_to(root))
    except ValueError:
        rel = str(shar_dir)
    return shar_dir.name if rel == "." else rel


# ---------------------------------------------------------------------------
# Per-shard scan
# ---------------------------------------------------------------------------


@dataclass
class _ShardJob:
    dataset_rel: str         # e.g. "de/other"
    cuts_path: str           # absolute
    recording_path: str      # absolute
    shar_subdir: str         # parent dir of the shard files (e.g. "worker_01")
    shard_index: int         # parsed from recording.NNNNNN.tar


@dataclass
class _ShardResult:
    dataset_rel: str
    shar_subdir: str
    shard_index: int
    items: list = field(default_factory=list)  # list[tuple[str, int]] = (cut_id, tar_offset)
    error: str | None = None


def _shard_jobs(shar_dir: Path, root: Path) -> list[_ShardJob]:
    idx = shar_dir / SHAR_INDEX_FILENAME
    payload = json.loads(idx.read_text())
    fields_ = payload.get("fields", {})
    cuts_rel = fields_.get("cuts") or []
    recs_rel = fields_.get("recording") or []
    if not cuts_rel or not recs_rel:
        raise ValueError(f"shar_index missing 'cuts' or 'recording' field: {idx}")
    if len(cuts_rel) != len(recs_rel):
        raise ValueError(
            f"shar_index cuts/recording length mismatch "
            f"({len(cuts_rel)} vs {len(recs_rel)}): {idx}"
        )

    dataset_rel = _dataset_rel(shar_dir, root)
    jobs: list[_ShardJob] = []
    for cr, rr in zip(cuts_rel, recs_rel):
        cp = Path(cr)
        rp = Path(rr)
        if cp.is_absolute() or rp.is_absolute():
            raise ValueError(f"absolute relpath in shar_index: {idx}")
        m = _SHARD_RE.search(rp.name)
        if m is None:
            raise ValueError(f"can't parse shard index from {rp.name}")
        parent = rp.parent.as_posix()
        jobs.append(_ShardJob(
            dataset_rel=dataset_rel,
            cuts_path=str(shar_dir / cp),
            recording_path=str(shar_dir / rp),
            shar_subdir="" if parent == "." else parent,
            shard_index=int(m.group(1)),
        ))
    return jobs


def _scan_one(job: _ShardJob) -> _ShardResult:
    res = _ShardResult(
        dataset_rel=job.dataset_rel,
        shar_subdir=job.shar_subdir,
        shard_index=job.shard_index,
    )
    try:
        cut_ids: list[str] = []
        with gzip.open(job.cuts_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cut_ids.append(json.loads(line)["id"])

        offsets: list[int] = []
        with tarfile.open(job.recording_path, "r") as tar:
            for m in tar:
                if m.isfile():
                    offsets.append(m.offset)

        if len(cut_ids) != len(offsets):
            res.error = (
                f"length mismatch: {len(cut_ids)} cuts vs {len(offsets)} tar "
                f"members ({job.cuts_path} / {job.recording_path})"
            )
            return res

        res.items = list(zip(cut_ids, offsets))
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


# ---------------------------------------------------------------------------
# Write & reporting
# ---------------------------------------------------------------------------


def _write_index(out_path: Path, entries: dict, dry_run: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, separators=(",", ":"), ensure_ascii=False)
    if dry_run:
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, out_path)


@dataclass
class _DatasetTotals:
    cuts: int = 0
    shards: int = 0
    duplicates: int = 0
    errors: list = field(default_factory=list)


def _print_table(totals: dict[str, _DatasetTotals], dry_run: bool, out=sys.stdout) -> None:
    rows = sorted(totals.items())
    name_w = max([len("DATASET")] + [len(n) for n, _ in rows])
    widths = [name_w, 12, 7, 10, 7]
    header = ["DATASET", "CUTS", "SHARDS", "DUP", "ERRORS"]
    fmt = "  ".join(
        f"{{:{'<' if i == 0 else '>'}{w}}}" for i, w in enumerate(widths)
    )
    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt.format(*header), file=out)
    print(sep, file=out)

    g = _DatasetTotals()
    for name, t in rows:
        g.cuts += t.cuts
        g.shards += t.shards
        g.duplicates += t.duplicates
        g.errors.extend(f"[{name}] {e}" for e in t.errors)
        print(fmt.format(name, f"{t.cuts:,}", t.shards, t.duplicates, len(t.errors)),
              file=out)
    print(sep, file=out)
    print(fmt.format("TOTAL", f"{g.cuts:,}", g.shards, g.duplicates, len(g.errors)),
          file=out)

    if dry_run:
        print("\n[dry-run] no files were written", file=out)
    if g.errors:
        print(f"\nERRORS ({len(g.errors)}):", file=out)
        for e in g.errors:
            print(f"  {e}", file=out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build cut_id → tar offset index for Lhotse Shar trees."
    )
    p.add_argument("shar_root", type=Path,
                   help="root of the shar tree to scan (read-only)")
    p.add_argument("output_root", type=Path,
                   help="results dir; one cut_offset_index.json per dataset, "
                        "mirroring the shar layout")
    p.add_argument("--workers", type=int, default=1,
                   help="process pool size for scanning shards (default 1)")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild even if cut_offset_index.json already exists "
                        "(default: skip those datasets)")
    p.add_argument("--dry-run", action="store_true",
                   help="scan and report but write nothing")
    args = p.parse_args(argv)

    shar_root = args.shar_root.resolve()
    out_root = args.output_root.resolve()
    if not shar_root.is_dir():
        print(f"error: not a directory: {shar_root}", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    shar_dirs = _find_shar_dirs(shar_root)
    if not shar_dirs:
        print(f"no shar datasets found under {shar_root}", file=sys.stderr)
        return 1

    if not args.overwrite:
        kept = []
        for sd in shar_dirs:
            if (out_root / _dataset_rel(sd, shar_root) / OUT_FILENAME).is_file():
                continue
            kept.append(sd)
        skipped = len(shar_dirs) - len(kept)
        if skipped:
            print(f"skipping {skipped} dataset(s) with existing index "
                  f"(use --overwrite to rebuild)", file=sys.stderr)
        shar_dirs = kept

    if not shar_dirs:
        print("nothing to do", file=sys.stderr)
        return 0

    all_jobs: list[_ShardJob] = []
    totals: dict[str, _DatasetTotals] = {}
    for sd in shar_dirs:
        ds = _dataset_rel(sd, shar_root)
        totals.setdefault(ds, _DatasetTotals())
        try:
            all_jobs.extend(_shard_jobs(sd, shar_root))
        except Exception as e:
            totals[ds].errors.append(f"shar_index: {e}")

    results: list[_ShardResult] = []
    if args.workers <= 1:
        for j in all_jobs:
            results.append(_scan_one(j))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed(ex.submit(_scan_one, j) for j in all_jobs):
                results.append(fut.result())

    per_dataset: dict[str, dict[str, dict]] = {}
    for r in results:
        t = totals.setdefault(r.dataset_rel, _DatasetTotals())
        t.shards += 1
        if r.error:
            t.errors.append(f"shard {r.shard_index} ({r.shar_subdir}): {r.error}")
            continue
        bucket = per_dataset.setdefault(r.dataset_rel, {})
        for cid, off in r.items:
            if cid in bucket:
                t.duplicates += 1
                continue
            bucket[cid] = {
                "shar": r.shar_subdir,
                "shard": r.shard_index,
                "tar_offset": off,
            }
            t.cuts += 1

    for ds, entries in per_dataset.items():
        out_path = out_root / ds / OUT_FILENAME
        _write_index(out_path, entries, args.dry_run)

    _print_table(totals, dry_run=args.dry_run)
    elapsed = time.monotonic() - t0
    print(f"\nscanned {len(shar_dirs)} dataset(s) in {elapsed:.1f}s",
          file=sys.stderr)

    any_error = any(t.errors for t in totals.values())
    return 3 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
