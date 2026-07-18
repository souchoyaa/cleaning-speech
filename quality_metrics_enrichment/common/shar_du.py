"""du-like cut counter for Lhotse Shar trees.

Walk a directory, find every shar dataset (any subdir containing
``shar_index.json``), and report ``#cuts``, total hours, and ``#shards``
per dataset plus a grand total.

Counting requires gzip-streaming the ``cuts.*.jsonl.gz`` shards (the
index alone only knows shard counts). Metadata is small (~5 GB per
100 TB of audio), so a single login-node CPU is usually enough; bump
``--workers`` for very large trees on slow filesystems.

Examples
--------
    python shar_du.py /capstor/.../SHAR/stage_2
    python shar_du.py /capstor/.../SHAR/stage_2 --workers 16 --json
"""

from __future__ import annotations

import os
import sys

# This dir contains a `logging.py` that would shadow stdlib `logging` if our
# own directory ends up on sys.path (which happens when run as a script).
# Strip it before importing anything that imports `logging`.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SELF_DIR]

import argparse
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SHAR_INDEX_FILENAME = "shar_index.json"


@dataclass
class _ShardJob:
    dataset: str          # display name (path relative to root)
    shar_dir: str         # absolute shar root
    cuts_path: str        # absolute path to one cuts.*.jsonl.gz
    parse_duration: bool


@dataclass
class _ShardResult:
    dataset: str
    cuts: int
    duration_secs: float  # 0.0 if parse_duration=False


@dataclass
class _DatasetTotals:
    cuts: int = 0
    duration_secs: float = 0.0
    shards: int = 0


def _find_shar_dirs(root: Path, max_depth: int | None) -> list[Path]:
    """Recursively yield every directory under ``root`` (inclusive) that
    contains a ``shar_index.json``. Does not descend into a shar dir
    once found (shars don't nest)."""
    out: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if (d / SHAR_INDEX_FILENAME).is_file():
            out.append(d)
            return  # don't descend into a shar
        if max_depth is not None and depth >= max_depth:
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except (PermissionError, OSError):
            return
        for c in children:
            walk(c, depth + 1)

    walk(root, 0)
    return out


def _shard_jobs(shar_dir: Path, root: Path, parse_duration: bool) -> list[_ShardJob]:
    """Read shar_index.json and emit one job per cuts shard."""
    idx_path = shar_dir / SHAR_INDEX_FILENAME
    with open(idx_path) as f:
        payload = json.load(f)
    cuts_rel = payload.get("fields", {}).get("cuts", [])
    if not cuts_rel:
        raise ValueError(f"shar_index has no 'cuts' field: {idx_path}")

    try:
        dataset = str(shar_dir.relative_to(root))
    except ValueError:
        dataset = str(shar_dir)
    if dataset == ".":
        dataset = shar_dir.name

    jobs = []
    for r in cuts_rel:
        rp = Path(r)
        if rp.is_absolute():
            raise ValueError(f"Absolute paths in shar_index are forbidden: {rp}")
        jobs.append(_ShardJob(
            dataset=dataset,
            shar_dir=str(shar_dir),
            cuts_path=str(shar_dir / rp),
            parse_duration=parse_duration,
        ))
    return jobs


def _count_one(job: _ShardJob) -> _ShardResult:
    cuts = 0
    duration = 0.0
    with gzip.open(job.cuts_path, "rt", encoding="utf-8") as f:
        if job.parse_duration:
            for line in f:
                if not line.strip():
                    continue
                cuts += 1
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = d.get("duration")
                if v is not None:
                    duration += float(v)
        else:
            for line in f:
                if line.strip():
                    cuts += 1
    return _ShardResult(dataset=job.dataset, cuts=cuts, duration_secs=duration)


def _aggregate(results: Iterable[_ShardResult]) -> dict[str, _DatasetTotals]:
    totals: dict[str, _DatasetTotals] = {}
    for r in results:
        t = totals.setdefault(r.dataset, _DatasetTotals())
        t.cuts += r.cuts
        t.duration_secs += r.duration_secs
        t.shards += 1
    return totals


def _print_table(
    totals: dict[str, _DatasetTotals],
    show_duration: bool,
    out=sys.stdout,
) -> None:
    rows = sorted(totals.items())
    name_w = max([len("DATASET")] + [len(n) for n, _ in rows])

    if show_duration:
        header = f"{'DATASET':<{name_w}}  {'CUTS':>12}  {'HOURS':>10}  {'SHARDS':>7}"
    else:
        header = f"{'DATASET':<{name_w}}  {'CUTS':>12}  {'SHARDS':>7}"
    print(header, file=out)
    print("-" * len(header), file=out)

    grand_cuts = grand_dur = 0
    grand_shards = 0
    for name, t in rows:
        grand_cuts += t.cuts
        grand_dur += t.duration_secs
        grand_shards += t.shards
        if show_duration:
            print(f"{name:<{name_w}}  {t.cuts:>12,}  {t.duration_secs/3600:>10.2f}  {t.shards:>7}", file=out)
        else:
            print(f"{name:<{name_w}}  {t.cuts:>12,}  {t.shards:>7}", file=out)

    print("-" * len(header), file=out)
    if show_duration:
        print(f"{'TOTAL':<{name_w}}  {grand_cuts:>12,}  {grand_dur/3600:>10.2f}  {grand_shards:>7}", file=out)
    else:
        print(f"{'TOTAL':<{name_w}}  {grand_cuts:>12,}  {grand_shards:>7}", file=out)


def _print_json(totals: dict[str, _DatasetTotals], show_duration: bool, out=sys.stdout) -> None:
    for name, t in sorted(totals.items()):
        rec = {"dataset": name, "cuts": t.cuts, "shards": t.shards}
        if show_duration:
            rec["duration_secs"] = t.duration_secs
            rec["duration_hours"] = t.duration_secs / 3600
        print(json.dumps(rec), file=out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="du-like cut counter for Lhotse Shar trees")
    p.add_argument("root", type=Path, help="parent directory to scan recursively")
    p.add_argument("--workers", type=int, default=1,
                   help="process pool size for gzip-counting shards (default 1)")
    p.add_argument("--depth", type=int, default=None,
                   help="cap recursion depth when searching for shar_index.json")
    p.add_argument("--json", action="store_true",
                   help="emit one JSON object per dataset instead of a table")
    p.add_argument("--no-duration", action="store_true",
                   help="skip parsing the duration field (slightly faster)")
    args = p.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    parse_duration = not args.no_duration

    t0 = time.monotonic()
    shar_dirs = _find_shar_dirs(root, args.depth)
    if not shar_dirs:
        print(f"no shar datasets found under {root}", file=sys.stderr)
        return 1

    jobs: list[_ShardJob] = []
    for sd in shar_dirs:
        jobs.extend(_shard_jobs(sd, root, parse_duration))

    results: list[_ShardResult] = []
    if args.workers <= 1:
        for j in jobs:
            results.append(_count_one(j))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed(ex.submit(_count_one, j) for j in jobs):
                results.append(fut.result())

    totals = _aggregate(results)
    elapsed = time.monotonic() - t0

    if args.json:
        _print_json(totals, parse_duration)
    else:
        _print_table(totals, parse_duration)
        print(f"\nscanned {len(shar_dirs)} datasets, {len(jobs)} shards in {elapsed:.1f}s",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
