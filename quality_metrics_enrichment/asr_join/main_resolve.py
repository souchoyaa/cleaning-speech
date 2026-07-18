"""Driver: 1-slot (parakeet) ROVER + WER/CER + LLM conflict resolution (Task #8).

Standalone Phase-2 variant — does NOT touch main.py / itn.py / the blanket-ITN
run. For each cut:
  - rover.text = parakeet transcript (single slot),
  - rover.wer / rover.cer = parakeet vs the dataset reference,
  - if they disagree (WER > resolve.gate_wer) ask gemma to resolve the conflict
    → rover.text_resolved (fallback = reference on any failure),
  - otherwise rover.text_resolved = parakeet text.
Writes a fresh merged.jsonl under ``merge_output_dir`` (a NEW folder), leaving
the existing ITN run's merged.jsonl untouched.

Run:
  python -m asr_join.main_resolve --config asr_join/egs/voxpopuli_resolve_<lang>.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import httpx
import orjson
import yaml

from . import join
from .conflict_resolve import ConflictResolveClient, ResolveTransportError, wer_cer

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def _resolve_all(rows: list, to_resolve: list, rcfg: dict) -> dict:
    rc = ConflictResolveClient(rcfg)
    conc = int(rcfg.get("concurrency", 128))
    chunk = int(rcfg.get("chunk_size", 2000))
    cb_frac = float(rcfg.get("circuit_break_fraction", 0.8))
    timeout = httpx.Timeout(float(rcfg.get("timeout_seconds", 90.0)), connect=30.0)
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    sem = asyncio.Semaphore(conc)
    n_done = n_fail = 0
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def one(row) -> int:
            nonlocal n_done, n_fail
            async with sem:
                try:
                    res = await rc.resolve(client, row["rover"]["text"],
                                           row.get("ref_text") or "", row.get("language"))
                    row["rover"]["text_resolved"] = res
                    return 0
                except ResolveTransportError:
                    n_fail += 1
                    row["rover"]["text_resolved"] = row.get("ref_text") or row["rover"]["text"]
                    return 1
                finally:
                    n_done += 1

        for start in range(0, len(to_resolve), chunk):
            batch = to_resolve[start:start + chunk]
            fails = await asyncio.gather(*(one(r) for r in batch))
            if 0.0 < cb_frac <= 1.0 and batch:
                frac = sum(fails) / len(batch)
                if frac >= cb_frac:
                    raise RuntimeError(
                        f"resolve circuit breaker: chunk [{start}:{start+len(batch)}] "
                        f"had {sum(fails)}/{len(batch)} transport failures "
                        f"({frac:.0%} >= {cb_frac:.0%}). Server likely down. Aborting.")
            if n_done % 10000 < chunk:
                wall = time.monotonic() - t0
                logger.info("resolve: %d/%d done (%d failed, %.1f rows/s)",
                            n_done, len(to_resolve), n_fail, n_done / max(wall, 1e-6))
    return {"n_resolved": len(to_resolve), "n_failed": n_fail,
            "seconds": time.monotonic() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description="1-slot ROVER + WER/CER + conflict resolution")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    _setup_logging()

    cfg = yaml.safe_load(Path(args.config).read_text())
    input_root = Path(cfg["input_root"])
    out_root = Path(cfg.get("merge_output_dir") or cfg["input_root"])
    dataset = cfg.get("dataset")
    lang = cfg.get("expected_language")
    rcfg = cfg.get("resolve") or {}
    gate = float(rcfg.get("gate_wer", 0.0))
    sdef = (cfg.get("slots") or {}).get("parakeet") or {}
    dir_name = sdef.get("dir_name", "parakeet")
    payload_at = sdef.get("payload_at", "parakeet")

    t0 = time.time()
    slot = join.load_slot(input_root, dir_name, payload_at, "parakeet")
    rows = []
    for cid, rec in slot.items():
        ptext = (rec.get("_hyp") or {}).get("text") or ""
        ref = rec.get("ref_text") or ""
        w, c = wer_cer(ref, ptext)
        row = {
            "cut_id": cid,
            "duration": rec.get("duration"),
            "ref_text": ref,
            "language": lang or rec.get("language_hint"),
            "rover": {"text": ptext, "wer": w, "cer": c, "text_resolved": ptext},
        }
        if dataset:
            row["dataset"] = dataset
        rows.append(row)
    logger.info("loaded %d parakeet cuts in %.1fs", len(rows), time.time() - t0)

    # Gate: resolve cuts where parakeet disagrees with a non-empty reference.
    to_resolve = [r for r in rows
                  if r["rover"]["wer"] is not None
                  and r["rover"]["wer"] > gate
                  and (r.get("ref_text") or "").strip()]
    logger.info("conflict gate WER>%.3f → resolving %d / %d cuts (%.1f%%)",
                gate, len(to_resolve), len(rows), 100 * len(to_resolve) / max(len(rows), 1))

    stats = asyncio.run(_resolve_all(rows, to_resolve, rcfg))
    logger.info("resolve done: %s", stats)

    out_path = out_root / "quality_asr" / "rover" / "merged.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for row in rows:
            f.write(orjson.dumps(row)); f.write(b"\n")
    logger.info("wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
