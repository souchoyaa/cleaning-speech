"""Append-only run-history ledger — one line per lifecycle event.

A shared ledger (``$OUT_BASE/logs/run_history.jsonl``, override with
``RUN_HISTORY_LEDGER``) recording which YAMLs ran, succeeded, or crashed across a
multi-node run; ``status.sh`` joins it with ``squeue`` for a status table. Each
event is one short orjson line written ``O_APPEND`` (atomic ≤ PIPE_BUF), so
concurrent ranks need no locking.

Lifecycle: ``start_run(...)`` emits a ``start`` event and returns a context;
``end_run(ctx, status, **extras)`` emits ``end`` (status ∈ {ok, crashed,
terminated}) — call it from try/finally so crashes are still recorded.
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import orjson

logger = logging.getLogger(__name__)


def _ledger_path() -> Optional[Path]:
    """Return the ledger file path, or None when disabled.

    Resolution order:
      1. ``RUN_HISTORY_LEDGER`` env var (absolute path; ``""`` to disable)
      2. ``$OUT_BASE/logs/run_history.jsonl``
      3. ``$PIPELINE_LOG_DIR/../run_history.jsonl`` (sibling of pipeline log dir)
      4. None → disabled (no ledger writes; runner unaffected)
    """
    env = os.environ.get("RUN_HISTORY_LEDGER")
    if env is not None:
        return Path(env) if env else None
    out_base = os.environ.get("OUT_BASE")
    if out_base:
        return Path(out_base) / "logs" / "run_history.jsonl"
    pipeline_log_dir = os.environ.get("PIPELINE_LOG_DIR")
    if pipeline_log_dir:
        return Path(pipeline_log_dir).parent / "run_history.jsonl"
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(record: dict) -> None:
    """Append one record as a single JSONL line. Atomic per POSIX.

    Failures are swallowed (logged at WARNING) — never break the runner
    over a ledger write error.
    """
    path = _ledger_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = orjson.dumps(record) + b"\n"
        if len(data) > 4000:
            # Stay comfortably below PIPE_BUF (4096) so the kernel keeps
            # the append atomic across concurrent writers. Truncate the
            # most likely culprit (reason / traceback strings).
            r = dict(record)
            if "reason" in r and isinstance(r["reason"], str):
                r["reason"] = r["reason"][:1500] + "…[truncated]"
            data = orjson.dumps(r) + b"\n"
        # O_APPEND ensures kernel-level position atomicity; the write
        # itself is atomic as long as the buffer is ≤ PIPE_BUF.
        with path.open("ab") as f:
            f.write(data)
    except Exception as e:
        logger.warning("run_history: append failed (%s) — continuing.", e)


def start_run(
    *,
    stage: str,
    yaml_path: str,
    rank: int = 0,
    world_size: int = 1,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Emit a ``start`` event and return a context dict for ``end_run``.

    The context carries enough state to write a matching ``end`` later.
    ``run_id`` defaults to a short UUID so multiple restarts of the same
    YAML are distinguishable in the ledger.
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
    ctx = {
        "run_id":        run_id,
        "stage":         stage,
        "yaml":          yaml_path,
        "rank":          int(rank),
        "world_size":    int(world_size),
        "output_dir":    output_dir,
        "host":          socket.gethostname(),
        "slurm_job_id":  os.environ.get("SLURM_JOB_ID"),
        "pid":           os.getpid(),
        "started_at":    _now_iso(),
        "_t0":           time.monotonic(),
    }
    rec = {
        "event":        "start",
        "run_id":       ctx["run_id"],
        "stage":        ctx["stage"],
        "yaml":         ctx["yaml"],
        "rank":         ctx["rank"],
        "world_size":   ctx["world_size"],
        "output_dir":   ctx["output_dir"],
        "host":         ctx["host"],
        "slurm_job_id": ctx["slurm_job_id"],
        "pid":          ctx["pid"],
        "ts":           ctx["started_at"],
    }
    _append(rec)
    return ctx


def end_run(
    ctx: dict,
    *,
    status: str,
    reason: Optional[str] = None,
    **extras,
) -> None:
    """Emit an ``end`` event. ``status`` ∈ {"ok", "crashed", "terminated"}.

    Always call from a try/finally so SIGTERM and crash paths still write
    a record. Any keyword in ``extras`` (e.g. ``n_committed=4321``) is
    folded into the record verbatim.
    """
    rec = {
        "event":      "end",
        "run_id":     ctx.get("run_id"),
        "stage":      ctx.get("stage"),
        "yaml":       ctx.get("yaml"),
        "rank":       ctx.get("rank"),
        "status":     status,
        "ts":         _now_iso(),
        "wall_seconds": round(time.monotonic() - ctx.get("_t0", time.monotonic()), 2),
    }
    if reason:
        rec["reason"] = reason
    rec.update(extras)
    _append(rec)


__all__ = ["start_run", "end_run", "guard_cfg_hash"]


# ---------------------------------------------------------------------------
# Cfg-hash sentinel — refuse to append to an output dir whose previous
# run used a materially different config.
# ---------------------------------------------------------------------------


import hashlib


class CfgMismatch(RuntimeError):
    """Raised when ``<output_dir>/.cfg_hash`` exists with a different
    digest. The caller can let it propagate (refuse) or pass
    ``allow_overwrite=True`` to overwrite the sentinel (for intentional
    cfg edits — old rows in the JSONL stay; new rows use the new cfg).
    """


def _hash_cfg(cfg) -> str:
    """SHA256 of a stably-serialized YAML cfg. Sorted keys, no whitespace
    variance — same logical cfg always hashes to the same value."""
    payload = orjson.dumps(cfg, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


def guard_cfg_hash(
    output_dir,
    cfg: dict,
    *,
    allow_overwrite: bool = False,
    extra_ignore_keys: Optional[tuple] = None,
) -> None:
    """Refuse to write into ``output_dir`` if its previous run used a
    different cfg digest.

    - First run on a fresh dir: writes ``.cfg_hash`` and returns.
    - Same cfg as before: silently returns.
    - Different cfg: raises ``CfgMismatch`` with a clear message telling
      the user what differed enough to flag. Override with
      ``allow_overwrite=True`` (CLI ``--resume-anyway`` is the typical
      gateway) and the sentinel is updated to the new hash.

    The cfg is deep-copied before hashing and ``extra_ignore_keys`` are
    deleted — pass keys whose value doesn't affect data quality (e.g.
    ``output_dir`` itself, which differs per multilang iteration).
    """
    if not output_dir:
        return  # nothing to guard
    od = Path(str(output_dir))
    od.mkdir(parents=True, exist_ok=True)
    sentinel = od / ".cfg_hash"

    # Don't include keys that are operational rather than semantic — they
    # change between runs without altering the data.
    cfg_to_hash = {k: v for k, v in cfg.items() if k not in {"output_dir"}}
    if extra_ignore_keys:
        for k in extra_ignore_keys:
            cfg_to_hash.pop(k, None)

    current = _hash_cfg(cfg_to_hash)

    if sentinel.exists():
        try:
            previous = sentinel.read_text().strip()
        except Exception:
            previous = ""
        if previous and previous != current:
            if allow_overwrite:
                logger.warning(
                    "cfg-hash mismatch in %s (old=%s, new=%s) — overwriting "
                    "sentinel as requested. Old rows in this dir were "
                    "produced with the previous cfg.",
                    od, previous[:12], current[:12],
                )
                sentinel.write_text(current)
            else:
                raise CfgMismatch(
                    f"output_dir {od} was last written by a DIFFERENT cfg "
                    f"(old hash {previous[:12]}, new hash {current[:12]}).\n"
                    f"Either delete the dir to start fresh, OR pass "
                    f"--resume-anyway / set RESUME_ANYWAY=1 to overwrite "
                    f"the sentinel and continue appending."
                )
        # Same hash → no-op.
        return
    sentinel.write_text(current)

