#!/bin/bash
# status.sh — single-shot view of pipeline state.
#
# Reads the run_history.jsonl ledger ($OUT_BASE/logs/run_history.jsonl by
# default; override via RUN_HISTORY_LEDGER env). Groups events by (yaml,
# stage, language), pairs start/end, and prints a table.
#
# Also queries `squeue -u $USER` so you see what's currently running.
#
# Usage:
#   bash pipeline/scripts/status.sh                  # all stages
#   bash pipeline/scripts/status.sh --yaml cv_fr     # filter by yaml substring
#   bash pipeline/scripts/status.sh --stage canary   # filter by stage

set -euo pipefail

FILTER_YAML=""
FILTER_STAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaml)  FILTER_YAML="$2";  shift 2 ;;
        --stage) FILTER_STAGE="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

LEDGER="${RUN_HISTORY_LEDGER:-${OUT_BASE:-/capstor/scratch/cscs/sgodey/data_selection_runs/quality}/logs/run_history.jsonl}"

if [[ ! -f "${LEDGER}" ]]; then
    echo "No ledger at ${LEDGER} (yet) — run something with run_history wiring first."
    echo
fi

# Live SLURM jobs first — at-a-glance.
echo "=== live SLURM jobs ==="
squeue -u "$USER" --noheader -o "  %i  %.10P  %.18j  %.8T  %.10M  %.10L  %R" 2>/dev/null \
    | head -40 || echo "  (squeue unavailable)"
echo

# Ledger-grouped status table — done with Python for sane joins/sorting.
[[ -f "${LEDGER}" ]] || exit 0

echo "=== pipeline runs (from ${LEDGER}) ==="
python3 - "$LEDGER" "$FILTER_YAML" "$FILTER_STAGE" <<'PYEOF'
import json, sys, collections

ledger_path, yaml_filter, stage_filter = sys.argv[1], sys.argv[2], sys.argv[3]

# Read all events. The ledger is small (few thousand records typical),
# so just slurp it.
events = []
with open(ledger_path, "rb") as f:
    for raw in f:
        if not raw.strip():
            continue
        try:
            events.append(json.loads(raw))
        except Exception:
            continue

# Pair start/end by run_id. A start without an end means the run is
# either still going or got SIGKILLED before the finally block.
by_run = collections.defaultdict(dict)
for e in events:
    rid = e.get("run_id")
    if not rid:
        continue
    by_run[rid][e.get("event")] = e

# Group rows by (yaml, stage, language). Show counts of each status.
groups = collections.defaultdict(lambda: {
    "ok": 0, "crashed": 0, "terminated": 0, "running": 0,
    "last_n_committed": 0, "last_wall_s": 0.0,
    "last_reason": None, "last_ts": None,
    "last_job_id": None,
})

for rid, parts in by_run.items():
    s = parts.get("start") or {}
    e = parts.get("end")
    yaml = s.get("yaml", "?")
    stage = s.get("stage", "?")
    lang = (e or s).get("language") or "—"

    if yaml_filter and yaml_filter not in yaml:
        continue
    if stage_filter and stage_filter != stage:
        continue

    key = (yaml, stage, lang)
    g = groups[key]
    if e is None:
        g["running"] += 1
        continue
    status = e.get("status", "?")
    if status in g:
        g[status] += 1
    n_committed = (
        e.get("n_ok") if e.get("n_ok") is not None else e.get("n_total", 0)
    ) or 0
    if n_committed >= g["last_n_committed"]:
        g["last_n_committed"] = n_committed
        g["last_wall_s"] = e.get("wall_seconds", 0)
        g["last_ts"] = e.get("ts")
        g["last_job_id"] = s.get("slurm_job_id")
        if e.get("reason"):
            g["last_reason"] = e["reason"][:80]

# Print as a fixed-width table — single-pass since the data is small.
print(f"  {'YAML':<45} {'STAGE':<10} {'LANG':<6} {'RUNS':>4}  STATUS                LAST_ROWS  LAST_WALL  LAST_JOB")
print(f"  {'-'*45} {'-'*10} {'-'*6} {'-'*4}  {'-'*22} {'-'*9} {'-'*10} {'-'*10}")
for (yaml, stage, lang), g in sorted(groups.items()):
    total = g["ok"] + g["crashed"] + g["terminated"] + g["running"]
    status_parts = []
    if g["ok"]:         status_parts.append(f"\033[32m✓ok={g['ok']}\033[0m")
    if g["running"]:    status_parts.append(f"\033[33m…run={g['running']}\033[0m")
    if g["crashed"]:    status_parts.append(f"\033[31m✗cr={g['crashed']}\033[0m")
    if g["terminated"]: status_parts.append(f"\033[36m■tm={g['terminated']}\033[0m")
    status_str = " ".join(status_parts) or "?"
    wall = g["last_wall_s"] or 0
    wall_str = f"{wall/3600:.1f}h" if wall >= 3600 else f"{wall/60:.0f}m" if wall >= 60 else f"{wall:.0f}s"
    yaml_short = yaml[-45:] if len(yaml) > 45 else yaml
    print(f"  {yaml_short:<45} {stage:<10} {lang:<6} {total:>4}  {status_str:<30} "
          f"{g['last_n_committed']:>9} {wall_str:>10} {g['last_job_id'] or '—':>10}")

# Most recent crash reasons (if any) for quick triage.
crashes = [
    (parts["end"].get("ts"), parts["end"].get("yaml") or parts.get("start", {}).get("yaml"),
     parts["end"].get("stage") or parts.get("start", {}).get("stage"),
     parts["end"].get("language"),
     parts["end"].get("reason", "?")[:120])
    for parts in by_run.values()
    if parts.get("end") and parts["end"].get("status") == "crashed"
]
if crashes:
    print()
    print("=== recent crashes (last 10) ===")
    for ts, y, st, lang, reason in sorted(crashes, reverse=True)[:10]:
        print(f"  {ts}  {st}/{lang or '—'}  {y}")
        print(f"      {reason}")
PYEOF
