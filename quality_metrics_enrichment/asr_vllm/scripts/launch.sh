#!/bin/bash
# launch.sh — run the vLLM feeder on the interactive node, with three
# launch modes:
#
#   (A) NEW SERVER     — start a vLLM server via sml advanced, wait for
#                        it to come up, then feed it.
#   (B) EXISTING SML   — sml job is already RUNNING (e.g. submitted in
#                        an earlier session). Auto-discover its URL via
#                        squeue, skip the launch + wait-for-RUNNING.
#                        Triggered by --skip-vllm-launch.
#   (C) DIRECT URL     — you already have a vLLM URL (sml or otherwise).
#                        No squeue lookup at all. Triggered by either
#                        --api-base flag or a pre-set $VLLM_API_BASE.
#
# Usage:
#   bash asr_vllm/scripts/launch.sh                                       # mode A (default config)
#   bash asr_vllm/scripts/launch.sh --config cv_fr_qwen.yaml              # mode A, Qwen
#   bash asr_vllm/scripts/launch.sh --skip-vllm-launch                    # mode B
#   bash asr_vllm/scripts/launch.sh --api-base http://nid006249:5000      # mode C
#   bash asr_vllm/scripts/launch.sh --api-base http://nidA:5000,http://nidB:5000  # mode C, multi-replica
#   VLLM_API_BASE=http://nid006249:5000 bash asr_vllm/scripts/launch.sh  # mode C via env
#
# Mode C is the "no server management" path: the script does not touch
# sml, does not look at squeue, and exits immediately if /health fails.

set -euo pipefail

CONFIG_NAME="cv_fr_voxtral.yaml"
PORT="5000"
SKIP_LAUNCH=0
SHUTDOWN_VLLM_ON_EXIT=0
WAIT_TIMEOUT_SEC=180
HTTP_TIMEOUT_SEC=600
LAUNCH_TIME=""
EXPLICIT_API_BASE="${VLLM_API_BASE:-}"   # --reuse-vllm wins; --api-base overrides

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG_NAME="$2"; shift 2 ;;
        --port)             PORT="$2";        shift 2 ;;
        # --reuse-vllm is the preferred name; --skip-vllm-launch is the
        # original spelling, kept as an alias for back-compat.
        --reuse-vllm|--skip-vllm-launch) SKIP_LAUNCH=1; shift ;;
        --shutdown-vllm-on-exit) SHUTDOWN_VLLM_ON_EXIT=1; shift ;;
        --api-base)         EXPLICIT_API_BASE="$2"; shift 2 ;;
        --wait)             WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
        --time)             LAUNCH_TIME="$2";      shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${QME_DIR}/../../../.." && pwd)"
CONFIG="${QME_DIR}/asr_vllm/egs/${CONFIG_NAME}"
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 1; }

# Make python -m asr_vllm.scan / asr_vllm.main resolvable from anywhere.
export PYTHONPATH="${QME_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

# ----------------------------------------------------------------------
# Decide which mode we're in.
# ----------------------------------------------------------------------
MODE=""
if [[ "${SKIP_LAUNCH}" -eq 1 ]]; then
    # --skip-vllm-launch is the explicit "auto-discover via squeue" intent.
    # Ignore any pre-set VLLM_API_BASE so a stale export from a prior shell
    # session doesn't silently downgrade us to mode C with a wrong URL.
    if [[ -n "${EXPLICIT_API_BASE}" ]]; then
        echo "[launch] --skip-vllm-launch: ignoring stale VLLM_API_BASE=${EXPLICIT_API_BASE}" >&2
        EXPLICIT_API_BASE=""
    fi
    MODE="B"
elif [[ -n "${EXPLICIT_API_BASE}" ]]; then
    MODE="C"
else
    MODE="A"
fi
echo "[launch] mode=${MODE} config=${CONFIG_NAME}"

# ----------------------------------------------------------------------
# Background pre-scan — compute total audio duration in parallel with
# vLLM startup / health probe. Result lands in ${SCAN_OUT}; we collect
# it just before launching the feeder so it doesn't add wall time.
# Override workers via SCAN_WORKERS env (default 16).
# ----------------------------------------------------------------------
SCAN_OUT="$(mktemp -t asr_vllm_scan.XXXXXX)"
# SHUTDOWN_JOB_ID is set after mode A's squeue discovery finds the sml job
# we just launched. Cleanup scancels it iff --shutdown-vllm-on-exit was
# passed AND we were the one that started it (mode A only — mode B reuses
# a pre-existing sml job, which we never auto-cancel).
SHUTDOWN_JOB_ID=""
cleanup() {
    rm -f "${SCAN_OUT}"
    if [[ "${SHUTDOWN_VLLM_ON_EXIT}" -eq 1 && -n "${SHUTDOWN_JOB_ID}" ]]; then
        echo "[launch] --shutdown-vllm-on-exit: scancel ${SHUTDOWN_JOB_ID}" >&2
        scancel "${SHUTDOWN_JOB_ID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
SCAN_WORKERS="${SCAN_WORKERS:-16}"
echo "[launch] starting shar pre-scan (workers=${SCAN_WORKERS}) — runs in parallel."
( python -u -m asr_vllm.scan --config "${CONFIG}" --num-workers "${SCAN_WORKERS}" \
    > "${SCAN_OUT}" 2>&1 ) &
SCAN_PID=$!

# Pull model/launcher/port from the YAML, one value per line. Drop into
# YAML_MODEL/YAML_LAUNCHER/YAML_PORT via a `read` triple — no sed parsing.
# Also expands ${USER} in api_model so a portable YAML like
#   api_model: "Qwen/Qwen3-ASR-1.7B-${USER}"
# resolves to the actual served-model-name on this user's launch.
YAML_MODEL=""; YAML_LAUNCHER=""; YAML_PORT=""
{ read -r YAML_MODEL; read -r YAML_LAUNCHER; read -r YAML_PORT; } < <(python3 -c "
import os, sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
feeder = cfg.get('vllm_feeder') or {}
launcher = cfg.get('vllm_launcher') or {}
print((feeder.get('api_model') or '').replace('\${USER}', os.environ.get('USER', '')))
print(launcher.get('script', 'vllm_launcher.sh'))
print(launcher.get('port', 5000))
" "${CONFIG}" 2>/dev/null) || true

# Allow CLI / env overrides; the YAML is the source of truth otherwise.
export VLLM_MODEL="${VLLM_MODEL:-${YAML_MODEL:-}}"
LAUNCHER_SCRIPT="${VLLM_LAUNCHER_SCRIPT:-${YAML_LAUNCHER:-vllm_launcher.sh}}"
# Port: YAML > CLI default. Update the global PORT used downstream.
if [[ -n "${YAML_PORT:-}" && "${PORT}" == "5000" ]]; then
    PORT="${YAML_PORT}"
fi

# Some launchers (Qwen3-ASR) read VLLM_PORT to set both `--worker-port`
# and the framework's `--port`. Export it consistently.
export VLLM_PORT="${PORT}"
# Same for served-name in launchers that don't want to recompute it.
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-${VLLM_MODEL}}"

echo "[launch] VLLM_MODEL=${VLLM_MODEL}"
echo "[launch] launcher=${LAUNCHER_SCRIPT}  port=${PORT}"

# ----------------------------------------------------------------------
# Mode A — start a new vLLM server using the YAML-selected launcher.
# ----------------------------------------------------------------------
if [[ "${MODE}" == "A" ]]; then
    [[ -f "${SCRIPT_DIR}/${LAUNCHER_SCRIPT}" ]] \
        || { echo "ERROR: launcher script not found: ${SCRIPT_DIR}/${LAUNCHER_SCRIPT}" >&2; exit 1; }
    echo "[1/3] Starting vLLM via ${LAUNCHER_SCRIPT} ..."
    if [[ -n "${LAUNCH_TIME}" ]]; then
        ( cd "${SCRIPT_DIR}" && VLLM_TIME="${LAUNCH_TIME}" bash "${LAUNCHER_SCRIPT}" )
    else
        ( cd "${SCRIPT_DIR}" && bash "${LAUNCHER_SCRIPT}" )
    fi
fi

# ----------------------------------------------------------------------
# Modes A + B — discover the URL via squeue.
# Mode C — skip this entirely.
# ----------------------------------------------------------------------
VLLM_API_BASE=""
if [[ "${MODE}" == "A" || "${MODE}" == "B" ]]; then
    echo "[2/3] Waiting for sml_* job to RUNNING (timeout ${WAIT_TIMEOUT_SEC}s) ..."
    NODELIST=""
    SML_JOB=""
    DEADLINE=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
    while (( $(date +%s) < DEADLINE )); do
        LINE=$(squeue -u "$USER" -h -o "%T %N %j %i" 2>/dev/null \
               | grep -Ei 'sml_' \
               | awk '$1 == "RUNNING" && $2 ~ /^nid/' \
               | sort -k4,4 -n -r \
               | head -1)
        if [[ -n "${LINE}" ]]; then
            NODELIST=$(echo "${LINE}" | awk '{print $2}')
            SML_JOB=$(echo "${LINE}" | awk '{print $4}')
            N_HOSTS=$(scontrol show hostnames "${NODELIST}" | wc -l)
            echo "  found: ${NODELIST} (${N_HOSTS} nodes, jobid=${SML_JOB})"
            break
        fi
        sleep 5
    done
    if [[ -z "${NODELIST}" ]]; then
        echo "ERROR: no RUNNING sml_* job appeared within ${WAIT_TIMEOUT_SEC}s." >&2
        if [[ "${MODE}" == "B" ]]; then
            echo "       --reuse-vllm was set but no server is running. Either" >&2
            echo "       drop --reuse-vllm (mode A) or pass --api-base URL (mode C)." >&2
        fi
        exit 1
    fi
    # If we started it (mode A), record for optional --shutdown-vllm-on-exit.
    if [[ "${MODE}" == "A" ]]; then
        SHUTDOWN_JOB_ID="${SML_JOB}"
    fi
    URLS=""
    while IFS= read -r host; do
        [[ -z "${host}" ]] && continue
        [[ -n "${URLS}" ]] && URLS+=","
        URLS+="http://${host}:${PORT}"
    done < <(scontrol show hostnames "${NODELIST}")
    VLLM_API_BASE="${URLS}"
else
    VLLM_API_BASE="${EXPLICIT_API_BASE}"
    echo "[2/3] Using provided API base (no squeue lookup)."
fi

echo "[launch] VLLM_API_BASE=${VLLM_API_BASE}"
export VLLM_API_BASE

# ----------------------------------------------------------------------
# Health probe — always run (catches typos in --api-base + slow cold start).
# Mode C trusts the user's URL; if it's wrong, fail fast (15s) rather
# than waiting out the full cold-start budget.
# ----------------------------------------------------------------------
HEALTH_TIMEOUT_SEC="${HTTP_TIMEOUT_SEC}"
if [[ "${MODE}" == "C" ]]; then
    HEALTH_TIMEOUT_SEC=15
fi
echo "[launch] probing /health on each worker (timeout ${HEALTH_TIMEOUT_SEC}s) ..."
HEALTH_DEADLINE=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))
HEALTH_BACKOFF=2
IFS=',' read -ra WORKER_BASE_ARR <<< "${VLLM_API_BASE}"
N_WORKERS=${#WORKER_BASE_ARR[@]}
HEALTH_OK=0
LAST=""
while (( $(date +%s) < HEALTH_DEADLINE )); do
    OK=0
    for BASE in "${WORKER_BASE_ARR[@]}"; do
        for U in "/health" "/v1/models"; do
            CODE=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 5 \
                   -H "Authorization: Bearer EMPTY" "${BASE}${U}" 2>/dev/null || echo 000)
            if [[ "${CODE}" == "200" ]]; then OK=$((OK+1)); break; fi
            LAST="${BASE}${U}=${CODE}"
        done
    done
    if [[ "${OK}" -eq "${N_WORKERS}" ]]; then
        echo "  all ${N_WORKERS} vLLM worker(s) serving (HTTP 200)"
        HEALTH_OK=1
        break
    fi
    sleep "${HEALTH_BACKOFF}"
    HEALTH_BACKOFF=$(( HEALTH_BACKOFF < 15 ? HEALTH_BACKOFF + 2 : 15 ))
done
if [[ "${HEALTH_OK}" -ne 1 ]]; then
    echo "ERROR: /health did not return 200 on all workers within ${HEALTH_TIMEOUT_SEC}s (last: ${LAST})." >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Collect the pre-scan result (may have already finished while we
# waited for vLLM; if not, give it up to a minute more). Failures are
# non-fatal — feeder runs without ETA.
# ----------------------------------------------------------------------
echo "[launch] waiting for pre-scan to finish ..."
if wait "${SCAN_PID}" 2>/dev/null; then
    SCAN_RC=0
else
    SCAN_RC=$?
fi
# Always show what the scan emitted (progress + final line).
if [[ -s "${SCAN_OUT}" ]]; then
    sed 's/^/[scan] /' "${SCAN_OUT}" >&2
fi
TOTAL_SECONDS=$(grep -oE 'TOTAL_SECONDS=[0-9.]+' "${SCAN_OUT}" | tail -1 | cut -d= -f2)
if [[ -n "${TOTAL_SECONDS}" && "${TOTAL_SECONDS}" != "0" && "${TOTAL_SECONDS}" != "0.0" ]]; then
    export ASR_VLLM_TOTAL_SECONDS="${TOTAL_SECONDS}"
    HOURS_FMT=$(awk -v s="${TOTAL_SECONDS}" 'BEGIN{printf "%.1f", s/3600}')
    echo "[launch] ASR_VLLM_TOTAL_SECONDS=${TOTAL_SECONDS}  (${HOURS_FMT}h — ETA enabled)"
else
    echo "[launch] WARN: pre-scan produced no usable total (rc=${SCAN_RC}). Feeder will run without ETA." >&2
fi

# ----------------------------------------------------------------------
# Run the feeder. Stays in the foreground so logs land in the terminal
# you launched from. Ctrl+C exits cleanly (SIGINT handled by main.py).
# We deliberately do NOT exec — the trap above needs to fire after the
# feeder exits so --shutdown-vllm-on-exit can scancel the sml job.
# ----------------------------------------------------------------------
echo "[3/3] Running feeder: python -m asr_vllm.main --config ${CONFIG_NAME}"
cd "${QME_DIR}"
python -u -m asr_vllm.main --config "${CONFIG}"
