#!/bin/bash
# launch.sh — run the offline merge + ROVER on the interactive node.
#
# No SLURM submit. asr_join is CPU-only and fits on the interactive node
# alongside the asr_vllm feeder. Reads the 3 ASR runners' outputs and
# writes <merge_output_dir>[/<lang>]/asr_agg/rover/merged.jsonl.
#
# ITN endpoint resolution (in order of precedence):
#   1. --itn-base URL + --itn-model NAME   (CLI flags)
#   2. ITN_API_BASE + ITN_MODEL env        (your shell exports)
#   3. Auto-discovery via squeue           (scan for newest sml_*gemma* job)
#   4. Pre-set VLLM_API_BASE + VLLM_MODEL  (last-resort fallback)
#
# Flags:
#   --shutdown-itn-on-exit   scancel any sml_*gemma* job NOT present when
#                            this script started — frees the cluster
#                            after a one-shot join run.
#
# Usage:
#   bash asr_join/scripts/launch.sh
#   bash asr_join/scripts/launch.sh --config cv_multilang.yaml
#   bash asr_join/scripts/launch.sh --itn-base http://nidNNN:8080 --itn-model gemma…
#   bash asr_join/scripts/launch.sh --shutdown-itn-on-exit

set -euo pipefail

CONFIG_NAME="cv_fr.yaml"
EXPLICIT_ITN_BASE="${ITN_API_BASE:-}"
EXPLICIT_ITN_MODEL="${ITN_MODEL:-}"
SHUTDOWN_ITN_ON_EXIT=0
# Pattern matched against squeue job names for auto-discovery. Override
# with --itn-pattern if your ITN model isn't a Gemma variant.
ITN_PATTERN="${ITN_PATTERN:-gemma}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)             CONFIG_NAME="$2"; shift 2 ;;
        --itn-base)           EXPLICIT_ITN_BASE="$2"; shift 2 ;;
        --itn-model)          EXPLICIT_ITN_MODEL="$2"; shift 2 ;;
        --itn-pattern)        ITN_PATTERN="$2"; shift 2 ;;
        --shutdown-itn-on-exit) SHUTDOWN_ITN_ON_EXIT=1; shift ;;
        -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${QME_DIR}/../../../.." && pwd)"

CONFIG_PATH="${QME_DIR}/asr_join/egs/${CONFIG_NAME}"
[[ -f "${CONFIG_PATH}" ]] || { echo "ERROR: config not found: ${CONFIG_PATH}" >&2; exit 1; }

# ----------------------------------------------------------------------
# ITN endpoint resolution.
# ----------------------------------------------------------------------
if [[ -z "${EXPLICIT_ITN_BASE}" ]]; then
    echo "[asr_join] auto-discovering ITN endpoint (squeue pattern: ${ITN_PATTERN}) ..."
    # Pick the NEWEST RUNNING sml job that matches the pattern. Sorting by
    # JobID descending ensures a freshly-launched server wins over a stale
    # one — important when running multiple pipelines that rotate vLLMs.
    LINE=$(squeue -u "$USER" -h -o "%T %N %j %i" 2>/dev/null \
           | grep -Ei "sml_.*${ITN_PATTERN}" \
           | awk '$1 == "RUNNING" && $2 ~ /^nid/' \
           | sort -k4,4 -n -r \
           | head -1)
    if [[ -z "${LINE}" ]]; then
        echo "ERROR: no RUNNING sml_*${ITN_PATTERN}* job found." >&2
        echo "       Either launch one (asr_join/scripts/vllm_itn_launcher.sh)" >&2
        echo "       OR set ITN_API_BASE + ITN_MODEL env (or --itn-base + --itn-model)." >&2
        exit 1
    fi
    NID=$(echo "${LINE}" | awk '{print $2}')
    EXPLICIT_ITN_BASE="http://${NID}:8080"
    # IMPORTANT: sml mangles --served-model-name into the SLURM job name
    # (replaces "/" with "_" and adds a random suffix), so the job name
    # is NOT the actual served-model-name. Ask the vLLM /v1/models
    # endpoint for the real id — that's the canonical source. Falls back
    # to env/YAML if the endpoint is unreachable (we'll fail loudly later
    # on the first chat request anyway).
    if [[ -z "${EXPLICIT_ITN_MODEL}" ]]; then
        EXPLICIT_ITN_MODEL=$(curl -s -H "Authorization: Bearer EMPTY" \
                                  --max-time 10 \
                                  "${EXPLICIT_ITN_BASE}/v1/models" \
                              | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" \
                              2>/dev/null || true)
        if [[ -z "${EXPLICIT_ITN_MODEL}" ]]; then
            echo "[asr_join] WARN: /v1/models on ${EXPLICIT_ITN_BASE} not yet ready — model name unknown." >&2
            echo "           Either wait for the vLLM to finish loading, or set --itn-model explicitly." >&2
        fi
    fi
    echo "[asr_join] using ITN endpoint: ${EXPLICIT_ITN_BASE}  model=${EXPLICIT_ITN_MODEL}"
fi
export ITN_API_BASE="${EXPLICIT_ITN_BASE}"
export ITN_MODEL="${EXPLICIT_ITN_MODEL}"

# ----------------------------------------------------------------------
# --shutdown-itn-on-exit setup: snapshot sml jobs at start; on exit,
# scancel any new sml jobs that match the ITN pattern.
# (Only fires when we launched the server ourselves. If user just
# auto-discovered an existing one, the snapshot includes it → not killed.)
# ----------------------------------------------------------------------
PREEXISTING_SML_JOBS=""
cleanup() {
    if [[ "${SHUTDOWN_ITN_ON_EXIT}" -eq 1 ]]; then
        # Find current matching jobs; scancel any that weren't there at start.
        CURRENT=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
                  | grep -Ei "sml_.*${ITN_PATTERN}" \
                  | awk '{print $1}')
        for JID in ${CURRENT}; do
            if ! echo "${PREEXISTING_SML_JOBS}" | grep -qw "${JID}"; then
                echo "[asr_join] --shutdown-itn-on-exit: scancel ${JID}" >&2
                scancel "${JID}" 2>/dev/null || true
            fi
        done
    fi
}
if [[ "${SHUTDOWN_ITN_ON_EXIT}" -eq 1 ]]; then
    PREEXISTING_SML_JOBS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
                           | grep -Ei "sml_.*${ITN_PATTERN}" \
                           | awk '{print $1}')
    trap cleanup EXIT
fi

cd "${QME_DIR}"
export PYTHONPATH="${QME_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
python -m asr_join.main --config "${CONFIG_PATH}"
