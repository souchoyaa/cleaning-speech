#!/bin/bash
# preflight.sh — sanity-check a pipeline.yaml before submitting jobs.
#
# Catches the common "wasted 4h SLURM allocation because of a typo"
# failure modes BEFORE we burn compute:
#   - shar_dir exists, looks like a Lhotse Shar (has cuts.NNN.jsonl.gz)
#   - output_dir parent exists + writable, has at least <min_gb> free
#   - sml binary reachable (we must be on a login node)
#   - SLURM reservation valid (scontrol show reservation succeeds)
#   - vLLM model paths exist on capstor (for the launchers we'll invoke)
#
# Exits non-zero on any failure; greps a clear ✗ for each failed check.
#
# Usage:
#   bash pipeline/scripts/preflight.sh
#   bash pipeline/scripts/preflight.sh --config cv_fr.yaml
#   bash pipeline/scripts/preflight.sh --config cv_fr.yaml --min-gb 50

set -euo pipefail

CONFIG_NAME="cv_fr.yaml"
MIN_GB=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --min-gb) MIN_GB="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_CFG="${QME_DIR}/pipeline/egs/${CONFIG_NAME}"
[[ -f "${PIPELINE_CFG}" ]] || { echo "✗ config not found: ${PIPELINE_CFG}" >&2; exit 1; }

# Use Python for the YAML reads; bash + yq is a portability nightmare.
PY() { python3 -c "$@"; }

PASS=0
FAIL=0
ok()   { printf '  \033[32m✓\033[0m %s\n'                "$1"; PASS=$((PASS+1)); }
fail() { printf '  \033[31m✗\033[0m %s\n        \033[33m%s\033[0m\n' "$1" "$2"; FAIL=$((FAIL+1)); }

echo "=== preflight: ${CONFIG_NAME} ==="

# ---- Login-node sanity --------------------------------------------------
HOST=$(hostname)
case "$HOST" in
    nid*)
        fail "running on compute node ($HOST)"  \
             "sml only works on login nodes — exit salloc and rerun from clariden-ln*"
        ;;
    *)
        ok "login node ($HOST)"
        ;;
esac

# ---- sml binary ---------------------------------------------------------
if command -v sml >/dev/null 2>&1; then
    ok "sml binary in PATH ($(command -v sml))"
else
    fail "sml binary NOT in PATH" \
         "install swiss-ai-model-launch (uv tool install swiss-ai-model-launch) or add to PATH"
fi

# ---- Pipeline YAML structure -------------------------------------------
STAGES=$(PY "
import yaml
cfg = yaml.safe_load(open('${PIPELINE_CFG}').read())
print(' '.join(cfg.get('stages', {}).keys()))
" 2>/dev/null) || { fail "pipeline YAML unparseable" "fix YAML syntax in ${PIPELINE_CFG}"; STAGES=""; }
ok "pipeline YAML parses: stages=[${STAGES}]"

# ---- Per-stage config validation ---------------------------------------
for STAGE in ${STAGES}; do
    case "${STAGE}" in
      parakeet|canary|qwen) ;;
      *) fail "unknown stage '${STAGE}'" "valid: parakeet, canary, qwen"; continue ;;
    esac

    # Collect this stage's config path(s) — qwen may have 'configs: [a, b]'
    PATHS=$(PY "
import yaml
cfg = yaml.safe_load(open('${PIPELINE_CFG}').read())['stages'].get('${STAGE}') or {}
out = cfg.get('configs') or ([cfg['config']] if cfg.get('config') else [])
for p in out: print(p)
") || PATHS=""

    [[ -z "${PATHS}" ]] && { fail "${STAGE}: no config(s) declared" "set stages.${STAGE}.config or .configs"; continue; }

    while IFS= read -r REL; do
        CFG_PATH="${QME_DIR}/${REL}"
        if [[ ! -f "${CFG_PATH}" ]]; then
            fail "${STAGE}: cfg missing → ${CFG_PATH}" "create the file or fix the path in ${PIPELINE_CFG}"
            continue
        fi
        ok "${STAGE} cfg exists: ${REL}"

        # Validate shar_dir + output_dir from the stage config
        SHAR_DIR=$(PY "
import yaml; cfg = yaml.safe_load(open('${CFG_PATH}').read())
print(cfg.get('shar_dir') or cfg.get('language_split_dir') or '')
")
        OUT_DIR=$(PY "
import yaml; cfg = yaml.safe_load(open('${CFG_PATH}').read())
print(cfg.get('output_dir') or '')
")
        if [[ -n "${SHAR_DIR}" ]]; then
            if [[ -d "${SHAR_DIR}" ]]; then
                # Lhotse Shar marker — at least one cuts.NNN.jsonl.gz
                N_CUTS=$(find "${SHAR_DIR}" -maxdepth 2 -name 'cuts.*.jsonl.gz' 2>/dev/null | head -3 | wc -l | tr -d ' ')
                if [[ "${N_CUTS}" -gt 0 ]]; then
                    ok "${STAGE}/${REL##*/}: shar_dir has Lhotse shards (${SHAR_DIR})"
                else
                    fail "${STAGE}: shar_dir has no cuts.*.jsonl.gz" \
                         "${SHAR_DIR} exists but doesn't look like a Lhotse Shar"
                fi
            else
                fail "${STAGE}: shar_dir does not exist" \
                     "expected ${SHAR_DIR}"
            fi
        fi

        if [[ -n "${OUT_DIR}" ]]; then
            OUT_PARENT="$(dirname "${OUT_DIR}")"
            if [[ ! -d "${OUT_PARENT}" ]]; then
                fail "${STAGE}: output_dir parent missing" \
                     "mkdir -p ${OUT_PARENT}"
            elif [[ ! -w "${OUT_PARENT}" ]]; then
                fail "${STAGE}: output_dir parent not writable" \
                     "${OUT_PARENT}"
            else
                # Free space check on the parent's filesystem
                FREE_GB=$(df -BG "${OUT_PARENT}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
                if [[ -n "${FREE_GB}" && "${FREE_GB}" -lt "${MIN_GB}" ]]; then
                    fail "${STAGE}: only ${FREE_GB}G free at ${OUT_PARENT} (< ${MIN_GB}G)" \
                         "free space or set output_dir elsewhere"
                else
                    ok "${STAGE}/${REL##*/}: output_dir parent OK, ${FREE_GB:-?}G free"
                fi
            fi
        fi
    done <<< "${PATHS}"
done

# ---- SLURM reservation in submit.slurm files ---------------------------
# Pull reservation from each component's submit.slurm and validate it.
RESV_SEEN=()
for SUBMIT in "${QME_DIR}/asr_parakeet/scripts/submit.slurm" \
              "${QME_DIR}/asr_canary/scripts/submit.slurm"; do
    [[ -f "${SUBMIT}" ]] || continue
    RESV=$(grep -E '^#SBATCH\s+--reservation=' "${SUBMIT}" | head -1 | sed 's/.*--reservation=//;s/[[:space:]].*//')
    [[ -z "${RESV}" ]] && continue
    [[ " ${RESV_SEEN[*]} " == *" ${RESV} "* ]] && continue
    RESV_SEEN+=("${RESV}")
    if scontrol show reservation "${RESV}" >/dev/null 2>&1; then
        ok "SLURM reservation valid: ${RESV} ($(basename "${SUBMIT}"))"
    else
        fail "SLURM reservation invalid: ${RESV}" \
             "fix #SBATCH --reservation=… in ${SUBMIT}"
    fi
done

# ---- vLLM model weight paths (if vllm_qwen3_asr_launcher.sh is involved) -
for LAUNCHER in "${QME_DIR}/asr_vllm/scripts/vllm_qwen3_asr_launcher.sh" \
                "${QME_DIR}/asr_join/scripts/vllm_itn_launcher.sh"; do
    [[ -f "${LAUNCHER}" ]] || continue
    MODEL_PATH=$(grep -E '^VLLM_MODEL_PATH=' "${LAUNCHER}" | head -1 | sed 's/.*:-//;s/}.*//')
    [[ -z "${MODEL_PATH}" ]] && continue
    if [[ -d "${MODEL_PATH}" ]]; then
        ok "vLLM model weights present: $(basename "${MODEL_PATH}")"
    else
        fail "vLLM model weights MISSING: ${MODEL_PATH}" \
             "set VLLM_MODEL_PATH=… or download the model"
    fi
done

# ---- Summary ------------------------------------------------------------
echo
if [[ "${FAIL}" -eq 0 ]]; then
    printf '\033[32m=== preflight: %d/%d checks passed ===\033[0m\n' \
           "${PASS}" "$((PASS + FAIL))"
    exit 0
else
    printf '\033[31m=== preflight: %d FAILURE(S), %d passed ===\033[0m\n' \
           "${FAIL}" "${PASS}"
    echo "Fix the ✗ items above before running the pipeline."
    exit 1
fi
