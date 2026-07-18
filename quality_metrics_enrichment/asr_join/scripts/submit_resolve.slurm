#!/bin/bash
# submit_resolve.slurm — conflict-resolution feeders (Task #8), one DEDICATED
# gemma server per language. Same self-contained pattern as submit_itn.slurm,
# but runs asr_join.main_resolve (parakeet-vs-reference WER/CER + LLM resolve)
# and the voxpopuli_resolve_<lang>.yaml configs → writes voxpopuli_resolve/.
#
# Env (pass via --export):
#   QME_DIR    abs path to quality_metrics_enrichment (required)
#   GEMMA_MAP  "lang:server_jobid" pairs (required), e.g. "en:123 fr:124 ..."
#   ITN_MODEL  served model name (default google/gemma-4-26B-A4B-it-$USER)
#
#SBATCH --job-name=resolve-feed
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.err

set -euo pipefail
: "${QME_DIR:?QME_DIR required}"
: "${GEMMA_MAP:?GEMMA_MAP required (lang:jobid pairs)}"
SERVED="${ITN_MODEL:-google/gemma-4-26B-A4B-it-${USER}}"
LOGD="/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw"

LANGNODES=""
for pair in ${GEMMA_MAP}; do
    L="${pair%%:*}"; JID="${pair##*:}"
    N=""
    for i in $(seq 1 120); do
        read -r T NN <<< "$(squeue -h -j "${JID}" -o '%T %N' 2>/dev/null || true)"
        [ "${T:-}" = "RUNNING" ] && [ -n "${NN:-}" ] && [ "${NN}" != "(null)" ] && { N="${NN}"; break; }
        sleep 20
    done
    [ -z "${N}" ] && { echo "ERROR: server ${JID} for ${L} never RUNNING" >&2; exit 1; }
    CODE=000
    for i in $(seq 1 120); do
        CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 -H "Authorization: Bearer EMPTY" "http://${N}:8080/health" 2>/dev/null || echo 000)
        [ "${CODE}" = "200" ] && break
        sleep 20
    done
    [ "${CODE}" = "200" ] || { echo "ERROR: server ${N} (${L}) not healthy" >&2; exit 1; }
    echo "${L}: server ${JID} -> ${N} (health ${CODE})"
    LANGNODES="${LANGNODES} ${L}=${N}"
done

srun --environment=quality-assesment-gh200 --chdir="${QME_DIR}" --kill-on-bad-exit=0 bash -c "
    set -uo pipefail
    export PYTHONPATH='${QME_DIR}':\${PYTHONPATH:-}
    export ITN_MODEL='${SERVED}'
    pids=''
    for kv in ${LANGNODES}; do
        L=\${kv%%=*}; N=\${kv##*=}
        echo \"[resolve] \$L -> http://\$N:8080\"
        ITN_API_BASE=\"http://\$N:8080\" python -u -m asr_join.main_resolve \
            --config '${QME_DIR}'/asr_join/egs/voxpopuli_resolve_\${L}.yaml \
            > '${LOGD}'/resolve_\${L}.log 2>&1 &
        pids=\"\${pids} \$!\"
        sleep 8
    done
    rc=0
    for p in \${pids}; do wait \"\$p\" || rc=1; done
    exit \${rc}
"
echo "=== all resolve feeders finished ==="
