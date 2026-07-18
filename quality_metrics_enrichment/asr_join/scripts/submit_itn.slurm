#!/bin/bash
# submit_itn.slurm — overnight ITN feeders, ONE DEDICATED gemma server per
# language (so each feeder gets a full DP=4 server instead of sharing one and
# bottlenecking). Self-contained batch job (survives ssh drops / overnight):
#   1. for each "lang:server_jobid" in GEMMA_MAP, discover that server's node
#      (wait RUNNING) + wait /health,
#   2. run all the per-language asr_join ITN feeders IN PARALLEL on this node,
#      each pointed at ITS OWN server.
#
# Env (pass via --export):
#   QME_DIR    abs path to quality_metrics_enrichment (required)
#   GEMMA_MAP  space-sep "lang:server_jobid" pairs, e.g.
#              "en:2621900 fr:2621901 es:2621902 it:2621903"  (required)
#   ITN_MODEL  served model name (default google/gemma-4-26B-A4B-it-$USER)
#
#SBATCH --job-name=itn-feed
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

# 1) resolve each language's dedicated server node (wait RUNNING + /health)
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

# 2) run all feeders in parallel, each against its own dedicated server
srun --environment=quality-assesment-gh200 --chdir="${QME_DIR}" --kill-on-bad-exit=0 bash -c "
    set -uo pipefail
    export PYTHONPATH='${QME_DIR}':\${PYTHONPATH:-}
    export ITN_MODEL='${SERVED}'
    export RESUME_ANYWAY=1
    pids=''
    for kv in ${LANGNODES}; do
        L=\${kv%%=*}; N=\${kv##*=}
        echo \"[itn] \$L -> http://\$N:8080\"
        ITN_API_BASE=\"http://\$N:8080\" python -u -m asr_join.main \
            --config '${QME_DIR}'/asr_join/egs/voxpopuli_\${L}.yaml \
            > '${LOGD}'/itn_\${L}.log 2>&1 &
        pids=\"\${pids} \$!\"
        sleep 8
    done
    rc=0
    for p in \${pids}; do wait \"\$p\" || rc=1; done
    exit \${rc}
"
echo "=== all ITN feeders finished ==="
