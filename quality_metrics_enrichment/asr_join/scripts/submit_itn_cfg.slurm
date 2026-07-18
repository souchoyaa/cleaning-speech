#!/bin/bash
# submit_itn_cfg.slurm — ONE dedicated feeder node for an asr_join ITN run, fed
# by a separate dedicated vLLM (gemma) server. Generalised submit_itn.slurm:
# config-agnostic. Health is checked FROM THIS feeder node (not login) so the
# feed only starts once the feeder can actually reach the server.
# Env (--export): QME_DIR, GEMMA_JOBID (the vLLM sml job id), CONFIG (asr_join yaml).
#SBATCH --job-name=itn-cfg-feed
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --output=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.out
#SBATCH --error=/capstor/scratch/cscs/sgodey/data_selection_runs/quality/slurm_raw/%x_%j.err
set -euo pipefail
: "${QME_DIR:?}"; : "${GEMMA_JOBID:?}"; : "${CONFIG:?}"
ITN_MODEL="${ITN_MODEL:-google/gemma-4-26B-A4B-it-${USER}}"   # must match the vLLM --served-model-name
N=""
for i in $(seq 1 120); do
    read -r T NN <<< "$(squeue -h -j "${GEMMA_JOBID}" -o '%T %N' 2>/dev/null || true)"
    [ "${T:-}" = "RUNNING" ] && [ -n "${NN:-}" ] && [ "${NN}" != "(null)" ] && { N="${NN}"; break; }
    sleep 20
done
[ -z "${N}" ] && { echo "ERROR: gemma ${GEMMA_JOBID} never RUNNING" >&2; exit 1; }
CODE=000
for i in $(seq 1 150); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 -H "Authorization: Bearer EMPTY" "http://${N}:8080/health" 2>/dev/null || echo 000)
    [ "${CODE}" = "200" ] && break
    sleep 20
done
[ "${CODE}" = "200" ] || { echo "ERROR: gemma ${N} not healthy FROM FEEDER (code ${CODE})" >&2; exit 1; }
# /health can return 200 before the engine finishes loading (esp. the 26B). Gate
# on a REAL chat completion (also verifies the served model name) so the feed
# never starts against an engine that isn't actually serving.
OK=0
for i in $(seq 1 90); do
    R=$(curl -s --max-time 30 -H "Authorization: Bearer EMPTY" -H "Content-Type: application/json" \
        "http://${N}:8080/v1/chat/completions" \
        -d "{\"model\":\"${ITN_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" 2>/dev/null || true)
    echo "${R}" | grep -q '"choices"' && { OK=1; echo "engine serving (test chat OK) on ${N}"; break; }
    sleep 20
done
[ "${OK}" = "1" ] || { echo "ERROR: ${N} (model ${ITN_MODEL}) not serving chat completions after wait" >&2; exit 1; }
echo "gemma ${N} serving — running ITN: ${CONFIG}"
srun --environment=quality-assesment-gh200 --chdir="${QME_DIR}" --kill-on-bad-exit=0 bash -c "
    export PYTHONPATH='${QME_DIR}':\${PYTHONPATH:-}
    export ITN_API_BASE=http://${N}:8080
    export ITN_MODEL='${ITN_MODEL}'
    export RESUME_ANYWAY=1
    python -u -m asr_join.main --config '${CONFIG}'
"
echo "=== ITN feeder finished ==="
