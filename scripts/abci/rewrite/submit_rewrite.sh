#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -N 0356_rewrite
#PBS -l select=1:ngpus=8:ncpus=64
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HF
#PBS -m n
#
# Submit (RTYPE must always be included when using -v):
#   qsub -v RTYPE=rt_HF scripts/abci/rewrite/submit_rewrite.sh
#   qsub -v RTYPE=rt_HF,SPLIT=test,METHOD=natural,CONDITIONS="LP-1 LP-2" scripts/abci/rewrite/submit_rewrite.sh
#
# Optional env vars (override via qsub -v):
#   GEN_MODEL   HF model ID or local path for the rewrite LLM (default: openai/gpt-oss-120b)
#   DATASET     dataset key in config.py                       (default: sciclaimeval)
#   SPLIT       dataset split: dev or test                     (default: dev)
#   METHOD      "natural", "inject", or "both"                 (default: both)
#   CONDITIONS  space-separated condition list                 (default: all)
#               e.g. CONDITIONS="LP-1 LP-2 D_1"
#   TP          tensor parallel size                           (default: 8)
#   PORT        vllm server port                               (default: 8000)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
cd "${PBS_O_WORKDIR}"
JOBID="${PBS_JOBID%%.*}"
mkdir -p logs
exec > "logs/rewrite-${JOBID}.out" 2> "logs/rewrite-${JOBID}.err"
echo "JOBID=${JOBID}"

set -euxo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source scripts/abci/common/setup.sh

GEN_MODEL="${GEN_MODEL:-openai/gpt-oss-120b}"
DATASET="${DATASET:-sciclaimeval}"
SPLIT="${SPLIT:-dev}"
METHOD="${METHOD:-both}"
PORT="${PORT:-8000}"

# Auto-read TP from config/models.yaml; env var passed via qsub -v takes precedence
_yaml_tp=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('config/models.yaml'))
print(cfg.get('${GEN_MODEL}', {}).get('tensor_parallel_size', 8))
")
TP="${TP:-${_yaml_tp}}"

echo "GEN_MODEL=${GEN_MODEL}  DATASET=${DATASET}  SPLIT=${SPLIT}  METHOD=${METHOD}  TP=${TP}  PORT=${PORT}"
[[ -n "${CONDITIONS:-}" ]] && echo "CONDITIONS=${CONDITIONS}"

# ---------------------------------------------------------------------------
# Start vllm serve
# ---------------------------------------------------------------------------
vllm serve "${GEN_MODEL}" \
  --tensor-parallel-size "${TP}" \
  --dtype bfloat16 \
  --port "${PORT}" \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  &> "logs/vllm-rewrite-${JOBID}.log" &
VLLM_PID=$!

trap 'echo "Stopping vllm (pid=${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null' EXIT

# ---------------------------------------------------------------------------
# Wait for server to be ready
# ---------------------------------------------------------------------------
echo "Waiting for vllm server on port ${PORT}..."
WAIT_SECS=0
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
  sleep 5
  WAIT_SECS=$((WAIT_SECS + 5))
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vllm server process died. Check logs/vllm-rewrite-${JOBID}.log" >&2
    exit 1
  fi
  if [[ "${WAIT_SECS}" -gt 3600 ]]; then
    echo "ERROR: timed out waiting for vllm server (${WAIT_SECS}s)" >&2
    exit 1
  fi
done
echo "vllm server ready after ${WAIT_SECS}s"

BASE_URL="http://localhost:${PORT}/v1"

# ---------------------------------------------------------------------------
# Run rewrites
# ---------------------------------------------------------------------------
if [[ "${METHOD}" == "natural" || "${METHOD}" == "both" ]]; then
  python -m core.run rewrite natural \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --gen-model "${GEN_MODEL}" \
    --base-url "${BASE_URL}" \
    --resume \
    ${CONDITIONS:+--conditions ${CONDITIONS}}
fi

if [[ "${METHOD}" == "inject" || "${METHOD}" == "both" ]]; then
  python -m core.run rewrite inject \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --gen-model "${GEN_MODEL}" \
    --base-url "${BASE_URL}" \
    --resume \
    ${CONDITIONS:+--conditions ${CONDITIONS}}
fi

echo "Rewrite complete."
