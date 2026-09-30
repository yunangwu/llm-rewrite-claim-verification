#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -N 0356_infer
#PBS -l select=1:ngpus=8:ncpus=64
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HF
#PBS -m n
#
# Submit (RTYPE must always be included when using -v):
#   qsub -v RTYPE=rt_HF scripts/abci/infer/submit_infer.sh
#   qsub -v RTYPE=rt_HF,MODEL=Qwen/Qwen3-VL-8B-Instruct,CONDITIONS="original LP-1" scripts/abci/infer/submit_infer.sh
#
# Optional env vars (override via qsub -v):
#   MODEL            HF model ID or local path       (default: Qwen/Qwen3-VL-8B-Instruct)
#   DATASET          dataset key in config.py         (default: sciclaimeval)
#   SPLIT            dataset split: dev or test       (default: dev)
#   CONDITIONS       space-separated condition list   (default: all)
#                    e.g. CONDITIONS="original LP-1 LP-2"
#   CONSISTENCY_N    self-consistency samples/item    (default: from inference.yaml = 10)
#                    e.g. CONSISTENCY_N=3 for fast exploration (3.3× cheaper)
#   MAX_MODEL_LEN    vllm max context length          (default: 8192)
#                    models default to 262144 which OOMs on a single H200 — always keep at 8192
#   DP               data parallel replicas           (default: 8)
#   TP               tensor parallel size per rep     (default: 1)
#   PORT             vllm server port                 (default: 8000)
#
# Per-model DP/TP on 8× H200:
#   2B:  DP=8 TP=1   (default)
#   8B:  DP=8 TP=1   (default)
#   72B: DP=4 TP=2   → qsub -v RTYPE=rt_HF,MODEL=Qwen/Qwen3-VL-72B-Instruct,DP=4,TP=2,...
#
# concurrency is resolved automatically from config/models.yaml per model.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
cd "${PBS_O_WORKDIR}"
JOBID="${PBS_JOBID%%.*}"
mkdir -p logs
exec > "logs/infer-${JOBID}.out" 2> "logs/infer-${JOBID}.err"
echo "JOBID=${JOBID}"

set -euxo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source scripts/abci/common/setup.sh

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DATASET="${DATASET:-sciclaimeval}"
SPLIT="${SPLIT:-dev}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# Auto-read DP/TP from config/models.yaml; env vars passed via qsub -v take precedence
_yaml_dp=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('config/models.yaml'))
print(cfg.get('${MODEL}', {}).get('data_parallel_size', 8))
")
_yaml_tp=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('config/models.yaml'))
print(cfg.get('${MODEL}', {}).get('tensor_parallel_size', 1))
")
DP="${DP:-${_yaml_dp}}"
TP="${TP:-${_yaml_tp}}"

echo "MODEL=${MODEL}  DATASET=${DATASET}  SPLIT=${SPLIT}  DP=${DP}  TP=${TP}  PORT=${PORT}  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
[[ -n "${CONDITIONS:-}"    ]] && echo "CONDITIONS=${CONDITIONS}"
[[ -n "${CONSISTENCY_N:-}" ]] && echo "CONSISTENCY_N=${CONSISTENCY_N}"

# ---------------------------------------------------------------------------
# Start vllm serve
# ---------------------------------------------------------------------------
vllm serve "${MODEL}" \
  --data-parallel-size "${DP}" \
  --tensor-parallel-size "${TP}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --port "${PORT}" \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  &> "logs/vllm-${JOBID}.log" &
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
    echo "ERROR: vllm server process died. Check logs/vllm-${JOBID}.log" >&2
    exit 1
  fi
  if [[ "${WAIT_SECS}" -gt 1800 ]]; then
    echo "ERROR: timed out waiting for vllm server (${WAIT_SECS}s)" >&2
    exit 1
  fi
done
echo "vllm server ready after ${WAIT_SECS}s"

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
python -m core.run infer \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --model "${MODEL}" \
  --resume \
  ${CONDITIONS:+--conditions ${CONDITIONS}} \
  ${CONSISTENCY_N:+--consistency-n ${CONSISTENCY_N}}

echo "Inference complete."
