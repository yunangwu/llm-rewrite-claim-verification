#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920261000
#PBS -N 0356_analyze
#PBS -l select=1:ncpus=8
#PBS -l walltime=2:00:00
#PBS -m n
#
# CPU-only job — no GPUs needed.
#
# Submit (RTYPE must always be included when using -v):
#   qsub -v RTYPE=rt_HF scripts/abci/analyze/submit_analyze.sh
#   qsub -v RTYPE=rt_HF,SPLIT=test scripts/abci/analyze/submit_analyze.sh
#
# Optional env vars (override via qsub -v):
#   DATASET    dataset key in config.py     (default: sciclaimeval)
#   SPLIT      dataset split: dev or test   (default: dev)
#   EVAL_MODE  consistency                  (default: consistency)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
cd "${PBS_O_WORKDIR}"
JOBID="${PBS_JOBID%%.*}"
mkdir -p logs
exec > "logs/analyze-${JOBID}.out" 2> "logs/analyze-${JOBID}.err"
echo "JOBID=${JOBID}"

set -euxo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source scripts/abci/common/setup.sh

DATASET="${DATASET:-sciclaimeval}"
SPLIT="${SPLIT:-dev}"
EVAL_MODE="${EVAL_MODE:-consistency}"

echo "DATASET=${DATASET}  SPLIT=${SPLIT}  EVAL_MODE=${EVAL_MODE}"

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
python -m core.run analyze \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --eval-mode "${EVAL_MODE}"

echo "Analysis complete. Results → results/${DATASET}/${SPLIT}/${EVAL_MODE}/"
