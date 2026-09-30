#!/bin/bash
# Chain inference jobs for SciVer DEV split only, with mix-42 conditions.
# Kimi-VL-A3B-Thinking-2506 excluded (already completed).
# Ordered fastest → slowest.
#
# Usage:
#   bash scripts/abci/infer/submit_chain_sciver_mix42_dev.sh

set -euo pipefail

CONDS="original LP-mix-42 FI-mix-42 AS-mix-42 OA-mix-42 CT-mix-42 D_1 C_1"
SCRIPT="scripts/abci/infer/submit_infer.sh"

MODELS=(
  "OpenGVLab/InternVL3-2B"
  "Qwen/Qwen3-VL-2B-Instruct"
  "Qwen/Qwen3-VL-4B-Instruct"
  "zai-org/GLM-4.6V-Flash"
  "Qwen/Qwen3-VL-8B-Instruct"
  "OpenGVLab/InternVL3-8B"
  "google/gemma-4-26B-A4B-it"
  "google/gemma-4-31B-it"
  "Qwen/Qwen3-VL-32B-Instruct"
  "OpenGVLab/InternVL3-38B"
  "OpenGVLab/InternVL3-78B"
)

dependency=()

for MODEL in "${MODELS[@]}"; do
  jobid=$(qsub "${dependency[@]}" \
    -v "RTYPE=rt_HF,DATASET=sciver,SPLIT=dev,MODEL=${MODEL},CONDITIONS=${CONDS}" \
    "${SCRIPT}")
  echo "submitted [dev] ${MODEL}  →  ${jobid}"
  dependency=(-W "depend=afterany:${jobid}")
done

echo "Done. ${#MODELS[@]} dev jobs queued."
