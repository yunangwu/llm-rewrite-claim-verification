#!/bin/bash
# Chain inference jobs for all models on sciclaimeval dev+test with mix-42 conditions.
#
# Ordered fastest → slowest. For each model, dev and test are submitted back-to-back
# so the model only needs to be loaded twice in close succession rather than being
# spread across the whole queue.
#
# Prerequisite — generate mix rewrite files first (if not already done):
#   python -m core.run rewrite mix --dataset sciclaimeval --split dev  --seed 42
#   python -m core.run rewrite mix --dataset sciclaimeval --split test --seed 42
#
# Usage:
#   bash scripts/abci/infer/submit_chain_mix42.sh
#
# Conditions: original + 5 natural groups (mix-42) + D_1 + C_1 = 8 conditions total.

set -euo pipefail

CONDS="original LP-mix-42 FI-mix-42 AS-mix-42 OA-mix-42 CT-mix-42 D_1 C_1"
SCRIPT="scripts/abci/infer/submit_infer.sh"

# Ordered fastest → slowest (by active params / configured concurrency)
MODELS=(
  "moonshotai/Kimi-VL-A3B-Thinking-2506"   # 3B active MoE, concurrency=96
  "OpenGVLab/InternVL3-2B"                  # 2B,  concurrency=96
  "Qwen/Qwen3-VL-2B-Instruct"               # 2B,  concurrency=64
  "Qwen/Qwen3-VL-4B-Instruct"               # 4B,  concurrency=80
  "zai-org/GLM-4.6V-Flash"                  # 9B,  concurrency=64
  "Qwen/Qwen3-VL-8B-Instruct"               # 8B,  concurrency=64
  "OpenGVLab/InternVL3-8B"                  # 8B,  concurrency=64
  "google/gemma-4-26B-A4B-it"               # 4B active MoE, concurrency=64
  "google/gemma-4-31B-it"                   # 31B dense, concurrency=48
  "Qwen/Qwen3-VL-32B-Instruct"              # 32B dense, concurrency=32
  "OpenGVLab/InternVL3-38B"                 # 38B, concurrency=32
  "OpenGVLab/InternVL3-78B"                 # 78B, TP=2 DP=4, concurrency=16
)

dependency=()

for MODEL in "${MODELS[@]}"; do
  for SPLIT in dev test; do
    jobid=$(qsub "${dependency[@]}" \
      -v "RTYPE=rt_HF,DATASET=sciclaimeval,SPLIT=${SPLIT},MODEL=${MODEL},CONDITIONS=${CONDS}" \
      "${SCRIPT}")
    echo "submitted [${SPLIT}] ${MODEL}  →  ${jobid}"
    dependency=(-W "depend=afterany:${jobid}")
  done
done

echo "Done. ${#MODELS[@]} models × 2 splits = $((${#MODELS[@]} * 2)) jobs queued."
