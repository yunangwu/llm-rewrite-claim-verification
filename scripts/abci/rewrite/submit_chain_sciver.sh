#!/bin/bash
# Chain rewrite jobs for SciVer dev then test (natural + inject, both splits).
#
# After both jobs complete, run the mix step on the login node:
#   python -m core.run rewrite mix --dataset sciver --split dev  --seed 42
#   python -m core.run rewrite mix --dataset sciver --split test --seed 42
#
# Then submit inference with:
#   bash scripts/abci/infer/submit_chain_sciver_mix42.sh
#
# Usage:
#   bash scripts/abci/rewrite/submit_chain_sciver.sh

set -euo pipefail

SCRIPT="scripts/abci/rewrite/submit_rewrite.sh"

dependency=()

for SPLIT in dev test; do
  jobid=$(qsub "${dependency[@]}" \
    -v "RTYPE=rt_HF,DATASET=sciver,SPLIT=${SPLIT},METHOD=both" \
    "${SCRIPT}")
  echo "submitted [rewrite ${SPLIT}] → ${jobid}"
  dependency=(-W "depend=afterany:${jobid}")
done

echo ""
echo "Rewrite chain submitted (dev → test)."
echo "After both jobs complete, run on the login node:"
echo "  python -m core.run rewrite mix --dataset sciver --split dev  --seed 42"
echo "  python -m core.run rewrite mix --dataset sciver --split test --seed 42"
echo "Then: bash scripts/abci/infer/submit_chain_sciver_mix42.sh"
