#!/bin/bash
# Sourceable env setup for ABCI PBS jobs.
# Usage: source scripts/abci/common/setup.sh
# Requires PBS_O_WORKDIR to be set (automatically set by PBS when a job runs).

cd "${PBS_O_WORKDIR}"

# System profile files reference unbound variables; relax -u around them.
set +u
source /etc/profile.d/modules.sh
source ~/.bash_profile
set -u

module load cuda/12.4/12.4.1
module load cudnn/9.8/9.8.0
module load nccl/2.26/2.26.6-1

# Activate the uv-managed venv (created by: uv sync)
source .venv/bin/activate

export PYTHONPATH="${PBS_O_WORKDIR}"
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES