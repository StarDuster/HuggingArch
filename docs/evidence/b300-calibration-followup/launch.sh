#!/usr/bin/env bash
set -euo pipefail
source /tmp/scratch-space/k3-performance/env.sh
export GH_CONFIG_DIR=/tmp/scratch-space/k3-performance/gh-config
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/tmp/scratch-space/k3-performance/calibration-worktree:/tmp/scratch-space/k3-performance/benchmark/agentx-yyh/sglang-recipe/python
exec python /tmp/scratch-space/k3-performance/benchmark/b300-calibration-followup/controller.py
