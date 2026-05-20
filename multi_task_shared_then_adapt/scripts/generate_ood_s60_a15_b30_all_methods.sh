#!/usr/bin/env bash
set -euo pipefail

PY=/home/alperen/conda/miniconda/envs/in_context_reasoning_env/bin/python
SCRIPT=multi_task_shared_then_adapt/plotting/plot_ood_b30_adaptation_by_holdout.py

export MPLCONFIGDIR=/tmp/mplconfig

"$PY" "$SCRIPT" \
  --family circle_packing \
  --family circle_packing_rectangle \
  --family heilbronn_triangle \
  --baseline-budget 30 \
  --setting-prefix s60-a15-b30 \
  --include-shared \
  --single-budget-unified-sts-color \
  --output-stem-suffix _s60_a15_b30_all_methods \
  --run-timeout-seconds 0
