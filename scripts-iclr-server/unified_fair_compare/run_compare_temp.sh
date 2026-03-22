#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

METHOD_SCRIPTS=(
  "run_Pure-GRPO-unified.sh"
  "run_intuitor-unified.sh"
  "run_trajectory_entropy-unified.sh"
  "run_token_entropy-unified.sh"
  "run_prob_disparity-unified.sh"
)

TEMPS=("0.8" "1.2")

for script in "${METHOD_SCRIPTS[@]}"; do
  for temp in "${TEMPS[@]}"; do
    echo "=== Running ${script} with temperature=${temp} ==="
    EXP_SUFFIX="temp${temp}" TEMP_OVERRIDE="${temp}" bash "${BASE_DIR}/${script}" "$@"
  done
done

