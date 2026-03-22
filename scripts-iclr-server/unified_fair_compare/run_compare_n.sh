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

NS=("8" "12")

for script in "${METHOD_SCRIPTS[@]}"; do
  for n in "${NS[@]}"; do
    echo "=== Running ${script} with n=${n} ==="
    EXP_SUFFIX="n${n}" N_RESP_OVERRIDE="${n}" bash "${BASE_DIR}/${script}" "$@"
  done
done

