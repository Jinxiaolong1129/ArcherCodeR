#!/usr/bin/env bash
set -euo pipefail

RAY_TMPDIR="/data_storage/wyj/r/${HOSTNAME:0:6}_$(date +%m%d%H%M)_$((RANDOM%1000))"
mkdir -p "${RAY_TMPDIR}"
export RAY_TMPDIR


BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

METHOD_SCRIPTS=(
  "run_swe_Pure-GRPO-unified.sh"
  "run_swe_intuitor-unified.sh"
  "run_swe_trajectory_entropy-unified.sh"
  "run_swe_token_entropy-unified.sh"
  "run_swe_prob_disparity-unified.sh"
)

NS=("8" "12")

for script in "${METHOD_SCRIPTS[@]}"; do
  for n in "${NS[@]}"; do
    echo "=== Running ${script} with n=${n} ==="
    EXP_SUFFIX="n${n}" N_RESP_OVERRIDE="${n}" bash "${BASE_DIR}/${script}" "$@"
  done
done

