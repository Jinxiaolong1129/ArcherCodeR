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

TEMPS=("0.8" "1.2")

for script in "${METHOD_SCRIPTS[@]}"; do
  for temp in "${TEMPS[@]}"; do
    echo "=== Running ${script} with temperature=${temp} ==="
    EXP_SUFFIX="temp${temp}" TEMP_OVERRIDE="${temp}" bash "${BASE_DIR}/${script}" "$@"
  done
done

