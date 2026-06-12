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

KL_COEFS=("0.001" "0.005")

for script in "${METHOD_SCRIPTS[@]}"; do
  for kl_coef in "${KL_COEFS[@]}"; do
    coef_tag="${kl_coef/./}"
    echo "=== Running ${script} with kl_loss_coef=${kl_coef} ==="
    EXP_SUFFIX="kl${coef_tag}" \
    USE_KL_LOSS_OVERRIDE="True" \
    KL_LOSS_COEF_OVERRIDE="${kl_coef}" \
    bash "${BASE_DIR}/${script}" "$@"
  done
done

