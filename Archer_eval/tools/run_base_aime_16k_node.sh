#!/usr/bin/env bash
#
# Run BASE (A group: A1 + A2) AIME 16k eval on one 8-GPU node (4+4 split).
# Datasets: aime2024, aime2025
#

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-/data_storage/wyj/jxl/ArcherCodeR}"
EVAL_DIR="${EVAL_DIR:-${PROJ_DIR}/Archer_eval}"

cd "${EVAL_DIR}"

for DATASET in aime2024 aime2025; do
  DATASET="${DATASET}" \
  N_SAMPLES="${N_SAMPLES:-32}" \
  STEPS_CSV="${STEPS_CSV:-10,50,80,105}" \
  MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}" \
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}" \
  OUTPUT_SUBDIR="${OUTPUT_SUBDIR_PREFIX:-output_16k_n32}/${DATASET}" \
  bash tools/run_abcd_node_pair.sh A1 A2
done

