#!/usr/bin/env bash
#
# Run BASE (A1 + A2) mixed 16k eval on one 8-GPU node:
#   aime2024 -> livecodebench_v5
#

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-/data_storage/wyj/jxl/ArcherCodeR}"
EVAL_DIR="${EVAL_DIR:-${PROJ_DIR}/Archer_eval}"

cd "${EVAL_DIR}"

# AIME24 (pass@32 style)
DATASET="aime2024" \
N_SAMPLES="${AIME_N_SAMPLES:-32}" \
STEPS_CSV="${STEPS_CSV:-10,50,80,105}" \
MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}" \
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}" \
OUTPUT_SUBDIR="${OUTPUT_SUBDIR_PREFIX:-output_16k}/${DATASET}_n32" \
bash tools/run_abcd_node_pair.sh A1 A2

# LCB v5 (pass@8 style)
DATASET="livecodebench_v5" \
N_SAMPLES="${LCB_N_SAMPLES:-8}" \
STEPS_CSV="${STEPS_CSV:-10,50,80,105}" \
MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}" \
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}" \
OUTPUT_SUBDIR="${OUTPUT_SUBDIR_PREFIX:-output_16k}/${DATASET}_n8" \
bash tools/run_abcd_node_pair.sh A1 A2

