#!/usr/bin/env bash
set -euo pipefail

# Backfill script for final missing items after latest scan.
#
# Usage (run on 3 nodes in parallel):
#   NODE_ID=1 bash scripts-iclr-server/unified_fair_compare/submit_backfill_missing_final_3nodes_by_id.sh
#   NODE_ID=2 bash scripts-iclr-server/unified_fair_compare/submit_backfill_missing_final_3nodes_by_id.sh
#   NODE_ID=3 bash scripts-iclr-server/unified_fair_compare/submit_backfill_missing_final_3nodes_by_id.sh
#
# Optional:
#   RUN_GPU_F=1   # Node 3 also retries GROUP F (Intuitor partial ckpts, steps 80/105)
#
# Notes:
# - Main gap is missing LCB *.pass.lcb.csv while parquet exists.
# - We disable wandb for metric jobs to avoid "No API key configured" failures.
# - Metrics script auto-skips files that already have .pass.lcb.csv.

NODE_ID="${NODE_ID:?NODE_ID is required (1|2|3)}"
RUN_GPU_F="${RUN_GPU_F:-0}"

export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"

export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
mkdir -p tools/logs

# Prevent wandb login failure in LCB metric pass.
export WANDB_MODE=disabled
export WANDB_DISABLED=true

run_metrics_shard() {
  local shard_id="$1"
  # NUM_SHARDS=3 to utilize all three nodes.
  SHARD_ID="${shard_id}" NUM_SHARDS=3 NUM_WORKERS=96 \
    bash tools/run_lcb_metrics_shard_local.sh
}

run_gpu_group_f_retry() {
  # Retry only the partial Intuitor/TrajEntropy gaps (steps 80,105).
  # Missing checkpoints are skipped automatically by eval_abcd_worker.sh.
  for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
    if [[ "${DATASET}" == aime* ]]; then
      N_SAMPLES=32
    else
      N_SAMPLES=8
    fi
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="80,105" MERGE_IF_NEEDED="1" \
      bash tools/run_abcd_node_pair.sh F1 F2
  done
}

case "${NODE_ID}" in
  1)
    run_metrics_shard 0
    ;;
  2)
    run_metrics_shard 1
    ;;
  3)
    if [[ "${RUN_GPU_F}" == "1" ]]; then
      run_gpu_group_f_retry
    fi
    run_metrics_shard 2
    ;;
  *)
    echo "Invalid NODE_ID=${NODE_ID}, must be 1|2|3" >&2
    exit 2
    ;;
esac

