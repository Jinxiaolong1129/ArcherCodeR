#!/usr/bin/env bash
set -euo pipefail

# KL0005 16k backfill on 4 nodes.
# Usage:
#   NODE_ID=1 bash scripts-iclr-server/unified_fair_compare/submit_kl0005_16k_4nodes_by_id.sh
#   NODE_ID=2 bash scripts-iclr-server/unified_fair_compare/submit_kl0005_16k_4nodes_by_id.sh
#   NODE_ID=3 bash scripts-iclr-server/unified_fair_compare/submit_kl0005_16k_4nodes_by_id.sh
#   NODE_ID=4 bash scripts-iclr-server/unified_fair_compare/submit_kl0005_16k_4nodes_by_id.sh
#
# What it does per node:
# 1) Run GROUP=D worker (WORKER_TOTAL=4, one worker per node) for steps 10,50 on:
#    aime2024, aime2025, livecodebench_v5, livecodebench_v6
# 2) Run LCB CPU metric only inside that worker's experiment directory.

NODE_ID="${NODE_ID:?NODE_ID is required (1|2|3|4)}"

case "${NODE_ID}" in
  1)
    WORKER_INDEX=1
    EXP_DIR_NAME="Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    ;;
  2)
    WORKER_INDEX=2
    EXP_DIR_NAME="Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    ;;
  3)
    WORKER_INDEX=3
    EXP_DIR_NAME="Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    ;;
  4)
    WORKER_INDEX=4
    EXP_DIR_NAME="Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    ;;
  *)
    echo "Invalid NODE_ID=${NODE_ID}, must be 1|2|3|4" >&2
    exit 2
    ;;
esac

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

echo "[INFO] NODE_ID=${NODE_ID}, WORKER_INDEX=${WORKER_INDEX}, EXP=${EXP_DIR_NAME}"

# GPU generation: 16k only, steps 10 and 50
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then
    N_SAMPLES=32
  else
    N_SAMPLES=8
  fi

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    GROUP=D WORKER_INDEX="${WORKER_INDEX}" WORKER_TOTAL=4 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50" MERGE_IF_NEEDED=1 N_GPUS=8 \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh
done

# CPU metric: only LCB tasks under this node's experiment directory
WANDB_MODE=disabled WANDB_DISABLED=true \
MODEL_ROOT="${PROJ_DIR}/output/self-rl-jxl/${EXP_DIR_NAME}" \
SHARD_ID=0 NUM_SHARDS=1 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh

echo "[INFO] Done NODE_ID=${NODE_ID}"
