#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   NODE_ID=1 bash scripts-iclr-server/unified_fair_compare/submit_missing_eval_3nodes_by_id.sh
#   NODE_ID=2 bash scripts-iclr-server/unified_fair_compare/submit_missing_eval_3nodes_by_id.sh
#   NODE_ID=3 bash scripts-iclr-server/unified_fair_compare/submit_missing_eval_3nodes_by_id.sh

NODE_ID="${NODE_ID:?NODE_ID is required (1|2|3)}"

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

run_all_streams() {
  local DATASET="$1" N_SAMPLES="$2" TAG="$3" TS

  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w1" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E1_8k_${DATASET}_${TS}.log" 2>&1 &
  PID1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E2_16k_${DATASET}_${TS}.log" 2>&1 &
  PID2=$!
  wait "${PID1}" "${PID2}"

  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w1" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E2_8k_${DATASET}_${TS}.log" 2>&1 &
  PID3=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E1_16k_${DATASET}_${TS}.log" 2>&1 &
  PID4=$!
  wait "${PID3}" "${PID4}"
}

run_8k_only() {
  local DATASET="$1" N_SAMPLES="$2" TAG="$3" TS
  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w1" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E1_8k_${DATASET}_${TS}.log" 2>&1 &
  PID1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w2" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E2_8k_${DATASET}_${TS}.log" 2>&1 &
  PID2=$!
  wait "${PID1}" "${PID2}"
}

run_16k_e1_split() {
  local DATASET="$1" N_SAMPLES="$2" TAG="$3" TS
  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w1" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E1_16k_${DATASET}_s10_50_${TS}.log" 2>&1 &
  PID1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E1_16k_${DATASET}_s80_105_${TS}.log" 2>&1 &
  PID2=$!
  wait "${PID1}" "${PID2}"
}

run_16k_e2_pair() {
  local DS_A="$1" DS_B="$2" N_SAMPLES="$3" TAG="$4" TS
  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DS_A}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w1" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DS_A}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E2_16k_${DS_A}_${TS}.log" 2>&1 &
  PID1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DS_B}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DS_B}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${TAG}_E2_16k_${DS_B}_${TS}.log" 2>&1 &
  PID2=$!
  wait "${PID1}" "${PID2}"
}

case "${NODE_ID}" in
  1)
    run_all_streams aime2024 32 node1
    run_8k_only livecodebench_v5 8 node1
    run_16k_e1_split livecodebench_v5 8 node1
    SHARD_ID=0 NUM_SHARDS=3 NUM_WORKERS=96 bash tools/run_lcb_metrics_shard_local.sh
    ;;
  2)
    run_all_streams aime2025 32 node2
    run_8k_only livecodebench_v6 8 node2
    run_16k_e1_split livecodebench_v6 8 node2
    SHARD_ID=1 NUM_SHARDS=3 NUM_WORKERS=96 bash tools/run_lcb_metrics_shard_local.sh
    ;;
  3)
    run_16k_e2_pair livecodebench_v5 livecodebench_v6 8 node3
    for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
      if [[ "${DATASET}" == aime* ]]; then N_SAMPLES=32; else N_SAMPLES=8; fi
      DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="80,105" MERGE_IF_NEEDED="1" \
        bash tools/run_abcd_node_pair.sh F1 F2
    done
    SHARD_ID=2 NUM_SHARDS=3 NUM_WORKERS=96 bash tools/run_lcb_metrics_shard_local.sh
    ;;
  *)
    echo "Invalid NODE_ID=${NODE_ID}, must be 1|2|3" >&2
    exit 2
    ;;
esac
