#!/usr/bin/env bash
#
# Submit commands for MISSING evaluations — 3 GPU nodes (8 GPUs + 180 CPUs each).
#
# Workload:
#   GROUP E — two Pure-GRPO variants with NO eval at all (ckpts ready at steps 10,50,80,105):
#     E1: clip_ratio_c10-ppo_epoch3    (WORKER_INDEX=1 → experiment 0)
#     E2: token-separate-c3-e3         (WORKER_INDEX=2 → experiment 1)
#
#   GROUP F — partial completions (many ckpts still training, auto-skipped if missing):
#     F1 (round-robin 0,2): TrajectoryEntropy-n8 (step 105 v5 parquet was empty)
#                           Intuitor-n12          (steps 80,105 ckpts not ready yet)
#     F2 (round-robin 1,3): Intuitor-n8           (step 105 ckpt not ready yet)
#                           Intuitor-kl0005       (step 105 ckpt not ready yet)
#
# Dataset split across 3 nodes — LCB 16k split by experiment (E1 vs E2):
#
#   Node 1  aime2024 (8k+16k) + lcbv5 8k (E1+E2) + lcbv5 E1_16k (step-split, 4+4 GPUs)
#           eval runs: 16 + 8 + 4 = 28 runs
#
#   Node 2  aime2025 (8k+16k) + lcbv6 8k (E1+E2) + lcbv6 E1_16k (step-split, 4+4 GPUs)
#           eval runs: 16 + 8 + 4 = 28 runs
#
#   Node 3  lcbv5 E2_16k + lcbv6 E2_16k (concurrent 4+4) + GROUP F (mostly skips)
#           eval runs: 4 + 4 + ~1 = ~9 runs
#
# LCB 16k per node: 4 + 4 + 8 = 16 total, all 3 nodes participate.
# Each node runs LCB metrics on its CPUs after GPU eval (shard 0/3, 1/3, 2/3).
#
# Dependencies:
#   eval_abcd_worker.sh   must include GROUP E and F  (already added)
#   run_abcd_node_pair.sh must have E/F TOTAL=2       (already added)
#
# ─────────────────────────────────────────────────────────────────────────────

# ── helpers ──────────────────────────────────────────────────────────────────

# run_dataset_all_streams <DATASET> <N_SAMPLES> <NODE_TAG>
#   Full 4-stream eval (8k + 16k) in two concurrent passes.
run_dataset_all_streams() {
  local DATASET="$1" N_SAMPLES="$2" NODE_TAG="$3" TS
  TS="$(date +%Y%m%d_%H%M%S)"

  # Pass 1: E1 8k (GPU 0-3)  +  E2 16k (GPU 4-7)
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w1" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E1_8k_${DATASET}_${TS}.log" 2>&1 &
  local PID1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E2_16k_${DATASET}_${TS}.log" 2>&1 &
  local PID2=$!
  wait "${PID1}" "${PID2}"
  TS="$(date +%Y%m%d_%H%M%S)"

  # Pass 2: E2 8k (GPU 0-3)  +  E1 16k (GPU 4-7)
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w1" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E2_8k_${DATASET}_${TS}.log" 2>&1 &
  local PID3=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E1_16k_${DATASET}_${TS}.log" 2>&1 &
  local PID4=$!
  wait "${PID3}" "${PID4}"
}

# run_dataset_8k_only <DATASET> <N_SAMPLES> <NODE_TAG>
#   8k-only pass: E1_8k (GPU 0-3) + E2_8k (GPU 4-7) concurrently.
run_dataset_8k_only() {
  local DATASET="$1" N_SAMPLES="$2" NODE_TAG="$3" TS
  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w1" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E1_8k_${DATASET}_${TS}.log" 2>&1 &
  local PID1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w2" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E2_8k_${DATASET}_${TS}.log" 2>&1 &
  local PID2=$!
  wait "${PID1}" "${PID2}"
}

# run_dataset_16k_e1_split <DATASET> <N_SAMPLES> <NODE_TAG>
#   16k pass for E1 only, but split steps across two workers:
#   - GPU 0-3 runs steps 10,50
#   - GPU 4-7 runs steps 80,105
#   Total still 4 eval runs, while all 8 GPUs are used.
run_dataset_16k_e1_split() {
  local DATASET="$1" N_SAMPLES="$2" NODE_TAG="$3" TS
  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w1" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E1_16k_${DATASET}_s10_50_${TS}.log" 2>&1 &
  local PID1=$!

  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=1 WORKER_TOTAL=2 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E1_16k_${DATASET}_s80_105_${TS}.log" 2>&1 &
  local PID2=$!

  wait "${PID1}" "${PID2}"
}

# run_dataset_16k_e2_pair <DATASET_A> <DATASET_B> <N_SAMPLES> <NODE_TAG>
#   16k pass for E2 only, two datasets run concurrently:
#   DATASET_A on GPU 0-3, DATASET_B on GPU 4-7.  4+4 = 8 eval runs.
run_dataset_16k_e2_pair() {
  local DS_A="$1" DS_B="$2" N_SAMPLES="$3" NODE_TAG="$4" TS
  TS="$(date +%Y%m%d_%H%M%S)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DS_A}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w1" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DS_A}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E2_16k_${DS_A}_${TS}.log" 2>&1 &
  local PID1=$!
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
    GROUP=E WORKER_INDEX=2 WORKER_TOTAL=2 \
    DATASET="${DS_B}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
    N_GPUS=4 RAY_TMP_LINK="/tmp/rayds_${NODE_TAG}_w2" \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DS_B}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh \
    > "tools/logs/${NODE_TAG}_E2_16k_${DS_B}_${TS}.log" 2>&1 &
  local PID2=$!
  wait "${PID1}" "${PID2}"
}

# ─────────────────────────────────────────────────────────────────────────────


# ==============================
# Node 1: aime2024 (8k+16k)  +  lcbv5 8k (E1+E2)  +  lcbv5 E1_16k (step-split)
# eval runs: 16 + 8 + 4 = 28 runs
# Then: LCB metrics shard 0/3
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
mkdir -p tools/logs

run_dataset_all_streams aime2024         32 node1
run_dataset_8k_only     livecodebench_v5  8 node1
run_dataset_16k_e1_split livecodebench_v5 8 node1

SHARD_ID=0 NUM_SHARDS=3 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh


# ==============================
# Node 2: aime2025 (8k+16k)  +  lcbv6 8k (E1+E2)  +  lcbv6 E1_16k (step-split)
# eval runs: 16 + 8 + 4 = 28 runs
# Then: LCB metrics shard 1/3
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
mkdir -p tools/logs

run_dataset_all_streams aime2025         32 node2
run_dataset_8k_only     livecodebench_v6  8 node2
run_dataset_16k_e1_split livecodebench_v6 8 node2

SHARD_ID=1 NUM_SHARDS=3 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh


# ==============================
# Node 3: lcbv5 E2_16k  +  lcbv6 E2_16k (concurrent 4+4)  +  GROUP F (mostly skips)
# eval runs: 4 + 4 + ~1 ≈ 9 runs
# Then: LCB metrics shard 2/3
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
mkdir -p tools/logs

# E2_16k for lcbv5 and lcbv6 run concurrently on GPU 0-3 and GPU 4-7
run_dataset_16k_e2_pair livecodebench_v5 livecodebench_v6 8 node3

# GROUP F: partial completions — STEPS_CSV=80,105, missing ckpts auto-skip
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then N_SAMPLES=32; else N_SAMPLES=8; fi
  DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="80,105" MERGE_IF_NEEDED="1" \
  bash tools/run_abcd_node_pair.sh F1 F2
done

SHARD_ID=2 NUM_SHARDS=3 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh


# ─────────────────────────────────────────────────────────────────────────────
# Re-run block: Intuitor partial steps — run on any free node AFTER training ends
# (Intuitor-n8 step 105, Intuitor-n12 steps 80/105, Intuitor-kl0005 step 105)
# ─────────────────────────────────────────────────────────────────────────────

# === RERUN (any free node) ===
# export envs_dirs="/data_storage/wyj/systems/envs"
# export HF_HOME="/data_storage/wyj/systems/huggingface"
# export HTTP_PROXY="http://100.68.168.184:3128"
# export HTTPS_PROXY="http://100.68.168.184:3128"
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "/data_storage/wyj/systems/envs/archer"
# export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
# cd "${PROJ_DIR}/Archer_eval"
# for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
#   if [[ "${DATASET}" == aime* ]]; then N_SAMPLES=32; else N_SAMPLES=8; fi
#   DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="80,105" MERGE_IF_NEEDED="1" \
#   bash tools/run_abcd_node_pair.sh F1 F2
# done
# SHARD_ID=0 NUM_SHARDS=1 NUM_WORKERS=96 bash tools/run_lcb_metrics_shard_local.sh
