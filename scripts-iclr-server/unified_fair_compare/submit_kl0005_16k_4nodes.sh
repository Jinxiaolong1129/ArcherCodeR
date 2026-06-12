#!/usr/bin/env bash
# Submit commands for 4 GPU nodes to backfill KL0005 16k eval.
# - Each node handles one experiment (GROUP=D, WORKER_TOTAL=4).
# - Steps fixed to 10,50.
# - After GPU generation, each node runs LCB CPU metric only for its own experiment dir.
#
# Usage:
# 1) Run "Node1" block on node 1
# 2) Run "Node2" block on node 2
# 3) Run "Node3" block on node 3
# 4) Run "Node4" block on node 4

# ==============================
# Node1: D worker 1/4 (Intuitor-kl0005)
# ==============================
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
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then N_SAMPLES=32; else N_SAMPLES=8; fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    GROUP=D WORKER_INDEX=1 WORKER_TOTAL=4 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50" MERGE_IF_NEEDED=1 N_GPUS=8 \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh
done
WANDB_MODE=disabled WANDB_DISABLED=true \
MODEL_ROOT="${PROJ_DIR}/output/self-rl-jxl/Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005" \
SHARD_ID=0 NUM_SHARDS=1 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh


# ==============================
# Node2: D worker 2/4 (ProbDisparity-kl0005)
# ==============================
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
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then N_SAMPLES=32; else N_SAMPLES=8; fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    GROUP=D WORKER_INDEX=2 WORKER_TOTAL=4 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50" MERGE_IF_NEEDED=1 N_GPUS=8 \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh
done
WANDB_MODE=disabled WANDB_DISABLED=true \
MODEL_ROOT="${PROJ_DIR}/output/self-rl-jxl/Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005" \
SHARD_ID=0 NUM_SHARDS=1 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh


# ==============================
# Node3: D worker 3/4 (TokenEntropy-kl0005)
# ==============================
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
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then N_SAMPLES=32; else N_SAMPLES=8; fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    GROUP=D WORKER_INDEX=3 WORKER_TOTAL=4 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50" MERGE_IF_NEEDED=1 N_GPUS=8 \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh
done
WANDB_MODE=disabled WANDB_DISABLED=true \
MODEL_ROOT="${PROJ_DIR}/output/self-rl-jxl/Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005" \
SHARD_ID=0 NUM_SHARDS=1 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh


# ==============================
# Node4: D worker 4/4 (TrajectoryEntropy-kl0005)
# ==============================
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
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then N_SAMPLES=32; else N_SAMPLES=8; fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    GROUP=D WORKER_INDEX=4 WORKER_TOTAL=4 \
    DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" \
    STEPS_CSV="10,50" MERGE_IF_NEEDED=1 N_GPUS=8 \
    MAX_RESPONSE_LEN=16384 \
    OUTPUT_SUBDIR="output_16k/${DATASET}_n${N_SAMPLES}" \
    bash tools/eval_abcd_worker.sh
done
WANDB_MODE=disabled WANDB_DISABLED=true \
MODEL_ROOT="${PROJ_DIR}/output/self-rl-jxl/Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005" \
SHARD_ID=0 NUM_SHARDS=1 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh
