#!/usr/bin/env bash
# Submission commands for Archer_eval A/B/C/D evaluation batches.
# One block = one node. Run each block on its target node.

# ------------------------------------------------------------------
# Common notes
# - 4+4 split per node via tools/run_abcd_node_pair.sh
# - Run order is fixed: aime2024 -> aime2025 -> lcb_v5 -> lcb_v6
# - Steps default to 10,50,80,105
# - N_SAMPLES policy:
#     * AIME: pass@32 style generation (N_SAMPLES=32)
#     * LCB:  pass@8  style generation (N_SAMPLES=8)
# ------------------------------------------------------------------

# ==============================
# Node1: B1 + B2
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then
    N_SAMPLES=32
  else
    N_SAMPLES=8
  fi
  DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
  bash tools/run_abcd_node_pair.sh B1 B2
done


# ==============================
# Node2: B3 + B4
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then
    N_SAMPLES=32
  else
    N_SAMPLES=8
  fi
  DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
  bash tools/run_abcd_node_pair.sh B3 B4
done


# ==============================
# Node3: C1 + C2
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then
    N_SAMPLES=32
  else
    N_SAMPLES=8
  fi
  DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
  bash tools/run_abcd_node_pair.sh C1 C2
done


# ==============================
# Node4: C3 + C4
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then
    N_SAMPLES=32
  else
    N_SAMPLES=8
  fi
  DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
  bash tools/run_abcd_node_pair.sh C3 C4
done


# ==============================
# Node5: A1 + A2
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then
    N_SAMPLES=32
  else
    N_SAMPLES=8
  fi
  DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
  bash tools/run_abcd_node_pair.sh A1 A2
done


# ==============================
# Node6: D1 + D2
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
for DATASET in aime2024 aime2025 livecodebench_v5 livecodebench_v6; do
  if [[ "${DATASET}" == aime* ]]; then
    N_SAMPLES=32
  else
    N_SAMPLES=8
  fi
  DATASET="${DATASET}" N_SAMPLES="${N_SAMPLES}" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
  bash tools/run_abcd_node_pair.sh D1 D2
done


# ==============================
# Node7: reserved (optional)
# ==============================
# export envs_dirs="/data_storage/wyj/systems/envs"
# export HF_HOME="/data_storage/wyj/systems/huggingface"
# export HTTP_PROXY="http://100.68.168.184:3128"
# export HTTPS_PROXY="http://100.68.168.184:3128"
# source activate archer
# export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
# cd "${PROJ_DIR}/Archer_eval"
# # Example: rerun failed jobs or switch dataset
# DATASET="aime2024" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
# bash tools/run_abcd_node_pair.sh B1 C1


# ==============================
# Node8: reserved (optional)
# ==============================
# export envs_dirs="/data_storage/wyj/systems/envs"
# export HF_HOME="/data_storage/wyj/systems/huggingface"
# export HTTP_PROXY="http://100.68.168.184:3128"
# export HTTPS_PROXY="http://100.68.168.184:3128"
# source activate archer
# export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
# cd "${PROJ_DIR}/Archer_eval"
# DATASET="minervamath" STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" \
# bash tools/run_abcd_node_pair.sh B2 C2
