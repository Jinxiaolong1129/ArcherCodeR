#!/usr/bin/env bash
# Submission commands for BASE 16k evaluation on two nodes.
# Mixed scheduling for better balance:
# - Node7: aime2024 + livecodebench_v5
# - Node8: aime2025 + livecodebench_v6

# ==============================
# Node7: BASE MIX (aime2024 + lcb_v5) 16k
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" bash tools/run_base_mix_a24_lcb5_16k_node.sh


# ==============================
# Node8: BASE MIX (aime2025 + lcb_v6) 16k
# ==============================
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/data_storage/wyj/systems/envs/archer"
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}/Archer_eval"
STEPS_CSV="10,50,80,105" MERGE_IF_NEEDED="1" bash tools/run_base_mix_a25_lcb6_16k_node.sh

