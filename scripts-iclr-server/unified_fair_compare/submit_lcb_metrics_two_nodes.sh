#!/usr/bin/env bash
# Submit commands for two CPU nodes:
# - Both nodes run sequentially on their own shard
# - Combined shards cover all pending lcbv5 + lcbv6 metric tasks
#
# Usage:
# 1) Run "NodeA" block on CPU node A
# 2) Run "NodeB" block on CPU node B

# ==============================
# NodeA: shard 0/2
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
SHARD_ID=0 NUM_SHARDS=2 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh


# ==============================
# NodeB: shard 1/2
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
SHARD_ID=1 NUM_SHARDS=2 NUM_WORKERS=96 \
bash tools/run_lcb_metrics_shard_local.sh
