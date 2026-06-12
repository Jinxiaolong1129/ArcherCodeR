#!/usr/bin/env bash
# Temperature sweep submission commands.
# Copy one block at a time. One command block = one training job.

# ---------- Intuitor | temp=0.8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp08" TEMP_OVERRIDE="0.8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh

# ---------- Intuitor | temp=1.2 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp12" TEMP_OVERRIDE="1.2" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# ---------- Pure-GRPO | temp=0.8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp08" TEMP_OVERRIDE="0.8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh

# ---------- Pure-GRPO | temp=1.2 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp12" TEMP_OVERRIDE="1.2" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh

# ---------- Trajectory Entropy | temp=0.8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp08" TEMP_OVERRIDE="0.8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh

# ---------- Trajectory Entropy | temp=1.2 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp12" TEMP_OVERRIDE="1.2" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh

# ---------- Token Entropy | temp=0.8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp08" TEMP_OVERRIDE="0.8" \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh

# ---------- Token Entropy | temp=1.2 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp12" TEMP_OVERRIDE="1.2" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh

# ---------- Prob Disparity | temp=0.8 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp08" TEMP_OVERRIDE="0.8" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh

# ---------- Prob Disparity | temp=1.2 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="temp12" TEMP_OVERRIDE="1.2" SAVE_FREQ_OVERRIDE="5" \
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh
