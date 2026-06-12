#!/usr/bin/env bash
# KL sweep submission commands.
# Copy one block at a time. One command block = one training job.

# ---------- Intuitor | kl=0.001 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0001" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.001" \
bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# ---------- Intuitor | kl=0.005 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0005" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.005" \
bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# ---------- Pure-GRPO | kl=0.001 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0001" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.001" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh

# ---------- Pure-GRPO | kl=0.005 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0005" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.005" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh

# ---------- Trajectory Entropy | kl=0.001 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0001" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.001" \
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh

# ---------- Trajectory Entropy | kl=0.005 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0005" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.005" \
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh

# ---------- Token Entropy | kl=0.001 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0001" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.001" \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh

# ---------- Token Entropy | kl=0.005 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0005" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.005" \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh

# ---------- Prob Disparity | kl=0.001 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0001" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.001" \
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh

# ---------- Prob Disparity | kl=0.005 ----------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="kl0005" USE_KL_LOSS_OVERRIDE="True" KL_LOSS_COEF_OVERRIDE="0.005" \
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh
