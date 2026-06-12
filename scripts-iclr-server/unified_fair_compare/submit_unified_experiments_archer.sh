#!/usr/bin/env bash
# Submission commands for unified_fair_compare experiments.
# Copy one block at a time. One command block = one training job.

# ------------------------------
# 1) Intuitor (continue to total_epochs=2 from existing checkpoints)
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
TOTAL_EPOCHS_OVERRIDE="2" bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh


# ------------------------------
# 2) Pure-GRPO
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
SAVE_FREQ_OVERRIDE="5" bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh

# ------------------------------
# 2b) Pure-GRPO (clip_ratio_c=10, ppo_epochs=3)
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
SAVE_FREQ_OVERRIDE="5" bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified-c10-e3.sh

# ------------------------------
# 2c) Pure-GRPO + token entropy separate clip (c3, e3)
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
SAVE_FREQ_OVERRIDE="5" bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified-token-separate-e3.sh


# ------------------------------
# 3) Token Entropy
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
SAVE_FREQ_OVERRIDE="5" bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh


# ------------------------------
# 4) Trajectory Entropy
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
SAVE_FREQ_OVERRIDE="5" bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh


# ------------------------------
# 5) Probability Disparity
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
SAVE_FREQ_OVERRIDE="5" bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh


# ------------------------------
# 6) Epoch-3 override command set (no wrapper scripts needed)
# ------------------------------

# 6.1) Intuitor (ppo_epoch=3)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="ppo_epoch3" bash scripts-iclr-server/unified_fair_compare/run_swe_intuitor-unified.sh \
  actor_rollout_ref.actor.ppo_epochs=3

# 6.2) Token Entropy (ppo_epoch=3)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="ppo_epoch3" bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh \
  actor_rollout_ref.actor.ppo_epochs=3 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# 6.3) Trajectory Entropy (ppo_epoch=3)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="ppo_epoch3" bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh \
  actor_rollout_ref.actor.ppo_epochs=3

# 6.4) Probability Disparity (ppo_epoch=3)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="ppo_epoch3" bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh \
  actor_rollout_ref.actor.ppo_epochs=3 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
