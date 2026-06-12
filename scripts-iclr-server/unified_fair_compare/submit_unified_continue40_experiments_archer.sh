#!/usr/bin/env bash
# Submission commands for unified continue-training experiments (+40).
# Copy one block at a time. One command block = one training job.
# NOTE: Each block below is self-contained (includes env + cd), easy to copy directly.

# ------------------------------
# 0) Intuitor continue to epoch2 (resume from existing checkpoints)
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
# 0.1) TokenEntropy continue to epoch2
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh \
  trainer.total_epochs=2

# ------------------------------
# 0.2) TrajectoryEntropy continue to epoch2
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
bash scripts-iclr-server/unified_fair_compare/run_swe_trajectory_entropy-unified.sh \
  trainer.total_epochs=2

# ------------------------------
# 0.3) ProbDisparity continue to epoch2
# ------------------------------
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
bash scripts-iclr-server/unified_fair_compare/run_swe_prob_disparity-unified.sh \
  trainer.total_epochs=2

# ------------------------------
# 1) Intuitor base continuation
# ------------------------------
# 1A) Resume from step100 to step120 (was at step100, target 120)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-intuitor-base-step80-plus40-to120" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-intuitor-base-step80-plus40-to120/global_step_100" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=120

# 1B-1) Start from base step10 to step50 (not started yet)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
START_STEPS_OVERRIDE="10" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified-from-intuitor-base-continue40.sh

# 1B-2) Resume from step80 to step90 (was at step80, target 90)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-intuitor-base-step50-plus40-to90" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-intuitor-base-step50-plus40-to90/global_step_80" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=90

# 1C) Start from base step105 to step145
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
START_STEPS_OVERRIDE="105" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified-from-intuitor-base-continue40.sh


# ------------------------------
# 2) ProbDisparity base continuation
# ------------------------------
# 2A) Resume from step60 to step90 (resume_from_path=.../global_step_60)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-probdisparity-base-step50-plus40-to90" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-probdisparity-base-step50-plus40-to90/global_step_60" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=90

# 2B) Start from base step105 to step145
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-probdisparity-base-step105-plus40-to145" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl/global_step_105" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=145


# ------------------------------
# 3) TokenEntropy base continuation
# ------------------------------
# 3A) Resume from step60 to step90 (resume_from_path=.../global_step_60)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-tokenentropy-base-step50-plus40-to90" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-tokenentropy-base-step50-plus40-to90/global_step_60" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=90

# 3B) Start from base step105 to step145
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-tokenentropy-base-step105-plus40-to145" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl/global_step_105" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=145


# ------------------------------
# 4) TrajectoryEntropy base continuation
# ------------------------------
# 4A) Resume from step60 to step90 (resume_from_path=.../global_step_60)
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-trajentropy-base-step50-plus40-to90" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-trajentropy-base-step50-plus40-to90/global_step_60" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=90

# 4B) Start from base step105 to step145
export envs_dirs="/data_storage/wyj/systems/envs"
export HF_HOME="/data_storage/wyj/systems/huggingface"
export HTTP_PROXY="http://100.68.168.184:3128"
export HTTPS_PROXY="http://100.68.168.184:3128"
source activate archer
export PROJ_DIR="/data_storage/wyj/jxl/ArcherCodeR"
cd "${PROJ_DIR}"
EXP_SUFFIX="from-trajentropy-base-step105-plus40-to145" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl/Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl/global_step_105" \
  actor_rollout_ref.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.total_training_steps=145
