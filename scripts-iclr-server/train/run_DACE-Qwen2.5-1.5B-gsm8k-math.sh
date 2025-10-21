#!/bin/bash
set -x

# ================================
# DACE Training Script
# Difficulty-Aware Certainty Exploration
# ================================

# Data paths (adjust to your setup)
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"

# DACE-specific hyperparameters (paper defaults)
# α_scale: Scaling factor for intrinsic reward (paper default: 0.05)
#   - Controls the magnitude of the intrinsic reward signal
#   - Higher values = stronger exploration/exploitation signal
DACE_ALPHA_SCALE=0.05

# β_threshold: Difficulty threshold (paper default: 0.4)
#   - Determines the boundary between "easy" and "hard" tasks
#   - Tasks with difficulty > β_threshold encourage exploration (low certainty)
#   - Tasks with difficulty < β_threshold encourage exploitation (high certainty)
DACE_BETA_THRESHOLD=0.4

# Training configuration
TRAIN_BATCH_SIZE=1024
N_RESPONSES=8  # Number of responses per prompt (for difficulty estimation)

# Run DACE training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=dace \
    algorithm.dace_alpha_scale=$DACE_ALPHA_SCALE \
    algorithm.dace_beta_threshold=$DACE_BETA_THRESHOLD \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_dace_experiment' \
    trainer.experiment_name='dace_qwen2.5_1.5b_gsm8k_math' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

# ================================
# DACE Method Explanation
# ================================
# 
# DACE (Difficulty-Aware Certainty Exploration) adaptively combines:
# 1. External rewards (R_ext): Binary correctness signal (0 or 1)
# 2. Intrinsic rewards (R_int): Certainty-based exploration signal
#
# Key Components:
# - Difficulty estimation: diff(x) = 1 - mean(verify(y)) per prompt
# - Certainty metric: C(y,x) = -mean(log_prob(y|x))
# - Adaptive coefficient: α(x) = α_scale * sign(β_threshold - diff(x))
# - Intrinsic reward: R_int = α(x) * C(y,x)
# - Total reward: R_total = R_ext + R_int
#
# Behavior:
# - For HARD tasks (diff > β_threshold): α < 0 → encourages EXPLORATION (lower certainty)
# - For EASY tasks (diff < β_threshold): α > 0 → encourages EXPLOITATION (higher certainty)
#
# Hyperparameter Tuning:
# - α_scale: Start with 0.1, increase (0.2-0.5) for stronger intrinsic signal
# - β_threshold: Start with 0.5, adjust based on dataset difficulty distribution
#   - Lower (0.3-0.4): More tasks treated as "easy" → more exploitation
#   - Higher (0.6-0.7): More tasks treated as "hard" → more exploration
#

