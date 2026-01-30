#!/usr/bin/env bash
set -euo pipefail

# Comparison script for Intuitor vs GRPO using the same reward functions
# This script runs both algorithms with identical configurations except for the advantage estimator

echo "🚀 Starting Intuitor vs GRPO comparison..."

# Common configuration
export WANDB_API_KEY=${WANDB_API_KEY:-YOUR_WANDB_API_KEY}
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

# Data paths
TRAIN_DATA=${TRAIN_DATA:-$HOME/data/code/train.parquet}
VAL_DATA=${VAL_DATA:-$HOME/data/code/test.parquet}

# Model configuration
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}

# Training configuration
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
N_GPUS=${N_GPUS:-8}
EPOCHS=${EPOCHS:-1}

# Common arguments
COMMON_ARGS=(
    data.train_files="$TRAIN_DATA"
    data.val_files="$VAL_DATA"
    data.train_batch_size=$BATCH_SIZE
    data.max_prompt_length=$MAX_PROMPT_LEN
    data.max_response_length=$MAX_RESPONSE_LEN
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.reward_fn_key=data_source
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.optim.lr=3e-6
    actor_rollout_ref.actor.optim.warmup_style=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.005
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    reward_model.enable=False
    reward_model.reward_manager=wizard
    reward_model.use_general_reward=True
    trainer.critic_warmup=0
    trainer.val_before_train=False
    trainer.n_gpus_per_node=$N_GPUS
    trainer.nnodes=1
    trainer.logger='["console","wandb"]'
    trainer.project_name=verl_comparison
    trainer.save_freq=10
    trainer.test_freq=10
    trainer.total_epochs=$EPOCHS
)

# Function to run experiment
run_experiment() {
    local algorithm=$1
    local use_general_reward=$2
    local experiment_name="comparison_${algorithm}_$(date +%Y%m%d_%H%M%S)"
    
    echo "🎯 Running $algorithm with use_general_reward=$use_general_reward..."
    
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$algorithm \
        algorithm.norm_adv_by_std_in_grpo=true \
        reward_model.use_general_reward=$use_general_reward \
        trainer.experiment_name=$experiment_name \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "logs/${experiment_name}.log"
    
    echo "✅ Completed $algorithm experiment: $experiment_name"
}

# Create logs directory
mkdir -p logs

# Run GRPO with WizardRewardManager + rewards/general_reward.py
# This provides rich analysis: thinking tokens, repetition detection, etc.
echo "🔥 Starting GRPO experiment..."
run_experiment "grpo" "true"

# Run Intuitor with WizardRewardManager (uses rewards/general_reward.py for validation)
# Training uses self-certainty, validation uses same reward function as GRPO
echo "🧠 Starting Intuitor experiment..."
run_experiment "intuitor" "false"  # Intuitor handles this automatically

echo "🎉 All experiments completed! Check logs/ directory for results."
echo "📊 Compare the results in your WandB project: verl_comparison"
