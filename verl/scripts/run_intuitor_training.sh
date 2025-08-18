#!/bin/bash
set -x

# Environment setup
export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

# Default configuration - can be overridden by command line arguments
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B"}
TRAIN_DATA=${TRAIN_DATA:-"$HOME/data/math/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/data/math/test.parquet"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"archercoder_intuitor"}
PROJECT_NAME=${PROJECT_NAME:-"verl"}
N_GPUS=${N_GPUS:-8}
N_NODES=${N_NODES:-1}

# Training hyperparameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-3e-6}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# PPO specific parameters
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

# KL loss parameters
USE_KL_LOSS=${USE_KL_LOSS:-true}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}

# Memory and performance settings
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-false}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-true}

# Logging and checkpointing
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-"['console','wandb']"}

echo "=========================================="
echo "🚀 Starting INTUITOR Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Train Data: $TRAIN_DATA"
echo "Val Data: $VAL_DATA"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: ${N_GPUS}x${N_NODES} = $((N_GPUS * N_NODES))"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "=========================================="

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    2>&1 | tee "verl_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "✅ Training completed!"
echo "Log saved to: verl_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "==========================================" 