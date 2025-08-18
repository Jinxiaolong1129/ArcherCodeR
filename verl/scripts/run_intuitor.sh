#!/bin/bash

# INTUITOR Training Script for ArcherCodeR
# Usage: ./run_intuitor.sh [config_name]
# Example: ./run_intuitor.sh math_gsm8k

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
CONFIG_NAME=${1:-"default"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "Starting INTUITOR training with config: $CONFIG_NAME"
print_info "Script directory: $SCRIPT_DIR"
print_info "Project root: $PROJECT_ROOT"

# Environment setup
export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}
export ACCELERATE_LOG_LEVEL=${ACCELERATE_LOG_LEVEL:-"info"}
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration based on task
case $CONFIG_NAME in
    "math_gsm8k")
        print_info "Loading GSM8K math configuration..."
        MODEL_PATH="Qwen/Qwen2.5-3B"
        TRAIN_DATA="$HOME/data/gsm8k/train.parquet"
        VAL_DATA="$HOME/data/gsm8k/test.parquet"
        EXPERIMENT_NAME="intuitor_gsm8k"
        MAX_PROMPT_LENGTH=512
        MAX_RESPONSE_LENGTH=2048
        ;;
    "math_math500")
        print_info "Loading MATH500 configuration..."
        MODEL_PATH="Qwen/Qwen2.5-3B"
        TRAIN_DATA="$HOME/data/math500/train.parquet"
        VAL_DATA="$HOME/data/math500/test.parquet"
        EXPERIMENT_NAME="intuitor_math500"
        MAX_PROMPT_LENGTH=1024
        MAX_RESPONSE_LENGTH=3072
        ;;
    "code_humaneval")
        print_info "Loading HumanEval code configuration..."
        MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
        TRAIN_DATA="$HOME/data/humaneval/train.parquet"
        VAL_DATA="$HOME/data/humaneval/test.parquet"
        EXPERIMENT_NAME="intuitor_humaneval"
        MAX_PROMPT_LENGTH=1024
        MAX_RESPONSE_LENGTH=2048
        ;;
    "reasoning_arc")
        print_info "Loading ARC reasoning configuration..."
        MODEL_PATH="Qwen/Qwen2.5-7B"
        TRAIN_DATA="$HOME/data/arc/train.parquet"
        VAL_DATA="$HOME/data/arc/test.parquet"
        EXPERIMENT_NAME="intuitor_arc"
        MAX_PROMPT_LENGTH=512
        MAX_RESPONSE_LENGTH=1024
        ;;
    "default"|*)
        print_info "Loading default configuration..."
        MODEL_PATH="Qwen/Qwen2.5-3B"
        TRAIN_DATA="$HOME/data/train.parquet"
        VAL_DATA="$HOME/data/test.parquet"
        EXPERIMENT_NAME="intuitor_default"
        MAX_PROMPT_LENGTH=512
        MAX_RESPONSE_LENGTH=2048
        ;;
esac

# Common training parameters (can be overridden by environment variables)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-3e-6}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
N_GPUS=${N_GPUS:-8}
N_NODES=${N_NODES:-1}

# PPO parameters
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

# KL loss parameters
USE_KL_LOSS=${USE_KL_LOSS:-true}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-"low_var_kl"}

# Memory and performance
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-false}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-true}

# Logging and checkpointing
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
PROJECT_NAME=${PROJECT_NAME:-"verl"}
LOGGER=${LOGGER:-"['console','wandb']"}

# Validate data files
if [[ ! -f "$TRAIN_DATA" ]]; then
    print_warning "Training data file not found: $TRAIN_DATA"
    print_info "Please ensure your data files are in the correct location"
fi

if [[ ! -f "$VAL_DATA" ]]; then
    print_warning "Validation data file not found: $VAL_DATA"
    print_info "Please ensure your data files are in the correct location"
fi

# Print configuration summary
print_info "=== INTUITOR Training Configuration ==="
echo "Model: $MODEL_PATH"
echo "Train Data: $TRAIN_DATA"
echo "Val Data: $VAL_DATA"
echo "Experiment: $EXPERIMENT_NAME"
echo "GPUs: ${N_GPUS}x${N_NODES} = $((N_GPUS * N_NODES))"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Prompt Length: $MAX_PROMPT_LENGTH"
echo "Max Response Length: $MAX_RESPONSE_LENGTH"
echo "Rollout N: $ROLLOUT_N"
echo "Total Epochs: $TOTAL_EPOCHS"
print_info "======================================"

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/verl_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

print_info "Starting training... Log will be saved to: $LOG_FILE"

# Run training
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
    2>&1 | tee "$LOG_FILE"

# Check training result
if [ $? -eq 0 ]; then
    print_success "Training completed successfully!"
    print_info "Log saved to: $LOG_FILE"
    print_info "Check your experiment results in W&B: https://wandb.ai"
else
    print_error "Training failed! Check the log for details: $LOG_FILE"
    exit 1
fi 