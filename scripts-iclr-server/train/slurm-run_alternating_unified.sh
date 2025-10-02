#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=512GB
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=./output/ArcherCodeR/Alternating-Training/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/Alternating-Training/slurm_error_%j.txt
#SBATCH --job-name=alternating-training

# 导入环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env"
    echo "Your WANDB_API_KEY is: $WANDB_API_KEY"
    echo "Your HF_TOKEN is: $HF_TOKEN"
else
    echo "Warning: .env file not found. Please create .env file with WANDB_API_KEY and HF_TOKEN"
fi

# Clear Ray environment variables to force local cluster creation
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1

# Clear AMD GPU environment variables to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

# Make sure output directory exists for SLURM logs
mkdir -p ./output/ArcherCodeR/Alternating-Training

# Stop any existing Ray processes and clean up
ray stop --force 2>/dev/null || true
sleep 5  # Wait for Ray to fully shut down

# Kill any remaining Ray processes
pkill -f ray:: 2>/dev/null || true
pkill -f "ray start" 2>/dev/null || true
pkill -f "ray.worker" 2>/dev/null || true

# Clean up Ray temporary files
rm -rf /tmp/ray 2>/dev/null || true
rm -rf /dev/shm/ray* 2>/dev/null || true

# Default parameters for alternating training
ALGORITHMS="intuitor,grpo"
STEPS_PER_PHASE=50
START_ALGORITHM="intuitor"
TOTAL_EPOCHS=4
KL_MODE="no-kl"
PROJECT_NAME="ArcherCodeR"
CONFIG_NAME="alternating_official"

# Parse command line arguments to override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        --steps-per-phase)
            STEPS_PER_PHASE="$2"
            shift 2
            ;;
        --start-with)
            START_ALGORITHM="$2"
            shift 2
            ;;
        --total-epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --kl-mode)
            KL_MODE="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --test-mode)
            TEST_MODE="--test-mode"
            shift
            ;;
        --dataset-limit)
            DATASET_LIMIT="--dataset-limit $2"
            shift 2
            ;;
        *)
            # Pass through any other arguments
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "🚀 SLURM ALTERNATING TRAINING CONFIGURATION:"
echo "🤖 Algorithms: ${ALGORITHMS}"
echo "🔄 Steps per phase: ${STEPS_PER_PHASE}"
echo "🎯 Starting algorithm: ${START_ALGORITHM}"
echo "📊 Total epochs: ${TOTAL_EPOCHS}"
echo "🔧 KL mode: ${KL_MODE}"
echo "🏷️  Project name: ${PROJECT_NAME}"
echo "⚙️  Configuration: ${CONFIG_NAME}"
echo "🐍 Python: /data/xuandong_zhao/anaconda3/envs/archer/bin/python"
echo "💻 Working directory: $(pwd)"

bash scripts/train/run_alternating_unified.sh \
    --algorithms "$ALGORITHMS" \
    --steps-per-phase "$STEPS_PER_PHASE" \
    --start-with "$START_ALGORITHM" \
    --total-epochs "$TOTAL_EPOCHS" \
    --kl-mode "$KL_MODE" \
    --project-name "$PROJECT_NAME" \
    --config "$CONFIG_NAME" \
    $TEST_MODE \
    $DATASET_LIMIT \
    $EXTRA_ARGS 



# sbatch scripts/train/slurm-run_alternating_unified.sh --algorithms "intuitor,grpo" --steps-per-phase 20 --start-with grpo --kl-mode no-kl

# sbatch scripts/train/slurm-run_alternating_unified.sh --algorithms "intuitor,grpo" --steps-per-phase 40 --start-with grpo --kl-mode no-kl


# 使用自定义检查点保留数量的示例:
# sbatch scripts/train/slurm-run_alternating_unified.sh \
#     --algorithms "intuitor,grpo" \
#     --steps-per-phase 20 \
#     --start-with grpo \
#     --kl-mode no-kl \
#     trainer.max_actor_ckpt_to_keep=5 \
#     trainer.max_critic_ckpt_to_keep=3
