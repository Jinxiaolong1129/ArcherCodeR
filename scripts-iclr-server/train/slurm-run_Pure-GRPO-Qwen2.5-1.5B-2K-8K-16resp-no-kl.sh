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
#SBATCH --output=./output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-debug-v2/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-debug-v2/slurm_error_%j.txt
#SBATCH --job-name=pure-grpo-qwen2.5-1.5b-2k-8k-16resp-no-kl

# 导入环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Loaded environment variables from .env"
    echo "🔑 WANDB_API_KEY: ${WANDB_API_KEY:0:8}..."
    echo "🔑 HF_TOKEN: ${HF_TOKEN:0:8}..."
else
    echo "⚠️  Warning: .env file not found. Please create .env file with WANDB_API_KEY and HF_TOKEN"
fi

# Clear Ray environment variables to force local cluster creation
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1

# Clear AMD GPU environment variables to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

# Navigate to project directory
cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

# Make sure output directory exists for SLURM logs
mkdir -p ./output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl

# Stop any existing Ray processes
ray stop --force 2>/dev/null || true

echo "🚀 Starting Pure GRPO Training Job"
echo "📅 Job started at: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🎯 Job ID: $SLURM_JOB_ID"
echo "🔧 CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "💾 Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "🎮 GPUs allocated: $SLURM_GPUS"

# Run the training script with explicit Ray CPU configuration
bash scripts-iclr-server/train/run_Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl.sh

echo "✅ Pure GRPO Training Job completed at: $(date)"
