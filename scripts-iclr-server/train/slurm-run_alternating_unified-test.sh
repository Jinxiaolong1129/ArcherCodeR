#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=512GB
#SBATCH --gpus=8
#SBATCH --time=12:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=./output/ArcherCodeR/Alternating-Test/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/Alternating-Test/slurm_error_%j.txt
#SBATCH --job-name=alternating-test

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
mkdir -p ./output/ArcherCodeR/Alternating-Test

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

echo "🧪 SLURM ALTERNATING TRAINING TEST MODE:"
echo "🤖 Algorithms: intuitor,grpo"
echo "🔄 Steps per phase: 1 (test mode)"
echo "🎯 Starting algorithm: intuitor"
echo "📊 Total epochs: 2"
echo "📊 Dataset limit: 500"
echo "🔧 KL mode: no-kl"
echo "🏷️  Project name: ArcherCodeR"
echo "⚙️  Configuration: alternating_official"
echo "🐍 Python: /data/xuandong_zhao/anaconda3/envs/archer/bin/python"
echo "💻 Working directory: $(pwd)"

# Run the alternating training script in test mode with Ray CPU configuration
bash scripts/train/run_alternating_unified.sh \
    --test-mode \
    --dataset-limit 500 \
    --kl-mode no-kl \
    --project-name "ArcherCodeR" \
    --config "alternating_official" \
    ray_init.num_cpus=160



