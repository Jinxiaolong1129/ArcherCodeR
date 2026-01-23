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
#SBATCH --output=./output/ArcherCodeR/TrajectoryEntropy-LCB-Qwen2.5-1.5B-2k-8k-no-kl-v2/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/TrajectoryEntropy-LCB-Qwen2.5-1.5B-2k-8k-no-kl-v2/slurm_error_%j.txt
#SBATCH --job-name=trajectory-entropy-lcb-no-kl



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

# =====================================================================
# FIX: Set Ray temp directory to avoid /tmp disk space issues
# NOTE: Only set RAY_TMPDIR, NOT TMPDIR (to avoid Triton cache issues)
# =====================================================================
export RAY_TMPDIR=/data/xuandong_zhao/tmp/ray
mkdir -p $RAY_TMPDIR

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

# Make sure output directory exists for SLURM logs
mkdir -p ./output/ArcherCodeR/TrajectoryEntropy-LCB-Qwen2.5-1.5B-2k-8k-no-kl-v2

# Stop Ray and clean up old sessions to prevent disk space issues
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true
rm -rf $RAY_TMPDIR/session_* 2>/dev/null || true

bash scripts-iclr-server/trajectory_entropy/trajectory_entropy_Qwen-1.5B-lcb-train-no-kl.sh

