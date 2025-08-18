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
#SBATCH --output=./output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp/slurm_error_%j.txt
#SBATCH --job-name=archer-qwen2.5-1.5b-2k-8k-16resp

export HF_TOKEN=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
export WANDB_API_KEY=5c271ef60b4c4753def92be733cf80487f0c7e78

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
mkdir -p ./output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp

# Stop any existing Ray processes
ray stop --force 2>/dev/null || true

# Run the training script with explicit Ray CPU configuration
bash scripts/train/run_Archer-Qwen2.5-1.5B-2K-8K-16resp.sh ray_init.num_cpus=160 