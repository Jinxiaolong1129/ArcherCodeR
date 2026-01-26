#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=512GB
#SBATCH --gpus=8
#SBATCH --time=4:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=./output/ArcherCodeR/Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-v2-rep/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-v2-rep/slurm_error_%j.txt
#SBATCH --job-name=prob-disparity-v2-rep

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
unset RAY_CLUSTER_NAME
unset RAY_NODE_IP_ADDRESS
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0

# Clear AMD GPU environment variables to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

# Make sure output directory exists for SLURM logs
mkdir -p ./output/ArcherCodeR/Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-v2-rep

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

# Run the training script with explicit Ray CPU configuration (FROM SCRATCH)
bash scripts-iclr-server/prob_disparity/prob_disparity_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray_v2_rep.sh ray_init.num_cpus=160



