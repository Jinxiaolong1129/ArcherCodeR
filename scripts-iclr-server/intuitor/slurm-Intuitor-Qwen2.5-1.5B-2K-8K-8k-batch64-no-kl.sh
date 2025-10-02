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
#SBATCH --output=./output/ArcherCodeR/Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl/slurm_error_%j.txt
#SBATCH --job-name=intuitor-qwen2.5-1.5b-2k-8k-8k-batch64-no-kl



# 导入环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env"
    echo "Your WANDB_API_KEY is: $WANDB_API_KEY"
    echo "Your HF_TOKEN is: $HF_TOKEN"
else
    echo "Warning: .env file not found. Please create .env file with WANDB_API_KEY and HF_TOKEN"
fi


# Clear ALL Ray environment variables to force local cluster creation
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
unset RAY_CLUSTER_NAME
unset RAY_REDIS_ADDRESS
unset RAY_GCS_ADDRESS
unset RAY_RAYLET_PID
unset RAY_PLASMA_STORE_SOCKET_NAME
unset RAY_RAYLET_SOCKET_NAME
unset RAY_NODE_IP_ADDRESS
unset RAY_TMPDIR
export RAY_DISABLE_IMPORT_WARNING=1

# Clear AMD GPU environment variables to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

# Make sure output directory exists for SLURM logs
mkdir -p ./output/ArcherCodeR/Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl

# Stop any existing Ray processes and clean up completely
echo "Cleaning up Ray environment..."
ray stop --force 2>/dev/null || true

# Remove Ray temp directories (only user-owned files)
echo "Cleaning Ray temporary files safely..."
# 只清理当前用户的Ray临时文件
if [ -d "/tmp" ] && [ -w "/tmp" ]; then
    find /tmp -name "ray*" -user $(whoami) -exec rm -rf {} + 2>/dev/null || true
fi
# 清理用户主目录下的Ray文件
rm -rf ~/.ray* 2>/dev/null || true
# 清理当前工作目录下的Ray文件
rm -rf ./ray_* 2>/dev/null || true

# Kill any remaining Ray processes
pkill -f ray:: 2>/dev/null || true
sleep 5

# 强制Ray创建本地集群的环境变量
export RAY_ADDRESS=""  # 明确设置为空，强制本地集群
export RAY_DEDUP_LOGS=0

echo "Ray environment prepared for local cluster creation"

bash scripts/intuitor/intuitor_Qwen-1.5B-2k-8k-8k-batch64-no-kl.sh
