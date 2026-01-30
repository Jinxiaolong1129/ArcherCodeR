#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192GB
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs/slurm_eval_missing_minerva_%j.txt
#SBATCH --error=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs/slurm_eval_missing_minerva_error_%j.txt
#SBATCH --job-name=minerva-miss

# 创建日志目录
mkdir -p /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs

# 导入环境变量
if [ -f /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env ]; then
    export $(grep -v '^#' /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env | xargs)
    echo "Loaded environment variables from .env"
fi

# Clear Ray
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1

# 清除 AMD GPU 环境变量
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval

ray stop --force 2>/dev/null || true

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Environment check:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
echo "  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"

bash tools/batch_eval_missing_minerva.sh 2>&1 | tee tools/logs/batch_eval_missing_minerva_${SLURM_JOB_ID}.log


