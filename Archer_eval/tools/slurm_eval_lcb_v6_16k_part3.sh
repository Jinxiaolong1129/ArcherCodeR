#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs/slurm_eval_lcb_v6_16k_part3_%j.txt
#SBATCH --error=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs/slurm_eval_lcb_v6_16k_part3_error_%j.txt
#SBATCH --job-name=lcbv6-16k-p3

# 创建日志目录
mkdir -p /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs

# 导入环境变量
if [ -f /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env ]; then
    export $(grep -v '^#' /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env | xargs)
    echo "Loaded environment variables from .env"
fi

# Clear Ray environment variables to force local cluster creation
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1

# 关键修复: 彻底清除 AMD GPU 环境变量，避免与 CUDA 冲突
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval

ray stop --force 2>/dev/null || true

# 再次确认清除（ray stop 后环境变量可能重置）
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Environment check:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
echo "  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"

bash tools/batch_merge_eval_lcb_v6_16k_part3.sh 2>&1 | tee tools/logs/batch_eval_lcb_v6_16k_part3_${SLURM_JOB_ID}.log





