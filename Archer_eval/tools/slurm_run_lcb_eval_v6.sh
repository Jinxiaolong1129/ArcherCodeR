#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs/slurm_run_lcb_eval_v6_%j.txt
#SBATCH --error=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs/slurm_run_lcb_eval_v6_error_%j.txt
#SBATCH --job-name=lcb-eval-v6

# 创建日志目录
mkdir -p /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/tools/logs

# 导入环境变量
if [ -f /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env ]; then
    export $(grep -v '^#' /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env | xargs)
    echo "Loaded environment variables from .env"
fi

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

# 设置 NUM_WORKERS 为 CPU 数量
export NUM_WORKERS=80

bash tools/run_lcb_eval_v6.sh 2>&1 | tee tools/logs/run_lcb_eval_v6_${SLURM_JOB_ID}.log

echo "End time: $(date)"





