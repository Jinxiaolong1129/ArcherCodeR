#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=512GB
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --job-name=interactive-8gpu
#SBATCH --output=./output/interactive_node_%j.out
#SBATCH --error=./output/interactive_node_%j.err

# 打印分配的节点信息
echo "=========================================="
echo "Interactive Node Allocated!"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 8"
echo "Time: 24 hours"
echo ""
echo "To SSH into this node, run:"
echo "  ssh $SLURMD_NODENAME"
echo ""
echo "To check GPU status on the node:"
echo "  nvidia-smi"
echo ""
echo "Job started at: $(date)"
echo "=========================================="

# 保持作业运行 24 小时
# 你可以 ssh 到节点上运行任何程序
sleep 86400

echo "Job ended at: $(date)"


