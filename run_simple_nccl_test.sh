#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gpus-per-node=8
#SBATCH --time=30:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --exclude=compute-837
#SBATCH --job-name=simple-nccl-test
#SBATCH --output=simple_nccl_test_%j.txt
#SBATCH --error=simple_nccl_test_%j_error.txt

echo "🚀 Simple NCCL Communication Test"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Date: $(date)"
echo ""

# Activate conda environment
source /data/xuandong_zhao/anaconda3/etc/profile.d/conda.sh
conda activate archer

# Setup distributed environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "Environment:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  Node: $(hostname)"
echo ""

# Test 1: With InfiniBand (default)
echo "🧪 Test 1: NCCL with InfiniBand"
echo "----------------------------------------"
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=300
unset NCCL_IB_DISABLE
unset NCCL_SOCKET_IFNAME

echo "NCCL Settings:"
echo "  NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-Not set}"
echo "  NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-Not set}"
echo ""

srun python simple_nccl_test.py

echo ""
echo "🧪 Test 2: NCCL without InfiniBand (Ethernet only)"
echo "----------------------------------------"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

echo "NCCL Settings:"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo ""

srun python simple_nccl_test.py

echo ""
echo "✅ Simple NCCL tests completed!"
echo "End time: $(date)" 