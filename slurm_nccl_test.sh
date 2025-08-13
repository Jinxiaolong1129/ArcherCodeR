#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8  
#SBATCH --mem=128GB
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --exclude=compute-837
#SBATCH --job-name=nccl-test
#SBATCH --output=nccl_test_out_%j.txt
#SBATCH --error=nccl_test_error_%j.txt

echo "🚀 NCCL Diagnostics Test Suite"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "Date: $(date)"

# Activate conda environment
source /data/xuandong_zhao/anaconda3/etc/profile.d/conda.sh
conda activate archer

# Verify Python and environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Basic environment settings
export PYTHONPATH="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR:$PYTHONPATH"

# NCCL Debug settings for diagnostics
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# More permissive settings for testing
export NCCL_TIMEOUT=600
export NCCL_IB_TIMEOUT=100  
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=0

# Setup distributed environment for SLURM
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

# Optional: Test without InfiniBand first
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0

echo ""
echo "🧪 Test 1: Single Process (Basic CUDA + Model Loading)"
python test_nccl_diagnostics.py

echo ""
echo "🧪 Test 2: Multi-Process NCCL (8 GPUs)"
if [ $SLURM_NTASKS -gt 1 ]; then
    # Run with proper SLURM distributed setup
    srun python test_nccl_diagnostics.py --backend=nccl --timeout=5
    
    echo ""
    echo "🧪 Test 3: Multi-Process Gloo Fallback (8 GPUs)"
    srun python test_nccl_diagnostics.py --backend=gloo --timeout=5
else
    echo "⚠️  Single task mode - skipping multi-process tests"
fi

echo ""
echo "🧪 Test 4: Simple torch.distributed test"
if [ $SLURM_NTASKS -gt 1 ]; then
    srun python -c "
import torch
import torch.distributed as dist
import os

# Initialize distributed
rank = int(os.environ.get('SLURM_PROCID', 0))
world_size = int(os.environ.get('SLURM_NTASKS', 1))
local_rank = int(os.environ.get('SLURM_LOCALID', 0))

print(f'Rank {rank}/{world_size}, Local rank {local_rank}')

if world_size > 1:
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        print(f'Rank {rank}: NCCL init successful')
        
        # Simple all-reduce test
        tensor = torch.ones(10, device=f'cuda:{local_rank}') * rank
        dist.all_reduce(tensor)
        print(f'Rank {rank}: All-reduce result: {tensor[0].item()}')
        
        dist.destroy_process_group()
    except Exception as e:
        print(f'Rank {rank}: NCCL failed: {e}')
else:
    print('Single process mode')
"
fi

echo ""
echo "✅ NCCL Diagnostics completed!"
echo "End time: $(date)" 