#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8  
#SBATCH --mem=128GB
#SBATCH --gpus-per-node=8
#SBATCH --time=30:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --job-name=nccl-test-no-ib
#SBATCH --output=nccl_test_no_ib_out_%j.txt
#SBATCH --error=nccl_test_no_ib_error_%j.txt

echo "🚀 NCCL Test WITHOUT InfiniBand"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Date: $(date)"

# Activate conda environment
source /data/xuandong_zhao/anaconda3/etc/profile.d/conda.sh
conda activate archer

# Disable InfiniBand and force Ethernet
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# More permissive timeout settings
export NCCL_TIMEOUT=300
export NCCL_NET_GDR_LEVEL=0

# Setup distributed environment for SLURM
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "Network settings:"
echo "  NCCL_IB_DISABLE: $NCCL_IB_DISABLE"  
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  WORLD_SIZE: $WORLD_SIZE"

echo ""
echo "🧪 Testing NCCL with Ethernet only..."

srun python -c "
import torch
import torch.distributed as dist
import os
import time
from datetime import timedelta

rank = int(os.environ.get('SLURM_PROCID', 0))
world_size = int(os.environ.get('SLURM_NTASKS', 1))
local_rank = int(os.environ.get('SLURM_LOCALID', 0))

print(f'Process {rank}/{world_size}, Local rank {local_rank}')

if world_size > 1:
    try:
        torch.cuda.set_device(local_rank)
        print(f'Rank {rank}: Initializing distributed...')
        
        dist.init_process_group(
            backend='nccl',
            rank=rank, 
            world_size=world_size,
            timeout=timedelta(minutes=5)
        )
        
        print(f'Rank {rank}: ✅ Distributed init successful!')
        
        # Test basic operations
        device = torch.device(f'cuda:{local_rank}')
        
        # Test 1: Small tensor
        tensor = torch.ones(100, device=device) * rank
        print(f'Rank {rank}: Testing small tensor all-reduce...')
        dist.all_reduce(tensor)
        print(f'Rank {rank}: ✅ Small tensor all-reduce: {tensor[0].item()}')
        
        # Test 2: Larger tensor  
        large_tensor = torch.randn(1024*1024, device=device)  # 4MB
        print(f'Rank {rank}: Testing large tensor all-reduce...')
        start_time = time.time()
        dist.all_reduce(large_tensor)
        elapsed = time.time() - start_time
        print(f'Rank {rank}: ✅ Large tensor all-reduce: {elapsed:.3f}s')
        
        # Test 3: Broadcast
        if rank == 0:
            bcast_tensor = torch.arange(1000, dtype=torch.float32, device=device)
        else:
            bcast_tensor = torch.zeros(1000, device=device)
        
        print(f'Rank {rank}: Testing broadcast...')
        dist.broadcast(bcast_tensor, src=0)
        print(f'Rank {rank}: ✅ Broadcast: sum={bcast_tensor.sum().item()}')
        
        dist.destroy_process_group()
        print(f'Rank {rank}: 🎉 All tests passed!')
        
    except Exception as e:
        print(f'Rank {rank}: ❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
else:
    print('Single process mode - no distributed test needed')
"

echo ""
echo "✅ NCCL Test (No InfiniBand) completed!"
echo "End time: $(date)" 