#!/usr/bin/env python3
"""
Simple NCCL Test - 专注于测试分布式通信核心功能
"""

import os
import sys
import time
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def get_env_info():
    """获取环境信息"""
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    print(f"🔍 Environment Info:")
    print(f"   Hostname: {socket.gethostname()}")
    print(f"   Rank: {rank}/{world_size}, Local Rank: {local_rank}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   PyTorch: {torch.__version__}")
    
    # NCCL settings
    nccl_settings = [
        'NCCL_DEBUG', 'NCCL_IB_DISABLE', 'NCCL_SOCKET_IFNAME',
        'NCCL_TIMEOUT', 'MASTER_ADDR', 'MASTER_PORT'
    ]
    print(f"   NCCL Settings:")
    for setting in nccl_settings:
        value = os.environ.get(setting, 'Not set')
        print(f"     {setting}: {value}")
    
    return rank, world_size, local_rank


def test_basic_nccl(rank, world_size, local_rank):
    """测试基础NCCL通信"""
    if world_size == 1:
        print("⚠️  Single process - no distributed test needed")
        return True
    
    print(f"\n🚀 Rank {rank}: Starting NCCL tests...")
    
    try:
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"   Rank {rank}: Using device {device}")
        
        # Initialize distributed
        print(f"   Rank {rank}: Initializing process group...")
        start_time = time.time()
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5)
        )
        
        init_time = time.time() - start_time
        print(f"   Rank {rank}: ✅ Process group initialized in {init_time:.2f}s")
        
        # Test 1: Small all-reduce
        print(f"   Rank {rank}: Testing small all-reduce...")
        tensor = torch.ones(10, device=device) * rank
        
        start_time = time.time()
        dist.all_reduce(tensor)
        elapsed = time.time() - start_time
        
        expected_sum = sum(range(world_size))
        if torch.allclose(tensor, torch.ones(10, device=device) * expected_sum):
            print(f"   Rank {rank}: ✅ Small all-reduce passed ({elapsed:.3f}s)")
        else:
            print(f"   Rank {rank}: ❌ Small all-reduce failed")
            return False
        
        # Test 2: Broadcast
        print(f"   Rank {rank}: Testing broadcast...")
        if rank == 0:
            bcast_tensor = torch.arange(100, dtype=torch.float32, device=device)
        else:
            bcast_tensor = torch.zeros(100, device=device)
        
        start_time = time.time()
        dist.broadcast(bcast_tensor, src=0)
        elapsed = time.time() - start_time
        
        expected = torch.arange(100, dtype=torch.float32, device=device)
        if torch.allclose(bcast_tensor, expected):
            print(f"   Rank {rank}: ✅ Broadcast passed ({elapsed:.3f}s)")
        else:
            print(f"   Rank {rank}: ❌ Broadcast failed")
            return False
        
        # Test 3: Large tensor all-reduce
        print(f"   Rank {rank}: Testing large tensor all-reduce...")
        large_sizes = [1024*1024, 4*1024*1024]  # 4MB, 16MB
        
        for size in large_sizes:
            large_tensor = torch.randn(size, device=device)
            
            start_time = time.time()
            dist.all_reduce(large_tensor)
            elapsed = time.time() - start_time
            
            bandwidth_gb_s = (size * 4 * 2 * (world_size - 1)) / (elapsed * 1024**3)  # Rough estimate
            print(f"   Rank {rank}: ✅ Large tensor ({size:,} elements) all-reduce: {elapsed:.3f}s, ~{bandwidth_gb_s:.2f} GB/s")
        
        # Test 4: All-gather
        print(f"   Rank {rank}: Testing all-gather...")
        gather_tensor = torch.ones(1000, device=device) * rank
        gathered = [torch.zeros(1000, device=device) for _ in range(world_size)]
        
        start_time = time.time()
        dist.all_gather(gathered, gather_tensor)
        elapsed = time.time() - start_time
        
        success = True
        for i, t in enumerate(gathered):
            if not torch.allclose(t, torch.ones(1000, device=device) * i):
                success = False
                break
        
        if success:
            print(f"   Rank {rank}: ✅ All-gather passed ({elapsed:.3f}s)")
        else:
            print(f"   Rank {rank}: ❌ All-gather failed")
            return False
        
        # Cleanup
        dist.destroy_process_group()
        print(f"   Rank {rank}: 🎉 All NCCL tests passed!")
        
        return True
        
    except Exception as e:
        print(f"   Rank {rank}: ❌ NCCL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 Simple NCCL Communication Test")
    print("=" * 50)
    
    rank, world_size, local_rank = get_env_info()
    
    success = test_basic_nccl(rank, world_size, local_rank)
    
    if success:
        print(f"\n🎉 Rank {rank}: All tests completed successfully!")
        return 0
    else:
        print(f"\n💥 Rank {rank}: Tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 