#!/usr/bin/env python3
"""
NCCL Diagnostics Test Suite
测试分布式通信的各个层面，定位问题所在
"""

import os
import sys
import time
import socket
import subprocess
import argparse
from datetime import timedelta
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np

def setup_environment():
    """设置环境变量"""
    print("🔧 Setting up environment...")
    
    # Basic NCCL settings
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    os.environ.setdefault('NCCL_DEBUG_SUBSYS', 'ALL')
    
    # Timeout settings - more lenient for testing
    os.environ.setdefault('NCCL_TIMEOUT', '300')  # 5 minutes
    os.environ.setdefault('NCCL_IB_TIMEOUT', '100')
    os.environ.setdefault('NCCL_IB_RETRY_CNT', '20')
    
    # Network settings
    os.environ.setdefault('NCCL_IB_GID_INDEX', '3')
    os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0')
    
    print(f"   NCCL_DEBUG: {os.environ.get('NCCL_DEBUG')}")
    print(f"   NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def get_system_info():
    """获取系统信息"""
    print("\n📊 System Information:")
    print(f"   Hostname: {socket.gethostname()}")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    # Check InfiniBand
    try:
        ib_info = subprocess.run(['ibstat'], capture_output=True, text=True, timeout=10)
        if ib_info.returncode == 0:
            print("   InfiniBand: Available")
            # Count active ports
            active_ports = ib_info.stdout.count('State: Active')
            print(f"   IB Active Ports: {active_ports}")
        else:
            print("   InfiniBand: Not available or not accessible")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   InfiniBand: Cannot check (ibstat not found)")

def test_basic_cuda():
    """测试基础CUDA功能"""
    print("\n🧪 Testing Basic CUDA...")
    
    try:
        # Test basic CUDA operations
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print("   ✅ Basic CUDA operations: PASSED")
        
        # Test multi-GPU if available
        if torch.cuda.device_count() > 1:
            x1 = torch.randn(1000, 1000, device='cuda:0')
            x2 = torch.randn(1000, 1000, device='cuda:1')
            print("   ✅ Multi-GPU tensor creation: PASSED")
        
        return True
    except Exception as e:
        print(f"   ❌ Basic CUDA test FAILED: {e}")
        return False

def init_distributed(backend='nccl', timeout_minutes=10):
    """初始化分布式环境"""
    print(f"\n🌐 Initializing distributed environment (backend={backend})...")
    
    try:
        # Get distributed info from environment
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"   Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        
        if world_size == 1:
            print("   ⚠️  Single process mode - skipping distributed init")
            return True, rank, world_size, local_rank
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # Initialize process group with extended timeout
        timeout = timedelta(minutes=timeout_minutes)
        
        dist.init_process_group(
            backend=backend,
            timeout=timeout,
            rank=rank,
            world_size=world_size
        )
        
        print(f"   ✅ Distributed initialization: PASSED")
        print(f"   Backend: {dist.get_backend()}")
        return True, rank, world_size, local_rank
        
    except Exception as e:
        print(f"   ❌ Distributed initialization FAILED: {e}")
        return False, 0, 1, 0

def test_basic_nccl_ops(rank, world_size, local_rank):
    """测试基础NCCL操作"""
    print(f"\n🔗 Testing Basic NCCL Operations...")
    
    if world_size == 1:
        print("   ⚠️  Single process - skipping NCCL tests")
        return True
    
    device = torch.device(f'cuda:{local_rank}')
    
    try:
        # Test 1: All-reduce
        print("   Testing all-reduce...")
        tensor = torch.ones(10, device=device) * rank
        original = tensor.clone()
        dist.all_reduce(tensor)
        expected_sum = sum(range(world_size))
        
        if torch.allclose(tensor, torch.ones(10, device=device) * expected_sum):
            print("   ✅ All-reduce: PASSED")
        else:
            print(f"   ❌ All-reduce: FAILED (got {tensor[0]}, expected {expected_sum})")
            return False
        
        # Test 2: Broadcast
        print("   Testing broadcast...")
        if rank == 0:
            tensor = torch.arange(10, dtype=torch.float32, device=device)
        else:
            tensor = torch.zeros(10, device=device)
        
        dist.broadcast(tensor, src=0)
        expected = torch.arange(10, dtype=torch.float32, device=device)
        
        if torch.allclose(tensor, expected):
            print("   ✅ Broadcast: PASSED")
        else:
            print("   ❌ Broadcast: FAILED")
            return False
        
        # Test 3: All-gather
        print("   Testing all-gather...")
        tensor = torch.ones(5, device=device) * rank
        gathered = [torch.zeros(5, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        
        for i, t in enumerate(gathered):
            if not torch.allclose(t, torch.ones(5, device=device) * i):
                print(f"   ❌ All-gather: FAILED at rank {i}")
                return False
        
        print("   ✅ All-gather: PASSED")
        
        return True
        
    except Exception as e:
        print(f"   ❌ NCCL operations FAILED: {e}")
        return False

def test_large_tensor_ops(rank, world_size, local_rank):
    """测试大张量操作"""
    print(f"\n📊 Testing Large Tensor Operations...")
    
    if world_size == 1:
        print("   ⚠️  Single process - skipping large tensor tests")
        return True
    
    device = torch.device(f'cuda:{local_rank}')
    
    try:
        # Test with progressively larger tensors
        sizes = [1024, 1024*1024, 10*1024*1024]  # 4KB, 4MB, 40MB
        
        for size in sizes:
            print(f"   Testing tensor size: {size} elements ({size*4/1024/1024:.1f} MB)")
            
            tensor = torch.randn(size, device=device)
            start_time = time.time()
            dist.all_reduce(tensor)
            elapsed = time.time() - start_time
            
            print(f"   ✅ Size {size}: {elapsed:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Large tensor operations FAILED: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print(f"\n🤖 Testing Model Loading...")
    
    try:
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        
        # Test config loading
        print("   Loading model config...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        print(f"   ✅ Config loaded: {config.model_type}")
        
        # Test tokenizer loading
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        print(f"   ✅ Tokenizer loaded: {len(tokenizer)} tokens")
        
        # Test model loading (on CPU first)
        print("   Loading model to CPU...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model loaded: {num_params/1e9:.2f}B parameters")
        
        # Test moving to GPU
        if torch.cuda.is_available():
            print("   Moving model to GPU...")
            device = torch.device('cuda:0')
            model = model.to(device)
            print("   ✅ Model moved to GPU")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"   ❌ Model loading FAILED: {e}")
        return False, None, None

def test_fsdp_initialization(model, rank, world_size, local_rank):
    """测试FSDP初始化"""
    print(f"\n🔄 Testing FSDP Initialization...")
    
    if world_size == 1:
        print("   ⚠️  Single process - skipping FSDP test")
        return True
    
    if model is None:
        print("   ❌ No model available for FSDP test")
        return False
    
    try:
        device = torch.device(f'cuda:{local_rank}')
        
        # Move model to device
        model = model.to(device)
        
        # Test FSDP wrapping with simple policy
        print("   Wrapping model with FSDP...")
        
        # Use a simple wrap policy for testing
        from functools import partial
        
        # Try to identify transformer layers
        transformer_layer_cls = None
        for name, module in model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                transformer_layer_cls = type(module)
                break
        
        if transformer_layer_cls:
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_cls}
            )
            print(f"   Using transformer layer: {transformer_layer_cls}")
        else:
            wrap_policy = None
            print("   Using default wrap policy")
        
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=None,  # Disable mixed precision for testing
            device_id=local_rank,
            sync_module_states=True,
        )
        
        print("   ✅ FSDP initialization: PASSED")
        
        # Test a simple forward pass
        print("   Testing FSDP forward pass...")
        test_input = torch.randint(0, 1000, (2, 10), device=device)
        
        with torch.no_grad():
            output = fsdp_model(test_input)
        
        print("   ✅ FSDP forward pass: PASSED")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FSDP initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_fallback():
    """测试网络后端降级"""
    print(f"\n🔄 Testing Network Fallback Options...")
    
    try:
        # Test if we can fallback to Gloo
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if world_size == 1:
            print("   ⚠️  Single process - skipping network fallback test")
            return True
        
        print("   Testing Gloo backend...")
        
        # Destroy existing process group if any
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Try Gloo backend
        dist.init_process_group(
            backend='gloo',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5)
        )
        
        # Test basic operation
        tensor = torch.ones(10) * rank
        dist.all_reduce(tensor)
        
        print("   ✅ Gloo backend: PASSED")
        
        # Cleanup
        dist.destroy_process_group()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Network fallback FAILED: {e}")
        return False

def run_diagnostics():
    """运行完整的诊断测试"""
    parser = argparse.ArgumentParser(description='NCCL Diagnostics')
    parser.add_argument('--backend', default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--timeout', type=int, default=10, help='Timeout in minutes')
    parser.add_argument('--skip-model', action='store_true', help='Skip model loading tests')
    args = parser.parse_args()
    
    print("🚀 NCCL Diagnostics Test Suite")
    print("=" * 50)
    
    setup_environment()
    get_system_info()
    
    # Test 1: Basic CUDA
    if not test_basic_cuda():
        print("\n💥 Basic CUDA test failed - stopping")
        return 1
    
    # Test 2: Distributed initialization
    success, rank, world_size, local_rank = init_distributed(args.backend, args.timeout)
    if not success:
        print("\n💥 Distributed initialization failed")
        
        # Try fallback
        if args.backend == 'nccl':
            print("🔄 Trying fallback to Gloo...")
            success, rank, world_size, local_rank = init_distributed('gloo', args.timeout)
            if not success:
                return 1
        else:
            return 1
    
    # Test 3: Basic NCCL operations
    if not test_basic_nccl_ops(rank, world_size, local_rank):
        print("\n💥 Basic NCCL operations failed")
        if world_size > 1:
            test_network_fallback()
        return 1
    
    # Test 4: Large tensor operations
    if not test_large_tensor_ops(rank, world_size, local_rank):
        print("\n💥 Large tensor operations failed")
        return 1
    
    # Test 5: Model loading
    model, tokenizer = None, None
    if not args.skip_model:
        success, model, tokenizer = test_model_loading()
        if not success:
            print("\n⚠️  Model loading failed - continuing without model tests")
    
    # Test 6: FSDP initialization
    if model is not None and world_size > 1:
        if not test_fsdp_initialization(model, rank, world_size, local_rank):
            print("\n💥 FSDP initialization failed")
            return 1
    
    print("\n🎉 All tests completed!")
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return 0

if __name__ == "__main__":
    exit_code = run_diagnostics()
    sys.exit(exit_code) 