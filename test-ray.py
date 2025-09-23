#!/usr/bin/env python3
import os
import sys
import ray
import time

def test_ray_init():
    """Test Ray initialization with different configurations"""
    
    print("Testing Ray initialization...")
    
    # Test 1: Basic initialization
    try:
        print("\n1. Testing basic Ray init...")
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        print("✅ Basic Ray init successful")
        ray.shutdown()
    except Exception as e:
        print(f"❌ Basic Ray init failed: {e}")
        return False
    
    # Test 2: With reduced resources
    try:
        print("\n2. Testing Ray init with limited resources...")
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            num_cpus=2,
            num_gpus=1,
            object_store_memory=1000000000  # 1GB
        )
        print("✅ Limited resource Ray init successful")
        ray.shutdown()
    except Exception as e:
        print(f"❌ Limited resource Ray init failed: {e}")
        return False
    
    # Test 3: With custom temp directory
    try:
        print("\n3. Testing Ray init with custom temp dir...")
        temp_dir = "/home/ec2-user/ray_temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            temp_dir=temp_dir,
            _system_config={
                "object_store_memory": 1000000000,
                "plasma_directory": temp_dir
            }
        )
        print("✅ Custom temp dir Ray init successful")
        ray.shutdown()
    except Exception as e:
        print(f"❌ Custom temp dir Ray init failed: {e}")
        return False
    
    print("\n✅ All Ray tests passed!")
    return True

def check_system_resources():
    """Check system resources and configuration"""
    import subprocess
    import shutil
    
    print("System Resource Check:")
    print("=" * 50)
    
    # Check /tmp space
    tmp_usage = shutil.disk_usage("/tmp")
    print(f"/tmp space: {tmp_usage.free / (1024**3):.1f}GB free")
    
    # Check /dev/shm space
    try:
        shm_usage = shutil.disk_usage("/dev/shm")
        print(f"/dev/shm space: {shm_usage.free / (1024**3):.1f}GB free")
    except:
        print("/dev/shm not accessible")
    
    # Check memory
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        print("Memory usage:")
        print(result.stdout)
    except:
        print("Could not check memory")
    
    # Check GPU count
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line])
        print(f"Available GPUs: {gpu_count}")
    except:
        print("Could not check GPU count")

if __name__ == "__main__":
    print("Ray Diagnostic Tool")
    print("=" * 50)
    
    check_system_resources()
    print()
    
    if test_ray_init():
        print("\n🎉 Ray is working! The issue might be specific to your training configuration.")
    else:
        print("\n💥 Ray has fundamental issues on this system.")
        
    print("\nIf Ray is working, try reducing resources in your training script:")
    print("- Reduce trainer.n_gpus_per_node to actual GPU count")
    print("- Reduce tensor_model_parallel_size to 1")
    print("- Reduce batch sizes")