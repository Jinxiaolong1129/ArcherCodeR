#!/usr/bin/env python3
"""
Test script to verify Ray initialization fixes
"""
import os
import sys
import time

# Add the project root to Python path
sys.path.insert(0, '/home/ec2-user/ArcherCodeR')

def test_ray_init():
    """Test Ray initialization with the same configuration as the training script"""
    
    # Set environment variables like in the training script
    os.environ.update({
        "RAY_DISABLE_IMPORT_WARNING": "1",
        "RAY_DEDUP_LOGS": "0", 
        "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "1",
        "RAY_raylet_start_wait_time_s": "60",
        "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER": "1",
        "RAY_DISABLE_STRICT_VERSION_CHECK": "1"
    })
    
    try:
        import ray
        print("🔄 Testing Ray initialization with training script configuration...")
        
        # Test with the same parameters as the training script
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true", 
                "NCCL_DEBUG": "WARN", 
                "VLLM_LOGGING_LEVEL": "WARN",
            }},
            num_cpus=64,  # Same as in the training script
            object_store_memory=int(2e9),  # 2GB
            _temp_dir="/tmp/ray_temp",
        )
        
        print("✅ Ray initialization successful!")
        print(f"📊 Ray cluster resources: {ray.cluster_resources()}")
        
        # Test creating a simple remote task
        @ray.remote
        def test_task():
            return "Hello from Ray worker!"
        
        result = ray.get(test_task.remote())
        print(f"✅ Ray remote task successful: {result}")
        
        ray.shutdown()
        print("✅ Ray shutdown successful!")
        return True
        
    except Exception as e:
        print(f"❌ Ray test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hydra_config():
    """Test if we can load the Hydra configuration"""
    try:
        import hydra
        from omegaconf import DictConfig
        
        # Change to the dapo directory to load config
        os.chdir('/home/ec2-user/ArcherCodeR/dapo')
        
        with hydra.initialize(config_path="config", version_base=None):
            cfg = hydra.compose(config_name="dapo_trainer")
            
        print("✅ Hydra configuration loaded successfully!")
        print(f"📊 Ray init config: num_cpus = {cfg.ray_init.num_cpus}")
        return True
        
    except Exception as e:
        print(f"❌ Hydra config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Ray initialization fixes...")
    print("=" * 50)
    
    # Test 1: Ray initialization
    ray_success = test_ray_init()
    print()
    
    # Test 2: Hydra configuration
    hydra_success = test_hydra_config()
    print()
    
    if ray_success and hydra_success:
        print("🎉 All tests passed! The Ray fixes should work.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
