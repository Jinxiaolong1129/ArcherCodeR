#!/usr/bin/env python3
"""
Test script to reproduce the VLLM model loading error
Simulates the exact process that's failing in the training script
"""

import os
import sys

def test_vllm_model_loading():
    """Test VLLM model loading similar to the training script"""
    print("=== Testing VLLM Model Loading ===")
    
    # Set the same environment as the training script
    print(f"HF_TOKEN set: {os.environ['HF_TOKEN'][:8]}...{os.environ['HF_TOKEN'][-4:]}")
    
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    try:
        # This is the exact import that's failing in the training
        from vllm import LLM
        print("✓ VLLM imported successfully")
        
        # Test the model configuration that's causing issues
        print(f"Attempting to create LLM with model: {model_path}")
        
        # Use similar parameters to the training script
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.75,
            tensor_parallel_size=1,
            enable_chunked_prefill=True,
            max_num_batched_tokens=34816,
            max_model_len=34816,
            dtype='bfloat16',
            enforce_eager=True,
            trust_remote_code=False
        )
        
        print("✅ SUCCESS: VLLM model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ VLLM model loading failed: {e}")
        
        # Check specific error types
        if "401" in str(e) or "Unauthorized" in str(e):
            print("  → This is the same 401 error from the training script")
        
        # Try to get more detailed error info
        import traceback
        print("\n📋 Detailed Error Traceback:")
        traceback.print_exc()
        
        return False

def test_transformers_utils_config():
    """Test the specific module that's failing"""
    print("\n=== Testing transformers_utils.config ===")
    
    try:
        from vllm.transformers_utils.config import list_repo_files
        
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        print(f"Testing list_repo_files for: {model_path}")
        
        files = list_repo_files(model_path, revision=None)
        print(f"✓ Successfully listed {len(files)} files")
        return True
        
    except Exception as e:
        print(f"✗ list_repo_files failed: {e}")
        
        if "401" in str(e):
            print("  → This is likely the source of the 401 error")
        
        return False

def test_file_exists():
    """Test the file_exists function that's in the error stack"""
    print("\n=== Testing file_exists function ===")
    
    try:
        from vllm.transformers_utils.config import file_exists
        
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        filename = "1_Pooling/config.json"  # This is what it's looking for
        
        print(f"Testing file_exists for: {model_path}/{filename}")
        
        exists = file_exists(model_path, filename, revision=None)
        print(f"✓ file_exists returned: {exists}")
        return True
        
    except Exception as e:
        print(f"✗ file_exists failed: {e}")
        
        if "401" in str(e):
            print("  → Found the 401 error source!")
            print("  → VLLM is trying to check for pooling config which triggers API call")
        
        return False

def test_pooling_config():
    """Test get_pooling_config which is in the error trace"""
    print("\n=== Testing get_pooling_config ===")
    
    try:
        from vllm.transformers_utils.config import get_pooling_config
        
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        
        print(f"Testing get_pooling_config for: {model_path}")
        
        pooling_config = get_pooling_config(model_path, revision=None)
        print(f"✓ get_pooling_config returned: {pooling_config}")
        return True
        
    except Exception as e:
        print(f"✗ get_pooling_config failed: {e}")
        
        if "401" in str(e):
            print("  → This is the exact function causing the 401 error!")
            print("  → VLLM checks for pooling config even for causal LM models")
        
        return False

def main():
    """Run all tests"""
    print("🔍 VLLM Model Loading Diagnostic")
    print("=" * 50)
    
    # Test in order of specificity
    test_pooling_config()
    test_file_exists() 
    test_transformers_utils_config()
    test_vllm_model_loading()
    
    print("\n" + "=" * 50)
    print("💡 If pooling config test fails with 401:")
    print("   This means VLLM incorrectly tries to check for pooling")
    print("   config even for causal language models, and your token")
    print("   might have rate limiting or different permissions.")
    print("\n   Solutions:")
    print("   1. Wait a few minutes for rate limiting to reset")
    print("   2. Try with trust_remote_code=True")
    print("   3. Download model locally first")

if __name__ == "__main__":
    main() 