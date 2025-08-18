#!/usr/bin/env python3
"""
HuggingFace Model Access Test Script
Tests access to deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B and diagnoses auth issues
"""

import os
import sys
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("=== Testing Basic Imports ===")
    try:
        import requests
        print("✓ requests imported successfully")
        
        from huggingface_hub import HfApi, login, whoami
        print("✓ huggingface_hub imported successfully")
        
        from transformers import AutoTokenizer, AutoConfig
        print("✓ transformers imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_environment():
    """Check environment variables and authentication"""
    print("\n=== Checking Environment ===")
    
    # Check HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"✓ HF_TOKEN found: {hf_token[:8]}...{hf_token[-4:]}")
    else:
        print("✗ HF_TOKEN not found in environment")
    
    # Check HF_HOME
    hf_home = os.getenv('HF_HOME', '~/.cache/huggingface')
    print(f"✓ HF_HOME: {hf_home}")
    
    return hf_token

def test_authentication():
    """Test HuggingFace authentication"""
    print("\n=== Testing Authentication ===")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✓ Successfully authenticated as: {user_info}")
        return True
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        
        # Try to get more details
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                user_info = whoami()
                print(f"✓ Manual login successful: {user_info}")
                return True
            except Exception as e2:
                print(f"✗ Manual login also failed: {e2}")
        
        return False

def test_model_repository_access(model_name):
    """Test if we can access the model repository"""
    print(f"\n=== Testing Repository Access: {model_name} ===")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get repository info
        repo_info = api.repo_info(model_name)
        print(f"✓ Repository accessible")
        print(f"  - Model ID: {repo_info.id}")
        print(f"  - Private: {repo_info.private}")
        print(f"  - Downloads: {repo_info.downloads}")
        print(f"  - Likes: {repo_info.likes}")
        
        # Try to list files
        files = api.list_repo_files(model_name)
        print(f"✓ Repository files accessible ({len(files)} files)")
        config_files = [f for f in files if 'config' in f.lower()]
        print(f"  - Config files: {config_files}")
        
        return True
        
    except Exception as e:
        print(f"✗ Repository access failed: {e}")
        
        # Check if it's a 401 error specifically
        if "401" in str(e) or "Unauthorized" in str(e):
            print("  → This appears to be an authentication/permission issue")
        elif "404" in str(e) or "Not Found" in str(e):
            print("  → Model repository not found or doesn't exist")
        
        return False

def test_model_config_access(model_name):
    """Test accessing model configuration"""
    print(f"\n=== Testing Model Config Access: {model_name} ===")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        print("✓ Model config loaded successfully")
        print(f"  - Architecture: {config.architectures}")
        print(f"  - Model type: {config.model_type}")
        print(f"  - Hidden size: {config.hidden_size}")
        return True
    except Exception as e:
        print(f"✗ Model config access failed: {e}")
        return False

def test_tokenizer_access(model_name):
    """Test accessing model tokenizer"""
    print(f"\n=== Testing Tokenizer Access: {model_name} ===")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Model max length: {tokenizer.model_max_length}")
        
        # Test encoding
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        print(f"  - Test encoding '{test_text}': {tokens}")
        
        return True
    except Exception as e:
        print(f"✗ Tokenizer access failed: {e}")
        return False

def test_alternative_models():
    """Test access to alternative models"""
    print("\n=== Testing Alternative Models ===")
    
    alternative_models = [
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "microsoft/DialoGPT-medium",
        "gpt2"
    ]
    
    accessible_models = []
    
    for model in alternative_models:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model)
            print(f"✓ {model} - accessible")
            accessible_models.append(model)
        except Exception as e:
            print(f"✗ {model} - {str(e)[:100]}...")
    
    return accessible_models

def test_direct_api_request(model_name):
    """Test direct API request to HuggingFace"""
    print(f"\n=== Testing Direct API Request: {model_name} ===")
    
    try:
        import requests
        
        url = f"https://huggingface.co/api/models/{model_name}"
        headers = {}
        
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            headers['Authorization'] = f'Bearer {hf_token}'
        
        response = requests.get(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ Direct API access successful")
            data = response.json()
            print(f"  - Model ID: {data.get('id', 'N/A')}")
            print(f"  - Private: {data.get('private', 'N/A')}")
            return True
        elif response.status_code == 401:
            print("✗ 401 Unauthorized - Invalid or insufficient token permissions")
        elif response.status_code == 404:
            print("✗ 404 Not Found - Model doesn't exist or no access")
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
        
        return False
        
    except Exception as e:
        print(f"✗ Direct API request failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 HuggingFace Model Access Diagnostic Tool")
    print("=" * 50)
    
    target_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Run all tests
    results = {}
    
    results['imports'] = test_basic_imports()
    if not results['imports']:
        print("\n❌ Basic imports failed. Please install required packages.")
        return
    
    hf_token = check_environment()
    results['auth'] = test_authentication()
    results['direct_api'] = test_direct_api_request(target_model)
    results['repo_access'] = test_model_repository_access(target_model)
    results['config_access'] = test_model_config_access(target_model)
    results['tokenizer_access'] = test_tokenizer_access(target_model)
    
    # Test alternatives if main model fails
    if not results['repo_access']:
        accessible_alternatives = test_alternative_models()
        results['alternatives'] = accessible_alternatives
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if results['auth'] and results['repo_access']:
        print("✅ SUCCESS: Model is accessible with current credentials")
    else:
        print("❌ ISSUES DETECTED:")
        
        if not results['auth']:
            print("   - Authentication failed")
            print("   - Solution: Check HF_TOKEN or run 'huggingface-cli login'")
        
        if not results['repo_access']:
            print(f"   - Cannot access {target_model}")
            print("   - Possible causes:")
            print("     * Model requires special permissions")
            print("     * Token lacks necessary permissions")
            print("     * Model is private/gated")
            print("     * Model name is incorrect")
        
        if results.get('alternatives'):
            print(f"\n🔄 ALTERNATIVE MODELS AVAILABLE:")
            for alt in results['alternatives']:
                print(f"   - {alt}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not hf_token:
        print("   1. Set HF_TOKEN environment variable")
        print("   2. Or run: huggingface-cli login")
    
    if not results['repo_access']:
        print("   3. Check model page: https://huggingface.co/" + target_model)
        print("   4. Request access if model is gated")
        print("   5. Verify token permissions in HF settings")
        print("   6. Consider using alternative models")

if __name__ == "__main__":
    main() 