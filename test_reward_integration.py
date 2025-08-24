#!/usr/bin/env python3
"""
Test script to verify that the reward integration works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_general_reward_import():
    """Test that we can import general_reward_fn from rewards package."""
    try:
        from rewards.general_reward import general_reward_fn
        print("✅ Successfully imported general_reward_fn from rewards package")
        return True
    except ImportError as e:
        print(f"❌ Failed to import general_reward_fn: {e}")
        return False

def test_reward_function():
    """Test that the reward function works with sample data."""
    try:
        from rewards.general_reward import general_reward_fn
        
        # Test code evaluation
        code_result = general_reward_fn(
            data_source="code",
            solution_str="def add(a, b):\n    return a + b",
            ground_truth={"answer": "def add(a, b):\n    return a + b"},
            is_eval=True
        )
        print(f"✅ Code evaluation result: {code_result}")
        
        # Test math evaluation  
        math_result = general_reward_fn(
            data_source="math",
            solution_str="The answer is \\boxed{42}",
            ground_truth={"answer": "42"},
            is_eval=True
        )
        print(f"✅ Math evaluation result: {math_result}")
        
        return True
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        return False

def test_verl_reward_loading():
    """Test that VERL can load the reward manager with general_reward."""
    try:
        # Mock config for testing
        class MockConfig:
            def __init__(self):
                self.reward_model = {
                    "reward_manager": "naive",
                    "use_general_reward": True,
                    "reward_kwargs": {}
                }
                self.data = {
                    "reward_fn_key": "data_source"
                }
                self.algorithm = None
        
        # Mock tokenizer
        class MockTokenizer:
            pass
        
        from verl.trainer.ppo.reward import load_reward_manager
        
        config = MockConfig()
        tokenizer = MockTokenizer()
        
        # Test loading with use_general_reward=True
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, for_validation=False)
        print("✅ Successfully loaded reward manager with general_reward")
        
        # Test loading for validation
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, for_validation=True)
        print("✅ Successfully loaded validation reward manager")
        
        return True
    except Exception as e:
        print(f"❌ VERL reward loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing reward integration...")
    
    tests = [
        ("Import Test", test_general_reward_import),
        ("Reward Function Test", test_reward_function),
        ("VERL Integration Test", test_verl_reward_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 All tests passed! The reward integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
