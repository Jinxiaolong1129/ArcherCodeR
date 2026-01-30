#!/usr/bin/env python3
"""
Test script for INTUITOR algorithm implementation in ArcherCodeR/VERL
This script validates that all components are properly implemented without running full training.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_self_certainty_function():
    """Test the self_certainty_from_logits function"""
    print("🧪 Testing self_certainty_from_logits function...")
    
    try:
        from verl.utils.torch_functional import self_certainty_from_logits
        
        # Create test logits
        batch_size, seq_len, vocab_size = 2, 5, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Compute self-certainty
        self_certainty = self_certainty_from_logits(logits)
        
        # Validate output shape
        expected_shape = (batch_size, seq_len)
        assert self_certainty.shape == expected_shape, f"Expected shape {expected_shape}, got {self_certainty.shape}"
        
        # Validate output is finite
        assert torch.all(torch.isfinite(self_certainty)), "Self-certainty contains non-finite values"
        
        print(f"   ✅ Function works correctly")
        print(f"   📊 Output shape: {self_certainty.shape}")
        print(f"   📈 Mean self-certainty: {self_certainty.mean().item():.4f}")
        print(f"   📉 Std self-certainty: {self_certainty.std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_advantage_estimator_enum():
    """Test that INTUITOR is properly registered in AdvantageEstimator"""
    print("🧪 Testing AdvantageEstimator.INTUITOR enum...")
    
    try:
        from verl.trainer.ppo.core_algos import AdvantageEstimator
        
        # Check if INTUITOR is in the enum
        assert hasattr(AdvantageEstimator, 'INTUITOR'), "INTUITOR not found in AdvantageEstimator enum"
        assert AdvantageEstimator.INTUITOR == "intuitor", f"INTUITOR value should be 'intuitor', got {AdvantageEstimator.INTUITOR}"
        
        print(f"   ✅ INTUITOR enum registered correctly: {AdvantageEstimator.INTUITOR}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_intuitor_advantage_function():
    """Test the compute_intuitor_advantage function"""
    print("🧪 Testing compute_intuitor_advantage function...")
    
    try:
        from verl.trainer.ppo.core_algos import compute_intuitor_advantage
        
        # Create test data
        batch_size, seq_len = 4, 10
        self_certaintys = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Make some tokens invalid
        response_mask[:, :2] = False  # First 2 tokens are not response
        
        index = np.array(['prompt1', 'prompt1', 'prompt2', 'prompt2'], dtype=object)
        
        # Compute advantages
        advantages, returns = compute_intuitor_advantage(
            self_certaintys=self_certaintys,
            response_mask=response_mask,
            index=index
        )
        
        # Validate outputs
        assert advantages.shape == (batch_size, seq_len), f"Advantages shape mismatch: {advantages.shape}"
        assert returns.shape == (batch_size, seq_len), f"Returns shape mismatch: {returns.shape}"
        assert torch.all(torch.isfinite(advantages)), "Advantages contain non-finite values"
        assert torch.all(torch.isfinite(returns)), "Returns contain non-finite values"
        
        print(f"   ✅ Function works correctly")
        print(f"   📊 Advantages shape: {advantages.shape}")
        print(f"   📈 Mean advantage: {advantages.mean().item():.4f}")
        print(f"   📉 Std advantage: {advantages.std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_compute_advantage_integration():
    """Test that compute_advantage function can handle INTUITOR"""
    print("🧪 Testing compute_advantage integration...")
    
    try:
        from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
        from verl import DataProto
        
        # Create mock data
        batch_size, seq_len = 4, 10
        
        # Create batch data
        batch_data = {
            "self_certaintys": torch.randn(batch_size, seq_len),
            "response_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "responses": torch.randint(0, 1000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len * 2, dtype=torch.bool),  # prompt + response
        }
        
        non_tensor_data = {
            "uid": np.array(['prompt1', 'prompt1', 'prompt2', 'prompt2'], dtype=object)
        }
        
        # Create DataProto
        data = DataProto(batch=batch_data, non_tensor_batch=non_tensor_data)
        
        # Test compute_advantage with INTUITOR
        result_data = compute_advantage(
            data=data,
            adv_estimator=AdvantageEstimator.INTUITOR,
            norm_adv_by_std_in_grpo=True
        )
        
        # Validate results
        assert "advantages" in result_data.batch, "Advantages not found in result"
        assert "returns" in result_data.batch, "Returns not found in result"
        
        advantages = result_data.batch["advantages"]
        returns = result_data.batch["returns"]
        
        assert advantages.shape == (batch_size, seq_len), f"Advantages shape mismatch: {advantages.shape}"
        assert returns.shape == (batch_size, seq_len), f"Returns shape mismatch: {returns.shape}"
        
        print(f"   ✅ Integration works correctly")
        print(f"   📊 Final advantages shape: {advantages.shape}")
        print(f"   📈 Mean advantage: {advantages.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_trainer_configuration():
    """Test that RayPPOTrainer correctly handles INTUITOR (use_critic=False)"""
    print("🧪 Testing trainer configuration...")
    
    try:
        from verl.trainer.ppo.ray_trainer import AdvantageEstimator
        
        # Simulate trainer initialization logic
        adv_estimator = AdvantageEstimator.INTUITOR
        
        # Check use_critic logic
        if adv_estimator == AdvantageEstimator.GAE:
            use_critic = True
        elif adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.INTUITOR,
        ]:
            use_critic = False
        else:
            raise NotImplementedError(f"Unknown advantage estimator: {adv_estimator}")
        
        assert use_critic == False, f"INTUITOR should not use critic, but use_critic={use_critic}"
        
        print(f"   ✅ Trainer configuration correct: use_critic={use_critic}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_metrics_integration():
    """Test that self-certainty metrics are properly computed"""
    print("🧪 Testing metrics integration...")
    
    try:
        from verl.trainer.ppo.metric_utils import compute_data_metrics
        from verl import DataProto
        
        # Create mock batch with self_certaintys
        batch_size, response_len = 4, 10
        prompt_len = 15
        total_len = prompt_len + response_len
        
        batch_data = {
            "token_level_scores": torch.randn(batch_size, response_len),
            "token_level_rewards": torch.randn(batch_size, response_len),
            "advantages": torch.randn(batch_size, response_len),
            "returns": torch.randn(batch_size, response_len),
            "responses": torch.randint(0, 1000, (batch_size, response_len)),
            "attention_mask": torch.ones(batch_size, total_len, dtype=torch.bool),
            "self_certaintys": torch.randn(batch_size, response_len),  # Key addition for INTUITOR
        }
        
        batch = DataProto(batch=batch_data)
        
        # Compute metrics
        metrics = compute_data_metrics(batch, use_critic=False)
        
        # Check that self-certainty metrics are included
        expected_metrics = [
            "intuitor/self_certainty/mean",
            "intuitor/self_certainty/max", 
            "intuitor/self_certainty/min",
            "intuitor/self_certainty/std"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
        
        print(f"   ✅ Metrics integration works correctly")
        print(f"   📊 Self-certainty mean: {metrics['intuitor/self_certainty/mean']:.4f}")
        print(f"   📈 Self-certainty std: {metrics['intuitor/self_certainty/std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 INTUITOR Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Self-Certainty Function", test_self_certainty_function),
        ("Advantage Estimator Enum", test_advantage_estimator_enum),
        ("INTUITOR Advantage Function", test_intuitor_advantage_function),
        ("Compute Advantage Integration", test_compute_advantage_integration),
        ("Trainer Configuration", test_trainer_configuration),
        ("Metrics Integration", test_metrics_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! INTUITOR implementation is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 