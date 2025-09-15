#!/usr/bin/env python3
"""
Enhanced Alternating Algorithm Trainer

An advanced version that supports algorithm-specific parameter configurations
for optimal performance of each algorithm (Intuitor, GRPO, etc.).
"""

import time
import copy
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, Sampler

from verl.trainer.ppo.alternating_trainer import AlternatingRayPPOTrainer, AlternatingConfig
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, WorkerType
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.data_structure import DataProto


@dataclass
class EnhancedAlternatingConfig(AlternatingConfig):
    """Enhanced configuration with algorithm-specific parameters"""
    algorithm_configs: Optional[Dict[str, Dict[str, Any]]] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Initialize algorithm_configs if not provided
        if self.algorithm_configs is None:
            self.algorithm_configs = {}
        
        # Ensure all algorithms have configs
        for algo in self.algorithms:
            if algo not in self.algorithm_configs:
                self.algorithm_configs[algo] = {}


class EnhancedAlternatingRayPPOTrainer(AlternatingRayPPOTrainer):
    """
    Enhanced trainer with algorithm-specific parameter support.
    
    This trainer can dynamically adjust hyperparameters when switching between
    algorithms to optimize performance for each specific algorithm.
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        alternating_config: Optional[EnhancedAlternatingConfig] = None,
    ):
        # Store original config for restoration
        self.original_config = copy.deepcopy(config)
        
        # Initialize with enhanced config
        if alternating_config is None:
            alternating_config = EnhancedAlternatingConfig(
                algorithms=["intuitor", "grpo"],
                steps_per_phase=50,
                start_algorithm="intuitor",
                algorithm_configs=self._get_default_algorithm_configs()
            )
        
        # Initialize parent class
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
            alternating_config=alternating_config,
        )
        
        print(f"🎯 EnhancedAlternatingRayPPOTrainer initialized")
        print(f"   📋 Algorithm-specific configs available: {list(self.alternating_config.algorithm_configs.keys())}")
    
    def _get_default_algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default algorithm-specific configurations based on official scripts"""
        return {
            "intuitor": {
                # 基于官方 intuitor_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh
                "actor": {
                    "optim": {
                        "lr": 3e-6,                    # 官方Intuitor学习率
                        "warmup_style": "cosine",      # 官方Intuitor预热方式
                        "lr_warmup_steps_ratio": 0.1,  # 官方Intuitor预热比例
                        "weight_decay": 0.1,           # 官方权重衰减
                    },
                    "ppo_epochs": 1,                   # 官方Intuitor PPO epochs
                    "entropy_coeff": 0,                # 官方Intuitor熵系数
                    "use_kl_loss": False,              # 官方设置
                    "grad_clip": 1.0,                  # 官方梯度裁剪
                    "fsdp_config": {
                        "param_offload": False,        # offload=False
                        "optimizer_offload": False,    # offload=False
                    },
                    "use_dynamic_bsz": False,          # 官方设置
                },
                "rollout": {
                    "gpu_memory_utilization": 0.8,    # 官方Intuitor设置
                    "enable_chunked_prefill": False,   # 官方Intuitor设置
                    "temperature": 1.0,                # 官方温度
                    "top_p": 1.0,                      # 官方top_p
                    "top_k": -1,                       # 官方top_k
                    "n": 16,                           # n_resp_per_prompt
                    "val_kwargs": {
                        "do_sample": True,             # 官方Intuitor验证设置
                        "temperature": 0.8,            # v_temperature
                    }
                },
                "ref": {
                    "fsdp_config": {
                        "param_offload": True,         # 官方Intuitor ref设置
                    }
                },
                "algorithm": {
                    "use_kl_in_reward": False,         # 官方设置
                }
            },
            "grpo": {
                # 基于官方 run_Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl.sh
                "actor": {
                    "optim": {
                        "lr": 1e-6,                    # 官方GRPO学习率
                        "lr_warmup_steps": 10,         # 官方GRPO预热步数
                        "weight_decay": 0.1,           # 官方权重衰减
                    },
                    "ppo_epochs": 3,                   # 官方GRPO PPO epochs
                    "entropy_coeff": 0,                # 官方GRPO熵系数
                    "use_kl_loss": False,              # use_kl_loss=False
                    "kl_loss_coef": 0.0,               # kl_loss_coef=0.0
                    "kl_loss_type": "low_var_kl",      # 官方设置
                    "grad_clip": 1.0,                  # 官方梯度裁剪
                    "clip_ratio_low": 0.2,             # 官方clip参数
                    "clip_ratio_high": 0.2,            # 官方clip参数
                    "clip_ratio_c": 10.0,              # 官方设置
                    "loss_agg_mode": "token-mean",     # 官方设置
                    # GRPO特有的token熵分离参数
                    "use_token_entropy_separate": True,     # 官方GRPO设置
                    "high_entropy_kl_loss_scale_coef": 0.0, # 官方设置
                    "low_entropy_clip_ratio_low": 0.2,      # 官方设置
                    "low_entropy_clip_ratio_high": 0.2,     # 官方设置
                    "high_entropy_clip_ratio_low": 0.5,     # 官方设置
                    "high_entropy_clip_ratio_high": 0.5,    # 官方设置
                    "fsdp_config": {
                        "param_offload": False,        # offload=False
                        "optimizer_offload": False,    # offload=False
                    },
                    "use_dynamic_bsz": False,          # 官方设置
                },
                "rollout": {
                    "gpu_memory_utilization": 0.75,   # 官方GRPO设置
                    "enable_chunked_prefill": True,    # 官方GRPO设置
                    "temperature": 1.0,                # 官方温度
                    "top_p": 1.0,                      # 官方top_p
                    "top_k": -1,                       # 官方top_k
                    "n": 16,                           # n_resp_per_prompt
                    "val_kwargs": {
                        "do_sample": True,             # 官方验证设置
                        "temperature": 0.8,            # v_temperature
                    }
                },
                "ref": {
                    "fsdp_config": {
                        "param_offload": False,        # 官方GRPO ref设置
                    }
                },
                "algorithm": {
                    "use_kl_in_reward": False,         # use_kl_in_reward=False
                    "kl_ctrl": {
                        "kl_coef": 0.0,                # kl_coef=0.0
                    },
                    "norm_adv_by_std_in_grpo": True,   # GRPO特有
                }
            }
        }
    
    def _apply_algorithm_config(self, algorithm: str):
        """Apply algorithm-specific configuration"""
        if algorithm not in self.alternating_config.algorithm_configs:
            print(f"⚠️  No specific config for {algorithm}, using defaults")
            return
        
        algo_config = self.alternating_config.algorithm_configs[algorithm]
        print(f"🔧 Applying {algorithm}-specific configuration...")
        
        # Apply actor configuration
        if "actor" in algo_config:
            self._apply_nested_config(self.config.actor_rollout_ref.actor, algo_config["actor"])
            print(f"   ✅ Applied actor config for {algorithm}")
        
        # Apply rollout configuration
        if "rollout" in algo_config:
            self._apply_nested_config(self.config.actor_rollout_ref.rollout, algo_config["rollout"])
            print(f"   ✅ Applied rollout config for {algorithm}")
        
        # Apply algorithm configuration
        if "algorithm" in algo_config:
            self._apply_nested_config(self.config.algorithm, algo_config["algorithm"])
            print(f"   ✅ Applied algorithm config for {algorithm}")
        
        # Log key parameter changes
        self._log_parameter_changes(algorithm, algo_config)
    
    def _apply_nested_config(self, target_config, source_config):
        """Recursively apply configuration changes"""
        for key, value in source_config.items():
            if hasattr(target_config, key):
                if isinstance(value, dict) and hasattr(getattr(target_config, key), '__dict__'):
                    # Recursively apply nested configs
                    self._apply_nested_config(getattr(target_config, key), value)
                else:
                    # Set the value
                    setattr(target_config, key, value)
            else:
                print(f"⚠️  Warning: Config key '{key}' not found in target config")
    
    def _log_parameter_changes(self, algorithm: str, algo_config: Dict[str, Any]):
        """Log important parameter changes for the algorithm"""
        print(f"📊 Key parameters for {algorithm}:")
        
        # Log learning rate
        if "actor" in algo_config and "optim" in algo_config["actor"] and "lr" in algo_config["actor"]["optim"]:
            lr = algo_config["actor"]["optim"]["lr"]
            print(f"   📈 Learning rate: {lr}")
        
        # Log PPO epochs
        if "actor" in algo_config and "ppo_epochs" in algo_config["actor"]:
            epochs = algo_config["actor"]["ppo_epochs"]
            print(f"   🔄 PPO epochs: {epochs}")
        
        # Log entropy coefficient
        if "actor" in algo_config and "entropy_coeff" in algo_config["actor"]:
            entropy = algo_config["actor"]["entropy_coeff"]
            print(f"   🎲 Entropy coefficient: {entropy}")
        
        # Log temperature
        if "rollout" in algo_config and "temperature" in algo_config["rollout"]:
            temp = algo_config["rollout"]["temperature"]
            print(f"   🌡️  Temperature: {temp}")
        
        # Algorithm-specific parameters
        if algorithm == "grpo":
            if "algorithm" in algo_config and "norm_adv_by_std_in_grpo" in algo_config["algorithm"]:
                norm_adv = algo_config["algorithm"]["norm_adv_by_std_in_grpo"]
                print(f"   📏 Normalize advantages: {norm_adv}")
        
        elif algorithm == "intuitor":
            print(f"   🧠 Uses self-certainty as reward signal")
    
    def _switch_algorithm(self, new_algorithm: str):
        """Enhanced algorithm switching with parameter adjustment"""
        old_algorithm = getattr(self, 'current_algorithm', None)
        
        print(f"🔄 Enhanced algorithm switch: {old_algorithm} → {new_algorithm}")
        
        # Apply algorithm-specific configuration BEFORE switching
        self._apply_algorithm_config(new_algorithm)
        
        # Call parent's switch method
        super()._switch_algorithm(new_algorithm)
        
        # Additional algorithm-specific setup
        self._setup_algorithm_specific_components(new_algorithm)
    
    def _setup_algorithm_specific_components(self, algorithm: str):
        """Setup algorithm-specific components"""
        if algorithm == "intuitor":
            print("   🧠 Intuitor setup: Self-certainty reward enabled")
            # Intuitor-specific setup could go here
            
        elif algorithm == "grpo":
            print("   🎯 GRPO setup: Group preference optimization enabled")
            # GRPO-specific setup could go here
        
        print(f"   ✅ {algorithm} components ready")
    
    def _compute_advantage_with_current_algorithm(self, batch: DataProto) -> DataProto:
        """Enhanced advantage computation with algorithm-specific handling"""
        from verl.trainer.ppo.ray_trainer import compute_advantage
        
        print(f"🧮 Computing advantages with {self.current_algorithm}")
        
        # Get algorithm-specific parameters
        algo_config = self.alternating_config.algorithm_configs.get(self.current_algorithm, {})
        algorithm_params = algo_config.get("algorithm", {})
        
        # Use algorithm-specific parameters for advantage computation
        norm_adv_by_std = algorithm_params.get("norm_adv_by_std_in_grpo", True)
        
        batch = compute_advantage(
            data=batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            norm_adv_by_std_in_grpo=norm_adv_by_std,
            config=self.config.algorithm
        )
        
        return batch
    
    def _log_alternating_metrics(self, metrics: Dict[str, Any]):
        """Enhanced metrics logging with algorithm-specific information"""
        # Call parent method
        metrics = super()._log_alternating_metrics(metrics)
        
        # Add enhanced metrics
        algo_config = self.alternating_config.algorithm_configs.get(self.current_algorithm, {})
        
        # Log current learning rate
        if "actor" in algo_config and "optim" in algo_config["actor"] and "lr" in algo_config["actor"]["optim"]:
            metrics[f'alternating/current_lr'] = algo_config["actor"]["optim"]["lr"]
        
        # Log current PPO epochs
        if "actor" in algo_config and "ppo_epochs" in algo_config["actor"]:
            metrics[f'alternating/current_ppo_epochs'] = algo_config["actor"]["ppo_epochs"]
        
        # Log algorithm-specific flags
        if self.current_algorithm == "grpo":
            metrics[f'alternating/using_grpo_normalization'] = algo_config.get("algorithm", {}).get("norm_adv_by_std_in_grpo", True)
        elif self.current_algorithm == "intuitor":
            metrics[f'alternating/using_self_certainty'] = True
        
        return metrics
    
    def get_algorithm_config_summary(self) -> Dict[str, Any]:
        """Get a summary of algorithm-specific configurations"""
        summary = {}
        
        for algo, config in self.alternating_config.algorithm_configs.items():
            algo_summary = {}
            
            # Extract key parameters
            if "actor" in config:
                actor_config = config["actor"]
                if "optim" in actor_config:
                    algo_summary["learning_rate"] = actor_config["optim"].get("lr", "default")
                    algo_summary["warmup_style"] = actor_config["optim"].get("warmup_style", "default")
                algo_summary["ppo_epochs"] = actor_config.get("ppo_epochs", "default")
                algo_summary["entropy_coeff"] = actor_config.get("entropy_coeff", "default")
            
            if "rollout" in config:
                rollout_config = config["rollout"]
                algo_summary["temperature"] = rollout_config.get("temperature", "default")
                algo_summary["n_responses"] = rollout_config.get("n", "default")
            
            if "algorithm" in config:
                algorithm_config = config["algorithm"]
                algo_summary.update(algorithm_config)
            
            summary[algo] = algo_summary
        
        return summary
    
    def get_enhanced_phase_summary(self) -> Dict[str, Any]:
        """Get enhanced phase summary with algorithm configurations"""
        base_summary = super().get_phase_summary()
        
        # Add algorithm configuration summary
        base_summary["algorithm_configs"] = self.get_algorithm_config_summary()
        
        # Add performance metrics per algorithm
        algo_performance = {}
        for phase_info in self.phase_history:
            algo = phase_info["algorithm"]
            if algo not in algo_performance:
                algo_performance[algo] = {
                    "total_steps": 0,
                    "total_phases": 0,
                    "avg_steps_per_phase": 0
                }
            
            algo_performance[algo]["total_steps"] += phase_info["steps"]
            algo_performance[algo]["total_phases"] += 1
            algo_performance[algo]["avg_steps_per_phase"] = (
                algo_performance[algo]["total_steps"] / algo_performance[algo]["total_phases"]
            )
        
        base_summary["algorithm_performance"] = algo_performance
        
        return base_summary
