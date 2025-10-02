#!/usr/bin/env python3
"""
Alternating Algorithm Trainer

A trainer that supports dynamic switching between different RL algorithms
(e.g., Intuitor, GRPO, GAE) within the same training process without restart.
"""

import time
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, Sampler

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.single_controller.ray import RayWorkerGroup
from verl import DataProto


@dataclass
class AlternatingConfig:
    """Configuration for alternating algorithm training"""
    algorithms: List[str]  # List of algorithms to alternate between
    steps_per_phase: int = 50  # Number of steps per algorithm phase
    start_algorithm: str = "intuitor"  # Algorithm to start with
    total_phases: Optional[int] = None  # Total number of phases (optional)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.algorithms:
            raise ValueError("algorithms list cannot be empty")
        
        if self.start_algorithm not in self.algorithms:
            raise ValueError(f"start_algorithm '{self.start_algorithm}' must be in algorithms list")
        
        if self.steps_per_phase <= 0:
            raise ValueError("steps_per_phase must be positive")
        
        # Validate algorithm names
        valid_algorithms = [e.value for e in AdvantageEstimator]
        for algo in self.algorithms:
            if algo not in valid_algorithms:
                raise ValueError(f"Unknown algorithm '{algo}'. Valid algorithms: {valid_algorithms}")


class AlternatingRayPPOTrainer(RayPPOTrainer):
    """
    PPO trainer with dynamic algorithm switching capability.
    
    This trainer extends RayPPOTrainer to support alternating between different
    advantage estimation algorithms (Intuitor, GRPO, GAE, etc.) during training
    without requiring process restart.
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
        alternating_config: Optional[AlternatingConfig] = None,
    ):
        # Initialize alternating configuration
        if alternating_config is None:
            alternating_config = AlternatingConfig(
                algorithms=["intuitor", "grpo"],
                steps_per_phase=50,
                start_algorithm="intuitor"
            )
        
        self.alternating_config = alternating_config
        
        # Initialize alternating state
        self.current_algorithm = alternating_config.start_algorithm
        self.current_phase = 0
        self.steps_in_current_phase = 0
        self.phase_history = []  # Track phase history
        
        # Set initial algorithm in config
        config.algorithm.adv_estimator = AdvantageEstimator(self.current_algorithm)
        
        # Initialize parent trainer
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
        )
        
        print(f"🔄 AlternatingRayPPOTrainer initialized")
        print(f"   📋 Algorithms: {alternating_config.algorithms}")
        print(f"   🎯 Starting with: {self.current_algorithm}")
        print(f"   📊 Steps per phase: {alternating_config.steps_per_phase}")
    
    def _switch_algorithm(self, new_algorithm: str):
        """Switch to a new algorithm"""
        old_algorithm = self.current_algorithm
        
        print(f"🔄 Switching algorithm: {old_algorithm} → {new_algorithm}")
        
        # Update algorithm in config
        self.config.algorithm.adv_estimator = AdvantageEstimator(new_algorithm)
        
        # Update critic usage based on algorithm
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
            print(f"   ✅ Critic enabled for {new_algorithm}")
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.INTUITOR,
        ]:
            self.use_critic = False
            print(f"   ✅ Critic disabled for {new_algorithm}")
        else:
            raise NotImplementedError(f"Algorithm {new_algorithm} not supported")
        
        # Update current algorithm
        self.current_algorithm = new_algorithm
        
        print(f"   🎯 Algorithm switched to: {new_algorithm}")
    
    def _should_switch_algorithm(self) -> bool:
        """Check if we should switch to the next algorithm"""
        return self.steps_in_current_phase >= self.alternating_config.steps_per_phase
    
    def _get_next_algorithm(self) -> str:
        """Get the next algorithm in the sequence"""
        current_idx = self.alternating_config.algorithms.index(self.current_algorithm)
        next_idx = (current_idx + 1) % len(self.alternating_config.algorithms)
        return self.alternating_config.algorithms[next_idx]
    
    def _start_new_phase(self):
        """Start a new algorithm phase"""
        # Record current phase info
        phase_info = {
            'phase': self.current_phase,
            'algorithm': self.current_algorithm,
            'steps': self.steps_in_current_phase,
            'global_step': self.global_steps,
            'timestamp': time.time()
        }
        self.phase_history.append(phase_info)
        
        print(f"📊 Phase {self.current_phase} completed:")
        print(f"   🧠 Algorithm: {self.current_algorithm}")
        print(f"   📈 Steps: {self.steps_in_current_phase}")
        print(f"   🌐 Global step: {self.global_steps}")
        
        # Switch to next algorithm
        next_algorithm = self._get_next_algorithm()
        self._switch_algorithm(next_algorithm)
        
        # Reset phase counters
        self.current_phase += 1
        self.steps_in_current_phase = 0
        
        print(f"🚀 Starting Phase {self.current_phase} with {self.current_algorithm}")
    
    def _compute_advantage_with_current_algorithm(self, batch: DataProto) -> DataProto:
        """Compute advantages using the current algorithm with automatic switching"""
        # Check if we should switch algorithms before computing advantages
        if self._should_switch_algorithm():
            self._start_new_phase()
        
        # Increment step counter
        self.steps_in_current_phase += 1
        self.training_step_count += 1
        
        print(f"🧮 Computing advantages with {self.current_algorithm} (Step {self.steps_in_current_phase}/{self.alternating_config.steps_per_phase} in phase {self.current_phase})")
        
        from verl.trainer.ppo.ray_trainer import compute_advantage
        
        # Use the current algorithm for advantage computation
        batch = compute_advantage(
            data=batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm
        )
        
        return batch
    
    def fit(self):
        """Main training loop with algorithm alternating"""
        print(f"🚀 Starting alternating training with {len(self.alternating_config.algorithms)} algorithms")
        print(f"   📋 Algorithms: {self.alternating_config.algorithms}")
        print(f"   🎯 Starting with: {self.current_algorithm}")
        print(f"   📊 Steps per phase: {self.alternating_config.steps_per_phase}")
        
        # Store original compute_advantage function
        from verl.trainer.ppo import ray_trainer
        original_compute_advantage = ray_trainer.compute_advantage
        
        # Create our alternating wrapper
        def alternating_compute_advantage(*args, **kwargs):
            # Check if we should switch algorithms before computing advantages
            if self._should_switch_algorithm():
                self._start_new_phase()
            
            # Increment step counter
            self.steps_in_current_phase += 1
            
            print(f"🧮 Computing advantages with {self.current_algorithm} (Step {self.steps_in_current_phase}/{self.alternating_config.steps_per_phase} in phase {self.current_phase})")
            
            # Call original function with current algorithm settings
            return original_compute_advantage(*args, **kwargs)
        
        # Monkey patch the function
        ray_trainer.compute_advantage = alternating_compute_advantage
        
        try:
            # Call parent's fit method
            super().fit()
        finally:
            # Restore original function
            ray_trainer.compute_advantage = original_compute_advantage
        
        # Print final summary
        summary = self.get_phase_summary()
        print(f"\n📊 Alternating Training Summary:")
        print(f"   🏁 Total steps: {summary['total_steps']}")
        print(f"   📈 Total phases: {summary['total_phases']}")
        print(f"   🎯 Final algorithm: {summary['current_algorithm']}")
        print("✅ Alternating training completed!")
    
    def _log_alternating_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Add alternating-specific metrics to the log"""
        # Add alternating info to metrics
        metrics.update({
            'alternating/current_algorithm': self.current_algorithm,
            'alternating/current_phase': self.current_phase,
            'alternating/steps_in_phase': self.steps_in_current_phase,
            'alternating/total_phases': len(self.phase_history),
        })
        
        return metrics
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get a summary of all training phases"""
        total_steps = sum(phase['steps'] for phase in self.phase_history)
        if self.steps_in_current_phase > 0:
            total_steps += self.steps_in_current_phase
        
        return {
            'total_steps': total_steps,
            'total_phases': len(self.phase_history) + (1 if self.steps_in_current_phase > 0 else 0),
            'current_algorithm': self.current_algorithm,
            'current_phase': self.current_phase,
            'steps_in_current_phase': self.steps_in_current_phase,
            'phase_history': self.phase_history,
            'algorithms_used': list(set(phase['algorithm'] for phase in self.phase_history)),
        }
    
    # Helper methods that need to be implemented based on parent class
    def _init_training(self):
        """Initialize training (placeholder - implement based on parent)"""
        # This is handled by the parent class's fit() method
        pass
    
    def _ppo_update(self, batch):
        """Run PPO update (placeholder - implement based on parent)"""
        # This would be handled by the parent class's training loop
        return {}
    
    def _should_validate(self) -> bool:
        """Check if validation should be run"""
        if self.val_reward_fn is None:
            return False
        if self.config.trainer.test_freq <= 0:
            return False
        return self.global_steps % self.config.trainer.test_freq == 0
    
    def _run_validation(self):
        """Run validation"""
        return self._validate()
    
    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved"""
        if self.config.trainer.save_freq <= 0:
            return False
        return self.global_steps % self.config.trainer.save_freq == 0
    
    def _save_checkpoint(self):
        """Save checkpoint with alternating info"""
        # Call parent's save method
        super()._save_checkpoint()
        
        # Save alternating-specific state
        self._save_alternating_state()
    
    def _save_alternating_state(self):
        """Save alternating training state"""
        import json
        from omegaconf import ListConfig
        
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, 
            f"global_step_{self.global_steps}"
        )
        
        # Convert algorithms to regular Python list if it's a ListConfig
        algorithms = self.alternating_config.algorithms
        if isinstance(algorithms, ListConfig):
            algorithms = list(algorithms)
        
        # Save alternating state
        alternating_state = {
            'current_algorithm': self.current_algorithm,
            'current_phase': self.current_phase,
            'steps_in_current_phase': self.steps_in_current_phase,
            'phase_history': self.phase_history,
            'alternating_config': {
                'algorithms': algorithms,
                'steps_per_phase': self.alternating_config.steps_per_phase,
                'start_algorithm': self.alternating_config.start_algorithm,
            }
        }
        
        alternating_state_path = os.path.join(local_global_step_folder, "alternating_state.json")
        with open(alternating_state_path, 'w') as f:
            json.dump(alternating_state, f, indent=2)
        
        print(f"💾 Alternating state saved to: {alternating_state_path}")