#!/usr/bin/env python3
"""
Intuitor-DAPO Alternating Trainer

A trainer that alternates between:
1. Intuitor algorithm (self-certainty based rewards)
2. DAPO training methodology (rejection sampling + external rewards)
"""

import time
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
import numpy as np
import math
from collections import defaultdict
from torch.utils.data import Dataset, Sampler

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.single_controller.ray import RayWorkerGroup
from verl import DataProto


@dataclass
class IntuitorDAPOAlternatingConfig:
    """Configuration for Intuitor-DAPO alternating training"""
    modes: List[str] = None  # ["intuitor", "dapo"]
    steps_per_phase: int = 50
    start_mode: str = "intuitor"
    total_phases: Optional[int] = None
    
    # DAPO specific settings
    dapo_rejection_sample: bool = True
    dapo_enable_overlong_filter: bool = True
    
    def __post_init__(self):
        if self.modes is None:
            self.modes = ["intuitor", "dapo"]
        
        if not self.modes:
            raise ValueError("modes list cannot be empty")
        
        if self.start_mode not in self.modes:
            raise ValueError(f"start_mode '{self.start_mode}' must be in modes list")
        
        if self.steps_per_phase <= 0:
            raise ValueError("steps_per_phase must be positive")
        
        # Validate mode names
        valid_modes = ["intuitor", "dapo"]
        for mode in self.modes:
            if mode not in valid_modes:
                raise ValueError(f"Unknown mode '{mode}'. Valid modes: {valid_modes}")


class IntuitorDAPOAlternatingTrainer(RayPPOTrainer):
    """
    Trainer that alternates between Intuitor algorithm and DAPO training methodology.
    
    - Intuitor phases: Use self-certainty as reward, standard PPO training flow
    - DAPO phases: Use external rewards with rejection sampling and special data processing
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
        alternating_config: Optional[IntuitorDAPOAlternatingConfig] = None,
    ):
        # Initialize alternating configuration
        if alternating_config is None:
            alternating_config = IntuitorDAPOAlternatingConfig(
                modes=["intuitor", "dapo"],
                steps_per_phase=50,
                start_mode="intuitor"
            )
        
        self.alternating_config = alternating_config
        
        # Initialize alternating state
        self.current_mode = alternating_config.start_mode
        self.current_phase = 0
        self.steps_in_current_phase = 0
        self.phase_history = []
        
        # Set initial algorithm based on mode
        if self.current_mode == "intuitor":
            config.algorithm.adv_estimator = AdvantageEstimator.INTUITOR
        else:  # dapo mode
            # DAPO can use any algorithm, let's default to GRPO
            config.algorithm.adv_estimator = AdvantageEstimator.GRPO
        
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
        
        print(f"🔄 IntuitorDAPOAlternatingTrainer initialized")
        print(f"   📋 Modes: {alternating_config.modes}")
        print(f"   🎯 Starting with: {self.current_mode}")
        print(f"   📊 Steps per phase: {alternating_config.steps_per_phase}")
    
    def _should_switch_mode(self) -> bool:
        """Check if we should switch to the next mode"""
        return self.steps_in_current_phase >= self.alternating_config.steps_per_phase
    
    def _get_next_mode(self) -> str:
        """Get the next mode in the sequence"""
        current_idx = self.alternating_config.modes.index(self.current_mode)
        next_idx = (current_idx + 1) % len(self.alternating_config.modes)
        return self.alternating_config.modes[next_idx]
    
    def _switch_mode(self, new_mode: str):
        """Switch to a new training mode"""
        old_mode = self.current_mode
        
        print(f"🔄 Switching mode: {old_mode} → {new_mode}")
        
        # Update algorithm based on mode
        if new_mode == "intuitor":
            self.config.algorithm.adv_estimator = AdvantageEstimator.INTUITOR
            print(f"   🧠 Switched to Intuitor algorithm (self-certainty based)")
        else:  # dapo mode
            self.config.algorithm.adv_estimator = AdvantageEstimator.GRPO
            print(f"   🎯 Switched to DAPO mode (rejection sampling + external rewards)")
        
        # Update critic usage
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False
        
        self.current_mode = new_mode
        print(f"   ✅ Mode switched to: {new_mode}")
    
    def _start_new_phase(self):
        """Start a new training phase"""
        # Record current phase info
        phase_info = {
            'phase': self.current_phase,
            'mode': self.current_mode,
            'steps': self.steps_in_current_phase,
            'global_step': self.global_steps,
            'timestamp': time.time()
        }
        self.phase_history.append(phase_info)
        
        print(f"📊 Phase {self.current_phase} completed:")
        print(f"   🎯 Mode: {self.current_mode}")
        print(f"   📈 Steps: {self.steps_in_current_phase}")
        print(f"   🌐 Global step: {self.global_steps}")
        
        # Switch to next mode
        next_mode = self._get_next_mode()
        self._switch_mode(next_mode)
        
        # Reset phase counters
        self.current_phase += 1
        self.steps_in_current_phase = 0
        
        print(f"🚀 Starting Phase {self.current_phase} with {self.current_mode}")
    
    def _dapo_rejection_sampling(self, batch: DataProto, reward_tensor: torch.Tensor) -> DataProto:
        """Apply DAPO-style rejection sampling"""
        print(f"   🎯 Applying DAPO rejection sampling...")
        
        # Group rewards by uid
        uids = batch.non_tensor_batch['uid']
        unique_uids = np.unique(uids)
        valid_mask = torch.ones(len(uids), dtype=torch.bool)
        solve_none = 0
        solve_all = 0
        
        for uid in unique_uids:
            uid_mask = uids == uid
            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
            
            # Check if all rewards are 0 or all are 1 for this uid
            if (uid_rewards == 0).all():
                valid_mask[uid_mask] = False
                solve_none += 1
            elif (uid_rewards == 1).all():
                valid_mask[uid_mask] = False
                solve_all += 1
        
        valid_prompts = len(unique_uids) - solve_all - solve_none
        print(f"      📊 Rejection sampling results:")
        print(f"         🎯 Valid prompts: {valid_prompts}/{len(unique_uids)} ({valid_prompts/len(unique_uids)*100:.1f}%)")
        print(f"         ❌ Solve none: {solve_none} ({solve_none/len(unique_uids)*100:.1f}%)")
        print(f"         ✅ Solve all: {solve_all} ({solve_all/len(unique_uids)*100:.1f}%)")
        
        if self.alternating_config.dapo_rejection_sample:
            # If no valid samples remain, return None to skip this batch
            if not valid_mask.any():
                print(f"      ⚠️  No valid samples remaining, skipping batch...")
                return None
            # Filter batch to keep only valid samples
            batch = batch[valid_mask]
            print(f"      ✅ Filtered to {len(batch)} valid samples for training")
        
        return batch
    
    def _dapo_length_filtering(self, batch: DataProto) -> DataProto:
        """Apply DAPO-style length filtering"""
        max_response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -max_response_length:]
        response_length = response_mask.sum(-1).float()
        response_clip_mask = ~torch.ge(response_length, max_response_length)
        
        within_limit = response_clip_mask.sum().item()
        print(f"      📏 Response length check: {within_limit}/{len(batch)} samples within length limit")
        
        if self.alternating_config.dapo_enable_overlong_filter:
            batch = batch[response_clip_mask]
            print(f"      ✂️  Filtered overlong responses, remaining: {len(batch)} samples")
        
        return batch
    
    def _dapo_batch_adjustment(self, batch: DataProto) -> DataProto:
        """Apply DAPO-style batch size adjustment"""
        # Sort by index
        def get_sorted_indices(lst):
            return [index for index, _ in sorted(enumerate(lst), key=lambda x: x[1])]
        
        sorted_indices = torch.tensor(get_sorted_indices(batch.non_tensor_batch['index']))
        batch.reorder(sorted_indices)
        
        # Round down to the nearest multiple of world size
        num_trainer_replicas = self.actor_rollout_wg.world_size
        if batch.batch['input_ids'].shape[0] < num_trainer_replicas and num_trainer_replicas/batch.batch['input_ids'].shape[0] <= 2:
            batch = batch.repeat(repeat_times=math.ceil(num_trainer_replicas/batch.batch['input_ids'].shape[0]), interleave=False)
        
        max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
        if not max_batch_size:
            print(f"      ⚠️  Batch size too small for {num_trainer_replicas} replicas, skipping...")
            return None
        
        batch = batch[:max_batch_size]
        print(f"      ✅ Final training batch size: {len(batch)} samples")
        
        return batch
    
    def fit(self):
        """Main training loop with Intuitor-DAPO alternating"""
        print(f"🚀 Starting Intuitor-DAPO alternating training")
        
        # Store original compute_advantage function
        from verl.trainer.ppo import ray_trainer
        original_compute_advantage = ray_trainer.compute_advantage
        
        # Create our alternating wrapper
        def alternating_compute_advantage(*args, **kwargs):
            # Check if we should switch modes before computing advantages
            if self._should_switch_mode():
                self._start_new_phase()
            
            # Increment step counter
            self.steps_in_current_phase += 1
            
            print(f"🧮 Computing advantages with {self.current_mode} mode (Step {self.steps_in_current_phase}/{self.alternating_config.steps_per_phase} in phase {self.current_phase})")
            
            # Call original function with current algorithm settings
            return original_compute_advantage(*args, **kwargs)
        
        # Monkey patch the function
        ray_trainer.compute_advantage = alternating_compute_advantage
        
        # Store original fit method parts we need to override
        self._original_process_batch = self._process_training_batch
        self._process_training_batch = self._alternating_process_training_batch
        
        try:
            # Call parent's fit method
            super().fit()
        finally:
            # Restore original functions
            ray_trainer.compute_advantage = original_compute_advantage
            self._process_training_batch = self._original_process_batch
        
        # Print final summary
        summary = self.get_phase_summary()
        print(f"\n📊 Alternating Training Summary:")
        print(f"   🏁 Total steps: {summary['total_steps']}")
        print(f"   📈 Total phases: {summary['total_phases']}")
        print(f"   🎯 Final mode: {summary['current_mode']}")
        print("✅ Intuitor-DAPO alternating training completed!")
    
    def _alternating_process_training_batch(self, batch: DataProto, reward_tensor: torch.Tensor) -> Optional[DataProto]:
        """Process training batch based on current mode"""
        if self.current_mode == "intuitor":
            # Intuitor mode: standard processing, no rejection sampling
            print(f"   🧠 Processing batch in Intuitor mode (standard PPO flow)")
            return batch
        
        else:  # dapo mode
            print(f"   🎯 Processing batch in DAPO mode (rejection sampling + filtering)")
            
            # Apply DAPO rejection sampling
            batch = self._dapo_rejection_sampling(batch, reward_tensor)
            if batch is None:
                return None  # Skip this batch
            
            # Apply DAPO length filtering
            batch = self._dapo_length_filtering(batch)
            if len(batch) == 0:
                return None  # Skip this batch
            
            # Apply DAPO batch adjustment
            batch = self._dapo_batch_adjustment(batch)
            if batch is None:
                return None  # Skip this batch
            
            return batch
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get a summary of all training phases"""
        total_steps = sum(phase['steps'] for phase in self.phase_history)
        if self.steps_in_current_phase > 0:
            total_steps += self.steps_in_current_phase
        
        return {
            'total_steps': total_steps,
            'total_phases': len(self.phase_history) + (1 if self.steps_in_current_phase > 0 else 0),
            'current_mode': self.current_mode,
            'current_phase': self.current_phase,
            'steps_in_current_phase': self.steps_in_current_phase,
            'phase_history': self.phase_history,
            'modes_used': list(set(phase['mode'] for phase in self.phase_history)),
        }
    
    def _save_alternating_state(self):
        """Save alternating training state"""
        import json
        
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, 
            f"global_step_{self.global_steps}"
        )
        
        # Save alternating state
        alternating_state = {
            'current_mode': self.current_mode,
            'current_phase': self.current_phase,
            'steps_in_current_phase': self.steps_in_current_phase,
            'phase_history': self.phase_history,
            'alternating_config': {
                'modes': self.alternating_config.modes,
                'steps_per_phase': self.alternating_config.steps_per_phase,
                'start_mode': self.alternating_config.start_mode,
            }
        }
        
        alternating_state_path = os.path.join(local_global_step_folder, "intuitor_dapo_alternating_state.json")
        os.makedirs(local_global_step_folder, exist_ok=True)
        with open(alternating_state_path, 'w') as f:
            json.dump(alternating_state, f, indent=2)
        
        print(f"💾 Intuitor-DAPO alternating state saved to: {alternating_state_path}")
