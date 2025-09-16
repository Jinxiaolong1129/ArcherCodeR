#!/usr/bin/env python3
"""
Main entry point for Intuitor-DAPO alternating training.

This module provides alternating training between:
1. Intuitor algorithm (self-certainty based rewards)
2. DAPO training methodology (rejection sampling + external rewards)
"""

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.intuitor_dapo_alternating_trainer import IntuitorDAPOAlternatingTrainer, IntuitorDAPOAlternatingConfig


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    """Main entry point for Intuitor-DAPO alternating training"""
    run_intuitor_dapo_alternating(config)


def run_intuitor_dapo_alternating(config: DictConfig) -> None:
    """Run Intuitor-DAPO alternating training"""
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN", 
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    # Create remote task runner
    runner = IntuitorDAPOTaskRunner.remote()
    ray.get(runner.run.remote(config))

    # Optional timeline trace
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class IntuitorDAPOTaskRunner:
    """Remote task runner for Intuitor-DAPO alternating training"""
    
    def run(self, config: DictConfig):
        """Execute the Intuitor-DAPO alternating training process"""
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        
        print("🎯 IntuitorDAPOTaskRunner started!")
        print("=" * 80)
        
        # Print and resolve configuration
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # Extract alternating configuration
        alternating_config = self._extract_alternating_config(config)
        print(f"🔄 Alternating Config: {alternating_config}")
        
        # Download model checkpoint
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, 
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        # Initialize tokenizer and processor
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Version validation for vLLM
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")
        
        # Define worker classes based on strategy
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")
        
        # Set up role worker mapping
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        
        # Resource pool configuration
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        
        # Add reward model worker if enabled
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(f"Reward model strategy {config.reward_model.strategy} not supported")
            
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # Add reference policy worker if needed
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        
        # Load reward functions
        from verl.trainer.ppo.reward import load_reward_manager
        
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, for_validation=False,
            **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, for_validation=True,
            **config.reward_model.get("reward_kwargs", {})
        )
        
        # Create resource pool manager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )
        
        # Create datasets
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        
        print("🎯 Initializing IntuitorDAPOAlternatingTrainer...")
        
        # Initialize the Intuitor-DAPO alternating trainer
        trainer = IntuitorDAPOAlternatingTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            alternating_config=alternating_config,
        )
        
        # Initialize workers
        trainer.init_workers()
        
        print("🚀 Starting Intuitor-DAPO alternating training...")
        
        # Start training
        trainer.fit()
        
        # Print final summary
        summary = trainer.get_phase_summary()
        print("\n📊 Final Training Summary:")
        print(f"   🏁 Total steps: {summary['total_steps']}")
        print(f"   📈 Total phases: {summary['total_phases']}")
        print(f"   🎯 Final mode: {summary['current_mode']}")
        
        print("✅ Intuitor-DAPO alternating training completed successfully!")
    
    def _extract_alternating_config(self, config: DictConfig) -> IntuitorDAPOAlternatingConfig:
        """Extract alternating configuration from main config"""
        
        # Get alternating config from config or use defaults
        alt_config = config.get("alternating", {})
        
        modes = alt_config.get("modes", ["intuitor", "dapo"])
        steps_per_phase = alt_config.get("steps_per_phase", 50)
        start_mode = alt_config.get("start_mode", "intuitor")
        
        # DAPO specific settings
        dapo_rejection_sample = alt_config.get("dapo_rejection_sample", True)
        dapo_enable_overlong_filter = alt_config.get("dapo_enable_overlong_filter", True)
        
        return IntuitorDAPOAlternatingConfig(
            modes=modes,
            steps_per_phase=steps_per_phase,
            start_mode=start_mode,
            dapo_rejection_sample=dapo_rejection_sample,
            dapo_enable_overlong_filter=dapo_enable_overlong_filter,
        )


if __name__ == "__main__":
    main()
