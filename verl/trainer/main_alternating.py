#!/usr/bin/env python3
"""
Main entry point for alternating algorithm training.

This module provides a unified interface for training with dynamic algorithm switching
between different advantage estimators (e.g., Intuitor and GRPO) without process restart.
"""

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.enhanced_alternating_trainer import EnhancedAlternatingRayPPOTrainer, EnhancedAlternatingConfig


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    """Main entry point for alternating training"""
    run_alternating_ppo(config)


def run_alternating_ppo(config: DictConfig) -> None:
    """Run alternating PPO training with dynamic algorithm switching"""
    
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
    runner = AlternatingTaskRunner.remote()
    ray.get(runner.run.remote(config))

    # Optional timeline trace
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class AlternatingTaskRunner:
    """Remote task runner for alternating training"""
    
    def run(self, config: DictConfig):
        """Execute the alternating training process"""
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        
        print("🎯 AlternatingTaskRunner started!")
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
        
        train_dataset = self._create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = self._create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = self._create_rl_sampler(config.data, train_dataset)
        
        print("🎯 Initializing EnhancedAlternatingRayPPOTrainer...")
        
        # Initialize the enhanced alternating trainer
        trainer = EnhancedAlternatingRayPPOTrainer(
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
        
        print("🚀 Starting alternating training...")
        
        # Start training
        trainer.fit()
        
        # Print final summary
        summary = trainer.get_phase_summary()
        print("\n📊 Final Training Summary:")
        print(f"   🏁 Total steps: {summary['total_steps']}")
        print(f"   📈 Total phases: {summary['total_phases']}")
        print(f"   🎯 Final algorithm: {summary['current_algorithm']}")
        
        print("✅ Alternating training completed successfully!")
    
    def _extract_alternating_config(self, config: DictConfig) -> EnhancedAlternatingConfig:
        """Extract alternating configuration from main config"""
        
        # Get alternating config from config or use defaults
        alt_config = config.get("alternating", {})
        
        algorithms = alt_config.get("algorithms", ["intuitor", "grpo"])
        steps_per_phase = alt_config.get("steps_per_phase", 50)
        start_algorithm = alt_config.get("start_algorithm", "intuitor")
        
        # Validate that start_algorithm is in the original config
        # This ensures compatibility with the base trainer
        original_algo = config.algorithm.adv_estimator
        if isinstance(original_algo, str):
            original_algo = original_algo
        else:
            original_algo = str(original_algo)
        
        # If start_algorithm is not specified, use the original algorithm
        if start_algorithm not in algorithms and original_algo in algorithms:
            start_algorithm = original_algo
        
        # Extract algorithm-specific configs if available
        algorithm_configs = alt_config.get("algorithm_configs", {})
        
        return EnhancedAlternatingConfig(
            algorithms=algorithms,
            steps_per_phase=steps_per_phase,
            start_algorithm=start_algorithm,
            algorithm_configs=algorithm_configs,
        )
    
    def _create_rl_dataset(self, data_paths, data_config, tokenizer, processor):
        """Create RL dataset (copied from main_ppo.py)"""
        from torch.utils.data import Dataset
        from verl.utils.dataset.rl_dataset import RLHFDataset
        
        if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
            from verl.utils.import_utils import load_extern_type
            dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"Custom dataset class must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset
        
        print(f"📊 Using dataset class: {dataset_cls.__name__}")
        
        dataset = dataset_cls(
            data_files=data_paths,
            tokenizer=tokenizer,
            processor=processor,
            config=data_config,
        )
        
        return dataset
    
    def _create_rl_sampler(self, data_config, dataset):
        """Create RL sampler (copied from main_ppo.py)"""
        import torch
        from torch.utils.data import RandomSampler, SequentialSampler
        
        if data_config.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(data_config.get("seed", 1))
            sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)
        
        return sampler


if __name__ == "__main__":
    main()