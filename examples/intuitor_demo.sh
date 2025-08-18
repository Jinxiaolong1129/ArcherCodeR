#!/bin/bash
# Intuitor Algorithm Demo Script
# This script demonstrates how to use the Intuitor algorithm in VERL

set -x

export WANDB_API_KEY=your_wandb_key_here
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=path/to/your/train/data.parquet \
    data.val_files=path/to/your/val/data.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    model.path=path/to/your/model \
    actor_rollout_ref.model.path=path/to/your/model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.sync_ref_model=True \
    actor_rollout_ref.ref.ref_model_sync_steps=1 \
    actor_rollout_ref.ref.ref_model_mixup_alpha=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    algorithm.adv_estimator=intuitor \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl \
    trainer.experiment_name=demo_intuitor \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 2>&1 | tee verl_demo_intuitor.log 