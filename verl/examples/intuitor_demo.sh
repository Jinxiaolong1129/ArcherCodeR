#!/bin/bash
# Intuitor Algorithm Demo Script
# This script demonstrates how to use the Intuitor algorithm in VERL

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
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_new_tokens=1024 \
    algorithm.adv_estimator=intuitor \
    algorithm.kl_ctrl.kl_coeff=0.1 \
    algorithm.kl_ctrl.adaptive=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl \
    trainer.experiment_name=intuitor_demo \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 2>&1 | tee verl_intuitor_demo.log 