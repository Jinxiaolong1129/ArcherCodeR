#!/usr/bin/env bash
set -xeuo pipefail

# Intuitor with WizardRewardManager + rewards/general_reward.py for validation (like DAPO)
# This script demonstrates how to use Intuitor with rich analysis and the same validation functions as DAPO
# Training: uses self-certainty (dummy reward), Validation: uses rewards/general_reward.py
# Features: thinking tokens analysis, repetition detection, response length analysis, etc.

export WANDB_API_KEY=YOUR_WANDB_API_KEY
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    algorithm.norm_adv_by_std_in_grpo=true \
    data.train_files=$HOME/data/code/train.parquet \
    data.val_files=$HOME/data/code/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=wizard \
    reward_model.use_general_reward=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl_comparison \
    trainer.experiment_name=intuitor_with_rewards \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee verl_intuitor_rewards.log
