#!/bin/bash

set -e
set -x

# 导入环境变量
export $(grep -v '^#' .env | xargs)
echo "Your WANDB_API_KEY is: $WANDB_API_KEY"
echo "Your HF_TOKEN is: $HF_TOKEN"

export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

# 设置Ray使用/mnt目录
export RAY_TMPDIR="/mnt/ray_tmp"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

echo "Ray temp directory: $RAY_TMPDIR"

# 创建Ray临时目录
mkdir -p "$RAY_TMPDIR"

# 停止之前的Ray会话
echo "Stopping previous Ray sessions..."
ray stop --force 2>/dev/null || true
sleep 3

PYTHONUNBUFFERED=1 /home/ubuntu/miniconda/envs/archer/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    data.train_files=/mnt/people/zhuoterq/xiaolong-swebench/ArcherCodeR/data/math/train.parquet \
    data.val_files=/mnt/people/zhuoterq/xiaolong-swebench/ArcherCodeR/data/math/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl \
    trainer.experiment_name=math_intuitor_archercoder \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee verl_math_intuitor_archercoder.log 