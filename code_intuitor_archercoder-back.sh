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

# Configuration variables (similar to Archer script)
project_name='ArcherCodeR'
exp_name='Archer-Intuitor-Qwen2.5-1.5B'
nnodes=1

# Sequence lengths (matching Archer script)
max_prompt_length=$((1024 * 2))  # 2K
max_response_length=$((1024 * 8))   # 16K ⚡️
v_max_response_length=$((1024 * 16)) # 16K ⚡️

# Batch sizes (adjusted for Intuitor)
train_prompt_bsz=64  # 改为64，与archer脚本一致
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=32  # 改为32，与archer脚本一致

# Model and data paths
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Add checkpoint and evaluation directories
CKPTS_DIR=./output/${project_name}/${exp_name}
data_dir=./data
TRAIN_FILE=$data_dir/train/archercoder-1.5b-train.json
TEST_FILE=$data_dir/test/livecodebench_v5.json

# Response generation (matching Archer validation settings)
n_resp_per_prompt=16  # 改为16，与archer脚本一致
temperature=1.0  # 改为1.0，与archer脚本一致
top_p=1.0
top_k=-1
v_n=4  # 改为4，与archer脚本一致
v_temperature=0.8  # 保持0.8
v_top_p=1.0
v_top_k=-1
v_do_sample=true  # 明确设置为true

# Performance settings
gen_tp=2  # 改为2，与archer脚本一致
micro_batch_size_per_gpu=1  # 改为1，与archer脚本一致
actor_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
offload=False

echo "🚀 INTUITOR CONFIGURATION:"
echo "🤖 Model: ${MODEL_PATH}"
echo "📏 Max prompt length: ${max_prompt_length}"
echo "📏 Max response length: ${max_response_length}"
echo "📦 Batch size: ${train_prompt_bsz}"
echo "🔢 Responses per prompt: ${n_resp_per_prompt}"
echo "🎯 Algorithm: Intuitor (self-certainty + livecodebench validation)"
echo "🎲 Validation sampling: n=${v_n}, do_sample=${v_do_sample}, temperature=${v_temperature}"

# Create output and evaluation directories
mkdir -p "${CKPTS_DIR}"
mkdir -p "${CKPTS_DIR}/eval"

# 使用标准PPO训练，Intuitor训练时用self-certainty，验证时用livecodebench reward
PYTHONUNBUFFERED=1 /home/ubuntu/miniconda/envs/archer/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.train_batch_size=${train_prompt_bsz} \
    +data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + v_max_response_length)) \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.val_kwargs.n=${v_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${v_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${v_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${v_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${v_top_k} \
    +actor_rollout_ref.rollout.val_kwargs.response_length=${v_max_response_length} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=wizard \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnodes} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    +trainer.max_actor_ckpt_to_keep=3 \
    +trainer.max_critic_ckpt_to_keep=3 \
    trainer.balance_batch=False 2>&1 | tee verl_${exp_name}_intuitor.log 