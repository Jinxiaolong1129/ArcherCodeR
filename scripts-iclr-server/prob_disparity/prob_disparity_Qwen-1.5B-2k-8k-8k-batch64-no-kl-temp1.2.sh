#!/usr/bin/env bash
set -xeuo pipefail


# 导入环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "✅ Loaded environment variables from .env$"
    echo -e "🔑 WANDB_API_KEY: ${WANDB_API_KEY:0:8}...$"
    echo -e "🔑 HF_TOKEN: ${HF_TOKEN:0:8}...$"
else
    echo -e "⚠️  Warning: .env file not found. Please create .env file with WANDB_API_KEY and HF_TOKEN"
fi


nnodes=1

project_name='ArcherCodeR'
exp_name='Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2'

adv_estimator=prob_disparity

# kl config - NO KL LOSS
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False

# Sequence lengths
max_prompt_length=$((1024 * 2))  # 2K
max_response_length=$((1024 * 8))  # 8K
v_max_response_length=$((1024 * 8))  # 8K

# Batch sizes
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=32

# Paths
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
CKPTS_DIR=./output/${project_name}/${exp_name}
data_dir=./data
TRAIN_FILE=$data_dir/train/archercoder-1.5b-train.json
TEST_FILE=$data_dir/test/livecodebench_v5.json

# Response generation - TEMPERATURE 1.2
n_resp_per_prompt=16
temperature=1.2
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
v_n=4
v_temperature=0.8
v_top_p=1.0
v_top_k=-1

# Performance settings
gen_tp=2
micro_batch_size_per_gpu=1
actor_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
offload=False

echo "🚀 PROBABILITY DISPARITY CONFIGURATION (NO KL LOSS - TEMP 1.2):"
echo "🤖 Model: ${MODEL_PATH}"
echo "📏 Max prompt length: ${max_prompt_length}"
echo "📏 Max response length: ${max_response_length}"
echo "📦 Batch size: ${train_prompt_bsz}"
echo "🔢 Responses per prompt: ${n_resp_per_prompt}"
echo "⚡ Tensor parallel: ${gen_tp}"
echo "🎯 Algorithm: Probability Disparity (max - second_max probability)"
echo "🌡️ Training temperature: ${temperature}"
echo "🎲 Validation sampling: n=${v_n}, do_sample=true, temperature=${v_temperature}"
echo "❌ KL Loss: DISABLED"

mkdir -p "${CKPTS_DIR}"
mkdir -p "${CKPTS_DIR}/eval"

/data/xuandong_zhao/anaconda3/envs/archer/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
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
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=${v_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${v_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${v_top_k} \
    +actor_rollout_ref.rollout.val_kwargs.response_length=${v_max_response_length} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    reward_model.reward_manager=wizard \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${nnodes}" \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.max_actor_ckpt_to_keep=20 \
    +trainer.max_critic_ckpt_to_keep=1 \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    trainer.balance_batch=False $@ 2>&1 | tee ${CKPTS_DIR}/${project_name}_${exp_name}_prob_disparity.log




