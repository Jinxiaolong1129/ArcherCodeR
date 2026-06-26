#!/usr/bin/env bash
# ============================================================
# Pure GRPO test launcher for Qwen2.5-Coder-7B-Instruct on LTP / PAI.
# Single node, 8 GPU. Based on run_pure_grpo_test_ltp.sh (1.5B, opt-A speedups).
# 7B differences:
#   - MODEL_PATH -> Qwen2.5-Coder-7B-Instruct (instruct model, chat template applied by verl)
#   - gen_tp=2 (4 vLLM replicas; safer KV-cache headroom for 7B). Try gen_tp=1 later.
#   - keep dynamic_bsz + offload=False; watch OOM on first run.
# Override via env for experiment sweeps: MODEL_PATH, EXP_TAG, GEN_TP.
# ============================================================
set -xeuo pipefail

CEPHFS=${CEPHFS:-/mnt/cephfs/data/processing/xiaolong.jin}
REPO_DIR=${REPO_DIR:-$CEPHFS/code/ArcherCodeR}
MODEL_PATH=${MODEL_PATH:-$CEPHFS/models/Qwen2.5-Coder-7B-Instruct}
EXP_TAG=${EXP_TAG:-7btest}

cd "$REPO_DIR"

project_name='ArcherCodeR'
exp_name="Pure-GRPO-Qwen2.5-Coder-7B-Instruct-2K-8K-16resp-no-kl-temp0.8-${EXP_TAG}"

nnodes=1
adv_estimator=grpo

# kl config - DISABLED for pure GRPO
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
kl_loss_type=low_var_kl

# clip
clip_ratio_low=0.2
clip_ratio_high=0.2
loss_agg_mode=token-mean

# Sequence lengths
max_prompt_length=$((1024 * 2))     # 2K
max_response_length=$((1024 * 8))   # 8K
enable_overlong_buffer=False
overlong_buffer_len=16
overlong_penalty_factor=1.0
v_max_response_length=$((1024 * 8)) # 8K

# Batch sizes
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=32

# Data (pre-staged on cephfs)
data_dir=$REPO_DIR/data
TRAIN_FILE=$data_dir/train/archercoder-1.5b-train.json
TEST_FILE=$data_dir/test/livecodebench_v5.json

CKPTS_DIR=$REPO_DIR/output/${project_name}/${exp_name}

# Response generation
n_resp_per_prompt=16
temperature=0.8
top_p=1.0
top_k=-1
v_n=4
v_temperature=0.8
v_top_p=1.0
v_top_k=-1

# Performance (7B)
#   gen_tp=2: 4 vLLM replicas, TP=2 across the 8 GPUs (safe KV-cache headroom for 7B)
#   dynamic_bsz kept (opt A); offload off first, enable if OOM.
sp_size=1
gen_tp=${GEN_TP:-2}
use_dynamic_bsz=True
micro_batch_size_per_gpu=1
actor_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
offload=${OFFLOAD:-False}   # set OFFLOAD=True to param/optimizer-offload (anti-OOM for long-response 7B)
use_overlong_filter=False

mkdir -p "${CKPTS_DIR}" "${CKPTS_DIR}/eval"

# wandb offline (GPU node has no external net): data -> cephfs, `wandb sync` later from cpu-2
export WANDB_MODE=offline
export WANDB_DIR="${CKPTS_DIR}"

echo "🚀 PURE GRPO 7B TEST (LTP) | model=${MODEL_PATH} | gen_tp=${gen_tp} | gpus=8 | nnodes=${nnodes}"

python -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.reward_fn_key=data_source \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + v_max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + v_max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${v_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${v_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${v_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${v_n} \
    +actor_rollout_ref.rollout.val_kwargs.response_length=${v_max_response_length} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=wizard \
    reward_model.use_general_reward=True \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${nnodes}" \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.save_first_step=20 \
    +trainer.save_interval_after_first=30 \
    +trainer.max_actor_ckpt_to_keep=10 \
    +trainer.max_critic_ckpt_to_keep=1 \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    +trainer.enable_overlong_filter=${use_overlong_filter} \
    +trainer.rejection_sample=False "$@" 2>&1 | tee "${CKPTS_DIR}/${project_name}_${exp_name}_grpo.log"
