#!/usr/bin/env bash
set -xeuo pipefail

# ================================
# DACE Training Script v2 for ArcherCodeR
# Exploring different hyperparameters
# α_scale=0.1 (doubled), β_threshold=0.5 (more strict)
# ================================

# Import environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "✅ Loaded environment variables from .env"
    echo -e "🔑 WANDB_API_KEY: ${WANDB_API_KEY:0:8}..."
    echo -e "🔑 HF_TOKEN: ${HF_TOKEN:0:8}..."
else
    echo -e "⚠️  Warning: .env file not found. Please create .env file with WANDB_API_KEY and HF_TOKEN"
fi

nnodes=1

project_name='ArcherCodeR'
exp_name='DACE-v2-Qwen2.5-1.5B-alpha0.1-beta0.5'

# ================================
# DACE Configuration v2
# ================================
adv_estimator=dace

# DACE-specific hyperparameters (MODIFIED for exploration)
dace_alpha_scale=0.1           # Doubled from 0.05 → stronger intrinsic reward
dace_beta_threshold=0.5        # Increased from 0.4 → stricter hard task definition
norm_adv_by_std_in_grpo=True   # Normalize advantages

# KL config - DISABLED (DACE uses intrinsic rewards instead)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
kl_loss_type=low_var_kl

# Clip ratios
clip_ratio_low=0.2
clip_ratio_high=0.2
loss_agg_mode=token-mean

# ================================
# Sequence Lengths
# ================================
max_prompt_length=$((1024 * 2))    # 2K
max_response_length=$((1024 * 8))  # 8K
enable_overlong_buffer=False
overlong_buffer_len=16
overlong_penalty_factor=1.0
v_max_response_length=$((1024 * 8))  # 8K for validation

# ================================
# Batch Sizes
# ================================
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=32

# ================================
# Paths
# ================================
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
CKPTS_DIR=./output/${project_name}/${exp_name}
data_dir=./data
TRAIN_FILE=$data_dir/train/archercoder-1.5b-train.json
TEST_FILE=$data_dir/test/livecodebench_v5.json

# ================================
# Response Generation (Important for DACE!)
# ================================
# DACE requires multiple responses per prompt for difficulty estimation
n_resp_per_prompt=16      # More responses = better difficulty estimation
temperature=1.0
top_p=1.0
top_k=-1                  # -1 for vLLM rollout, 0 for HF rollout

# Validation settings
v_n=4
v_temperature=0.8
v_top_p=1.0
v_top_k=-1

# ================================
# Performance Settings
# ================================
sp_size=1
gen_tp=2
use_dynamic_bsz=False
micro_batch_size_per_gpu=1
actor_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
offload=False

# ================================
# Token Mask (Optional - can disable for DACE)
# ================================
use_token_entropy_separate=False  # Disabled for pure DACE
token_entropy_quantile=0.8
high_entropy_kl_loss_scale_coef=0.0
low_entropy_clip_ratio_low=0.2
low_entropy_clip_ratio_high=0.2
high_entropy_clip_ratio_low=0.5
high_entropy_clip_ratio_high=0.5

# ================================
# Trainer Settings
# ================================
use_overlong_filter=False

echo "🚀 DACE v2 CONFIGURATION (Exploring Stronger Intrinsic Rewards):"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ DACE v2: Enhanced Intrinsic Reward & Stricter Difficulty     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "🤖 Model: ${MODEL_PATH}"
echo "📏 Max prompt length: ${max_prompt_length}"
echo "📏 Max response length: ${max_response_length}"
echo "📦 Batch size: ${train_prompt_bsz}"
echo "🔢 Responses per prompt: ${n_resp_per_prompt} (for difficulty estimation)"
echo "⚡ Tensor parallel: ${gen_tp}"
echo "🎯 Total tokens per batch: $((train_prompt_bsz * n_resp_per_prompt * v_max_response_length))"
echo ""
echo "🎲 DACE v2 Hyperparameters (MODIFIED):"
echo "  ├─ α_scale: ${dace_alpha_scale} (paper: 0.05, v1: 0.05, v2: 0.1 ⬆️ 2x)"
echo "  ├─ β_threshold: ${dace_beta_threshold} (paper: 0.4, v1: 0.4, v2: 0.5 ⬆️)"
echo "  └─ Advantage normalization: ${norm_adv_by_std_in_grpo}"
echo ""
echo "💡 Expected Effects of v2 Changes:"
echo "  📈 α_scale 0.05→0.1 (doubled):"
echo "     • Stronger intrinsic reward signal (~15-20% contribution vs ~8%)"
echo "     • More aggressive exploration on hard tasks"
echo "     • Larger penalties for uncertain responses"
echo ""
echo "  📈 β_threshold 0.4→0.5 (increased):"
echo "     • More tasks classified as 'hard' (need >50% success to be easy)"
echo "     • More samples get negative α (exploration mode)"
echo "     • Harder to switch to exploitation mode"
echo ""
echo "💡 How DACE v2 works:"
echo "  • Estimates task difficulty: diff(x) = 1 - success_rate"
echo "  • Computes certainty: C(y,x) = -mean(log_prob)"
echo "  • Adaptive coefficient: α = α_scale × sign(β_threshold - diff)"
echo "  • For HARD tasks (diff > 0.5): α = -0.1 → explore (2x stronger!)"
echo "  • For EASY tasks (diff < 0.5): α = +0.1 → exploit (2x stronger!)"
echo "  • Total reward: R = R_ext + α × C"
echo "╚═══════════════════════════════════════════════════════════════╝"

mkdir -p "${CKPTS_DIR}"
mkdir -p "${CKPTS_DIR}/eval"

# ================================
# Run DACE v2 Training
# ================================
/home/ec2-user/miniconda3/envs/archer/bin/python -m dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    +algorithm.dace_alpha_scale=${dace_alpha_scale} \
    +algorithm.dace_beta_threshold=${dace_beta_threshold} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    +actor_rollout_ref.actor.use_token_entropy_separate=${use_token_entropy_separate} \
    +actor_rollout_ref.actor.high_entropy_kl_loss_scale_coef=${high_entropy_kl_loss_scale_coef} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_low=${low_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_high=${low_entropy_clip_ratio_high} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_low=${high_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_high=${high_entropy_clip_ratio_high} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${nnodes}" \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.max_actor_ckpt_to_keep=30 \
    +trainer.max_critic_ckpt_to_keep=30 \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    +trainer.enable_overlong_filter=${use_overlong_filter} \
    +trainer.rejection_sample=True $@ 2>&1 | tee ${CKPTS_DIR}/${project_name}_${exp_name}_dace-v2.log

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ ✅ DACE v2 Training Completed                                 ║"
echo "║ 📁 Logs: ${CKPTS_DIR}/${project_name}_${exp_name}_dace-v2.log ║"
echo "║ 💾 Checkpoints: ${CKPTS_DIR}                                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"



