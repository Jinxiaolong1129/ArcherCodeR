#!/usr/bin/env bash
# ============================================================
# Pure GRPO on Qwen3-4B-Instruct-2507, verl 0.4.1 (vllm 0.8.5, Qwen3 OK).
# Runs INSIDE the verl0.4 base container (1 node x 8 GPU).
# verl source = repo's verl04/ (verl0.4.1 + wizard reward manager).
# Reward = rewards/general_reward.py:general_reward_fn via verl's native
#          custom_reward_function loader (no reward.py patch needed).
# First-bringup config: conservative (static bsz, gen_tp=2, gpu_mem 0.6,
# ref offload on, skip validation) — get it RUNNING first, speed-tune later.
# ============================================================
set -xeuo pipefail

CEPHFS=${CEPHFS:-/mnt/cephfs/data/processing/xiaolong.jin}
REPO_DIR=${REPO_DIR:-$CEPHFS/code/ArcherCodeR}
MODEL_PATH=${MODEL_PATH:-$CEPHFS/models/Qwen3-4B-Instruct-2507}
EXP_NAME=${EXP_NAME:-Pure-GRPO-Qwen3-4B-Instruct-2507-2K-8K-16resp-no-kl}
GEN_TP=${GEN_TP:-2}
# ENFORCE_EAGER=True is the PROVEN colocated path (free_cache_engine=True). enforce_eager=False
# = compile path, which verl requires paired with free_cache_engine=False (CUDA graph can't
# coexist with freeing the vllm cache engine) + a lower GPU_MEM. Default to the proven path.
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
FREE_CACHE=${FREE_CACHE:-True}
GPU_MEM=${GPU_MEM:-0.6}
# adv estimator: grpo (default) or internal-signal methods (intuitor/token_entropy/
# trajectory_entropy/prob_disparity). Internal methods canonically use temperature=1.0.
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
TEMPERATURE=${TEMPERATURE:-0.8}
# eval: TEST_FREQ>0 runs validation on livecodebench_v5; SAVE_FREQ>0 saves checkpoints.
TEST_FREQ=${TEST_FREQ:--1}
SAVE_FREQ=${SAVE_FREQ:--1}

# rewards/ pkg needs `from rewards import ...` (so $REPO_DIR on path), but $REPO_DIR
# also contains the OLD verl/ (0.1) which would SHADOW our pip-installed verl04 ->
# put verl04 FIRST so `import verl` resolves to verl0.4.1, then $REPO_DIR for rewards.
export PYTHONPATH="$REPO_DIR/verl04:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"
CKPTS_DIR=$REPO_DIR/output/ArcherCodeR-4B/${EXP_NAME}
mkdir -p "$CKPTS_DIR"

max_prompt_length=$((1024 * 2))    # 2K
max_response_length=$((1024 * 8))  # 8K

echo "🚀 PURE GRPO 4B | model=${MODEL_PATH} | gen_tp=${GEN_TP} | gpus=8"

echo "🚀 4B | adv_estimator=${ADV_ESTIMATOR} | temp=${TEMPERATURE} | test_freq=${TEST_FREQ} save_freq=${SAVE_FREQ} | enforce_eager=${ENFORCE_EAGER}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    data.train_files="$REPO_DIR/data/train/archercoder-1.5b-train.json" \
    data.val_files="$REPO_DIR/data/test/livecodebench_v5.json" \
    data.prompt_key=prompt \
    data.train_batch_size=64 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE} \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    +actor_rollout_ref.rollout.val_kwargs.response_length=${max_response_length} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=wizard \
    reward_model.use_general_reward=True \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console'] \
    trainer.project_name='ArcherCodeR-4B' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" "$@" 2>&1 | tee "${CKPTS_DIR}/grpo_4b.log"
