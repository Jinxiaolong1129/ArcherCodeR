#!/usr/bin/env bash
#
# Evaluate AIME24 for two specific checkpoints:
#   - global_step_50
#   - global_step_80
# Experiment:
#   Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-clip_ratio_c10-ppo_epoch3
#

set -euo pipefail

# ----------------------------
# Basic environment
# ----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PROJ_DIR="${PROJ_DIR:-/data_storage/wyj/jxl/ArcherCodeR}"
EVAL_DIR="${EVAL_DIR:-${PROJ_DIR}/Archer_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJ_DIR}/output/self-rl-jxl}"
DATA_DIR="${DATA_DIR:-${EVAL_DIR}/data/test}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${EVAL_DIR}:${PYTHONPATH:-}"

# ----------------------------
# Fixed target
# ----------------------------
EXP_NAME="Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-clip_ratio_c10-ppo_epoch3"
STEPS=(50 80)
DATASET="aime2024"

# Optional override, e.g. STEP_LIST="50" or STEP_LIST="50,80"
if [[ -n "${STEP_LIST:-}" ]]; then
  IFS=',' read -r -a STEPS <<< "${STEP_LIST}"
fi

# ----------------------------
# Eval settings (8k output)
# ----------------------------
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-32}"         # pass@32
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-output_aime24_8k_n${N_SAMPLES}}"
MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}"

LOG_DIR="${EVAL_DIR}/tools/logs"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/eval_aime24_clip_c10_steps50_80_$(date +%Y%m%d_%H%M%S).log"
RAY_TMP_ROOT_REAL="${RAY_TMP_ROOT_REAL:-${PROJ_DIR}/.ray_tmp}"
RAY_TMP_LINK="${RAY_TMP_LINK:-/tmp/rayds}"
mkdir -p "${RAY_TMP_ROOT_REAL}"

# Keep storage under ${PROJ_DIR}/.ray_tmp, but expose a short symlink path
# to avoid AF_UNIX socket path length limit in Ray.
ln -sfn "${RAY_TMP_ROOT_REAL}" "${RAY_TMP_LINK}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MAIN_LOG}"
}

check_checkpoint_exists() {
  local step="$1"
  local ckpt_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${step}/actor"
  [[ -d "${ckpt_path}" && -f "${ckpt_path}/config.json" ]]
}

check_hf_model_exists() {
  local step="$1"
  local hf_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${step}/actor/hf_model"
  [[ -d "${hf_path}" && -f "${hf_path}/config.json" ]]
}

merge_model_if_needed() {
  local step="$1"
  local ckpt_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${step}/actor"
  local hf_path="${ckpt_path}/hf_model"

  if check_hf_model_exists "${step}"; then
    log "✓ HF model already exists, skip merge: ${hf_path}"
    return 0
  fi

  log "🔧 Merge model for step ${step}"
  local start_ts
  start_ts=$(date +%s)

  "${PYTHON_BIN}" -m tools.model_merge merge \
    --backend fsdp \
    --local_dir "${ckpt_path}" \
    --target_dir "${hf_path}" 2>&1 | tee -a "${MAIN_LOG}"

  local exit_code=${PIPESTATUS[0]}
  local end_ts
  end_ts=$(date +%s)
  local cost=$((end_ts - start_ts))

  if [[ ${exit_code} -eq 0 && -f "${hf_path}/config.json" ]]; then
    log "✓ Merge success for step ${step} (${cost}s)"
    return 0
  fi

  log "✗ Merge failed for step ${step}"
  [[ -d "${hf_path}" ]] && rm -rf "${hf_path}"
  return 1
}

run_eval() {
  local step="$1"
  local model_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${step}/actor/hf_model"
  local output_dir="${model_path}/${OUTPUT_SUBDIR}"
  local output_path="${output_dir}/${DATASET}.parquet"
  local ray_tmp_dir="${RAY_TMP_LINK}/s${step}"

  mkdir -p "${ray_tmp_dir}"
  export RAY_TMPDIR="${ray_tmp_dir}"

  mkdir -p "${output_dir}"

  if [[ -f "${output_path}" ]]; then
    log "✓ Eval output exists, skip: ${output_path}"
    return 0
  fi

  log "📊 Start eval: ${EXP_NAME} step ${step} dataset=${DATASET} n=${N_SAMPLES}"
  log "🗂 Ray tmp dir (short): ${RAY_TMPDIR}"
  log "🗂 Ray tmp real path: ${RAY_TMP_ROOT_REAL}"
  local start_ts
  start_ts=$(date +%s)

  "${PYTHON_BIN}" -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    +trainer.project_name=ArcherCodeR_Eval \
    +trainer.experiment_name="${EXP_NAME}" \
    +trainer.task_name="${DATASET}" \
    +trainer.global_step="${step}" \
    +trainer.use_wandb=False \
    model.path="${model_path}" \
    data.path="${DATA_DIR}/${DATASET}.parquet" \
    data.output_path="${output_path}" \
    data.batch_size="${BATCH_SIZE}" \
    data.n_samples="${N_SAMPLES}" \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.9 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    rollout.disable_log_stats=False \
    rollout.tensor_model_parallel_size="${TP_SIZE}" \
    rollout.temperature="${TEMPERATURE}" \
    rollout.top_k=-1 \
    rollout.top_p="${TOP_P}" \
    rollout.prompt_length="${MAX_PROMPT_LEN}" \
    rollout.response_length="${MAX_RESPONSE_LEN}" \
    rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    2>&1 | tee -a "${MAIN_LOG}"

  local exit_code=${PIPESTATUS[0]}
  local end_ts
  end_ts=$(date +%s)
  local cost=$((end_ts - start_ts))

  if [[ ${exit_code} -eq 0 && -f "${output_path}" ]]; then
    log "✓ Eval success for step ${step} (${cost}s)"
    return 0
  fi

  log "✗ Eval failed for step ${step}"
  return 1
}

main() {
  log "============================================================"
  log "AIME24 eval for two checkpoints (step 50 / 80)"
  log "Experiment: ${EXP_NAME}"
  log "OUTPUT_ROOT: ${OUTPUT_ROOT}"
  log "DATA: ${DATA_DIR}/${DATASET}.parquet"
  log "GPU: ${CUDA_VISIBLE_DEVICES} (n_gpus=${N_GPUS}, tp=${TP_SIZE})"
  log "n_samples=${N_SAMPLES}, response_len=${MAX_RESPONSE_LEN}"
  log "============================================================"

  local ok=0
  local fail=0

  for step in "${STEPS[@]}"; do
    log ""
    log "------------------ step ${step} ------------------"
    if ! check_checkpoint_exists "${step}"; then
      log "⚠ checkpoint missing, skip step ${step}"
      continue
    fi

    if [[ "${MERGE_IF_NEEDED}" == "1" ]]; then
      if merge_model_if_needed "${step}" && run_eval "${step}"; then
        ok=$((ok + 1))
      else
        fail=$((fail + 1))
      fi
    else
      if ! check_hf_model_exists "${step}"; then
        log "⚠ hf_model missing and MERGE_IF_NEEDED=0, skip step ${step}"
        continue
      fi
      if run_eval "${step}"; then
        ok=$((ok + 1))
      else
        fail=$((fail + 1))
      fi
    fi
  done

  log ""
  log "==================== DONE ========================"
  log "success=${ok}, failed=${fail}"
  log "log_file=${MAIN_LOG}"
  log "=================================================="
}

cd "${EVAL_DIR}"
main

