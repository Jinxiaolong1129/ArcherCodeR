#!/usr/bin/env bash
#
# LCB speed test script (generation stage only).
# - Default: pass@8, 8k response
# - Default target: one checkpoint for quick throughput check
#

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PROJ_DIR="${PROJ_DIR:-/data_storage/wyj/jxl/ArcherCodeR}"
EVAL_DIR="${EVAL_DIR:-${PROJ_DIR}/Archer_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJ_DIR}/output/self-rl-jxl}"
DATA_DIR="${DATA_DIR:-${EVAL_DIR}/data/test}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${EVAL_DIR}:${PYTHONPATH:-}"

# Keep real tmp in project dir, use short symlink for Ray socket path length.
RAY_TMP_ROOT_REAL="${RAY_TMP_ROOT_REAL:-${PROJ_DIR}/.ray_tmp}"
RAY_TMP_LINK="${RAY_TMP_LINK:-/tmp/rayds}"
mkdir -p "${RAY_TMP_ROOT_REAL}"
ln -sfn "${RAY_TMP_ROOT_REAL}" "${RAY_TMP_LINK}"

# ----------------------------
# Target checkpoint (override if needed)
# ----------------------------
EXP_NAME="${EXP_NAME:-Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-clip_ratio_c10-ppo_epoch3}"
STEP="${STEP:-80}"
DATASET="${DATASET:-livecodebench_v5}"

# ----------------------------
# Eval settings
# ----------------------------
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-8}"      # pass@8
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-output_lcbv6_8k_n${N_SAMPLES}_speedtest}"

LOG_DIR="${EVAL_DIR}/tools/logs"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/eval_lcbv6_speed_test_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MAIN_LOG}"
}

check_checkpoint_exists() {
  local ckpt_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${STEP}/actor"
  [[ -d "${ckpt_path}" && -f "${ckpt_path}/config.json" ]]
}

check_hf_model_exists() {
  local hf_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${STEP}/actor/hf_model"
  [[ -d "${hf_path}" && -f "${hf_path}/config.json" ]]
}

merge_model_if_needed() {
  local ckpt_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${STEP}/actor"
  local hf_path="${ckpt_path}/hf_model"

  if check_hf_model_exists; then
    log "✓ HF model already exists, skip merge: ${hf_path}"
    return 0
  fi

  if [[ "${MERGE_IF_NEEDED}" != "1" ]]; then
    log "✗ hf_model missing and MERGE_IF_NEEDED=0"
    return 1
  fi

  log "🔧 Merge model: ${EXP_NAME} step ${STEP}"
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
    log "✓ Merge success (${cost}s)"
    return 0
  fi

  log "✗ Merge failed"
  [[ -d "${hf_path}" ]] && rm -rf "${hf_path}"
  return 1
}

run_eval() {
  local model_path="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${STEP}/actor/hf_model"
  local output_dir="${model_path}/${OUTPUT_SUBDIR}"
  local output_path="${output_dir}/${DATASET}.parquet"
  local ray_tmp_dir="${RAY_TMP_LINK}/l6s${STEP}"

  mkdir -p "${ray_tmp_dir}" "${output_dir}"
  export RAY_TMPDIR="${ray_tmp_dir}"

  if [[ ! -f "${DATA_DIR}/${DATASET}.parquet" && ! -f "${DATA_DIR}/${DATASET}.json" ]]; then
    log "✗ Missing dataset file: ${DATA_DIR}/${DATASET}.parquet or .json"
    return 1
  fi

  log "📊 Start LCB speed test"
  log "Experiment: ${EXP_NAME}"
  log "Step: ${STEP}"
  log "GPU: ${CUDA_VISIBLE_DEVICES} (n_gpus=${N_GPUS}, tp=${TP_SIZE})"
  log "n_samples=${N_SAMPLES}, prompt=${MAX_PROMPT_LEN}, response=${MAX_RESPONSE_LEN}"
  log "Ray tmp (short): ${RAY_TMPDIR}"
  log "Ray tmp (real): ${RAY_TMP_ROOT_REAL}"
  log "Output: ${output_path}"

  local start_ts
  start_ts=$(date +%s)

  "${PYTHON_BIN}" -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    +trainer.project_name=ArcherCodeR_Eval \
    +trainer.experiment_name="${EXP_NAME}" \
    +trainer.task_name="${DATASET}" \
    +trainer.global_step="${STEP}" \
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
    log "✓ LCB generation finished (${cost}s)"
    return 0
  fi

  log "✗ LCB generation failed"
  return 1
}

main() {
  log "============================================================"
  log "LCB speed test script"
  log "============================================================"

  if ! check_checkpoint_exists; then
    log "✗ checkpoint not found: ${OUTPUT_ROOT}/${EXP_NAME}/global_step_${STEP}"
    exit 1
  fi

  merge_model_if_needed
  run_eval

  log "============================================================"
  log "DONE"
  log "log_file=${MAIN_LOG}"
  log "============================================================"
}

cd "${EVAL_DIR}"
main

