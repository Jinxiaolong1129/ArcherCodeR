#!/usr/bin/env bash
#
# Generic worker for A/B/C/D experiment batches.
# - Splits experiments by WORKER_INDEX / WORKER_TOTAL (round-robin).
# - Iterates steps (default: 10,50,80,105), missing checkpoints are skipped.
# - Optional merge to hf_model before eval (MERGE_IF_NEEDED=1 by default).
# - Supports any dataset file under data/test/{dataset}.parquet|json.
#

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-/data_storage/wyj/jxl/ArcherCodeR}"
EVAL_DIR="${EVAL_DIR:-${PROJ_DIR}/Archer_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJ_DIR}/output/self-rl-jxl}"
DATA_DIR="${DATA_DIR:-${EVAL_DIR}/data/test}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GROUP="${GROUP:?GROUP is required (A|B|C|D)}"
WORKER_INDEX="${WORKER_INDEX:?WORKER_INDEX is required, 1-based}"
WORKER_TOTAL="${WORKER_TOTAL:?WORKER_TOTAL is required}"

DATASET="${DATASET:-livecodebench_v5}"
STEPS_CSV="${STEPS_CSV:-10,50,80,105}"
N_GPUS="${N_GPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-output_${DATASET}_8k_n${N_SAMPLES}}"
MERGE_IF_NEEDED="${MERGE_IF_NEEDED:-1}"

RAY_TMP_ROOT_REAL="${RAY_TMP_ROOT_REAL:-${PROJ_DIR}/.ray_tmp}"
RAY_TMP_LINK="${RAY_TMP_LINK:-/tmp/rayds}"

if [[ "${WORKER_INDEX}" -lt 1 ]]; then
  echo "WORKER_INDEX must be >= 1, got ${WORKER_INDEX}" >&2
  exit 2
fi
if [[ "${WORKER_INDEX}" -gt "${WORKER_TOTAL}" ]]; then
  echo "WORKER_INDEX must be <= WORKER_TOTAL, got ${WORKER_INDEX}/${WORKER_TOTAL}" >&2
  exit 2
fi

mkdir -p "${RAY_TMP_ROOT_REAL}"
ln -sfn "${RAY_TMP_ROOT_REAL}" "${RAY_TMP_LINK}"
mkdir -p "${EVAL_DIR}/tools/logs"

TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${EVAL_DIR}/tools/logs/eval_${GROUP}${WORKER_INDEX}_${DATASET}_${TS}.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"
}

declare -a ALL_EXPS=()
case "${GROUP}" in
  A)
    ALL_EXPS=(
      "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl"
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl"
      "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl"
      "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl"
      "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl"
    )
    ;;
  B)
    ALL_EXPS=(
      "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8"
      "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12"
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
      "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
      "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
      "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
      "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
      "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
      "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
    )
    ;;
  C)
    ALL_EXPS=(
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08"
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12"
      "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08"
      "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12"
      "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08"
      "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12"
      "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08"
      "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12"
    )
    ;;
  D)
    ALL_EXPS=(
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
      "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
      "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
      "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    )
    ;;
  E)
    # New Pure-GRPO variants that have never been evaluated.
    ALL_EXPS=(
      "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-clip_ratio_c10-ppo_epoch3"
      "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-token-separate-c3-e3"
    )
    ;;
  F)
    # Partial completions: experiments with some missing checkpoint evals.
    # TrajectoryEntropy-n8 step 105 v5 parquet is missing (empty dir).
    # Intuitor-n8/n12/kl0005 step 80/105 ckpts not yet available — will be
    # skipped automatically and can be re-run once training finishes.
    ALL_EXPS=(
      "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
      "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    )
    ;;
  *)
    echo "GROUP must be one of A/B/C/D, got ${GROUP}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a STEPS <<< "${STEPS_CSV}"

declare -a MY_EXPS=()
idx=0
for exp in "${ALL_EXPS[@]}"; do
  if (( (idx % WORKER_TOTAL) == (WORKER_INDEX - 1) )); then
    MY_EXPS+=("${exp}")
  fi
  idx=$((idx + 1))
done

if [[ ${#MY_EXPS[@]} -eq 0 ]]; then
  log "No experiments assigned for ${GROUP}${WORKER_INDEX} (total workers=${WORKER_TOTAL}). Exit."
  exit 0
fi

if [[ ! -f "${DATA_DIR}/${DATASET}.parquet" && ! -f "${DATA_DIR}/${DATASET}.json" ]]; then
  log "Dataset not found: ${DATA_DIR}/${DATASET}.parquet|json"
  exit 2
fi

cd "${EVAL_DIR}"

check_ckpt() {
  local exp="$1"
  local step="$2"
  [[ -d "${OUTPUT_ROOT}/${exp}/global_step_${step}/actor" ]]
}

check_hf() {
  local exp="$1"
  local step="$2"
  [[ -d "${OUTPUT_ROOT}/${exp}/global_step_${step}/actor/hf_model" ]]
}

merge_if_needed() {
  local exp="$1"
  local step="$2"
  local actor_dir="${OUTPUT_ROOT}/${exp}/global_step_${step}/actor"
  local hf_dir="${actor_dir}/hf_model"
  if [[ -d "${hf_dir}" ]]; then
    return 0
  fi
  if [[ "${MERGE_IF_NEEDED}" != "1" ]]; then
    log "hf_model missing (MERGE_IF_NEEDED=0), skip merge: ${exp} step ${step}"
    return 1
  fi
  log "Merging ckpt -> hf_model: ${exp} step ${step}"
  if (cd "${EVAL_DIR}" && "${PYTHON_BIN}" -m tools.model_merge merge --backend fsdp --local_dir "${actor_dir}" --target_dir "${hf_dir}") >> "${MAIN_LOG}" 2>&1; then
    [[ -d "${hf_dir}" ]] && return 0
  fi
  log "Merge failed: ${exp} step ${step}"
  return 2
}

run_one_eval() {
  local exp="$1"
  local step="$2"
  local hf_dir="${OUTPUT_ROOT}/${exp}/global_step_${step}/actor/hf_model"
  local out_dir="${hf_dir}/${OUTPUT_SUBDIR}"
  local out_path="${out_dir}/${DATASET}.parquet"
  local ray_tmp_dir="${RAY_TMP_LINK}/${GROUP}${WORKER_INDEX}/s${step}"

  mkdir -p "${out_dir}" "${ray_tmp_dir}"
  export RAY_TMPDIR="${ray_tmp_dir}"

  if [[ -f "${out_path}" ]]; then
    log "Output exists, skip: ${out_path}"
    return 0
  fi

  log "Start eval: ${exp} step=${step} dataset=${DATASET} n=${N_SAMPLES} gpus=${N_GPUS}"
  log "Ray tmp: ${RAY_TMPDIR} -> ${RAY_TMP_ROOT_REAL}"

  # Keep compatibility with legacy batch config:
  # - LCB: prompt 4096
  # - AIME/non-LCB: prompt 2048
  local prompt_len="${MAX_PROMPT_LEN}"
  if [[ -z "${prompt_len}" ]]; then
    if [[ "${DATASET}" == livecodebench* ]]; then
      prompt_len=4096
    else
      prompt_len=2048
    fi
  fi

  "${PYTHON_BIN}" -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    +trainer.project_name=ArcherCodeR_Eval \
    +trainer.experiment_name="${exp}_s${step}_${DATASET}" \
    +trainer.task_name="${DATASET}" \
    +trainer.global_step="${step}" \
    +trainer.use_wandb=False \
    data.path="${DATA_DIR}/${DATASET}.parquet" \
    data.prompt_key=prompt \
    data.batch_size="${BATCH_SIZE}" \
    data.n_samples="${N_SAMPLES}" \
    data.output_path="${out_path}" \
    model.path="${hf_dir}" \
    rollout.name=vllm \
    rollout.temperature="${TEMPERATURE}" \
    rollout.top_k="${TOP_K}" \
    rollout.top_p="${TOP_P}" \
    rollout.tensor_model_parallel_size="${TP_SIZE}" \
    rollout.gpu_memory_utilization=0.9 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    rollout.disable_log_stats=False \
    rollout.prompt_length="${prompt_len}" \
    rollout.response_length="${MAX_RESPONSE_LEN}" \
    rollout.max_num_batched_tokens=$((prompt_len + MAX_RESPONSE_LEN)) \
    >> "${MAIN_LOG}" 2>&1
}

ok=0
fail=0
skip=0

log "Assigned experiments (${#MY_EXPS[@]}): ${MY_EXPS[*]}"
log "Steps: ${STEPS[*]}"

for exp in "${MY_EXPS[@]}"; do
  for step in "${STEPS[@]}"; do
    if ! check_ckpt "${exp}" "${step}"; then
      log "Checkpoint missing, skip: ${exp} step ${step}"
      skip=$((skip + 1))
      continue
    fi
    merge_if_needed "${exp}" "${step}"
    merge_rc=$?
    if [[ ${merge_rc} -eq 1 ]]; then
      skip=$((skip + 1))
      continue
    elif [[ ${merge_rc} -ne 0 ]]; then
      fail=$((fail + 1))
      continue
    fi
    if run_one_eval "${exp}" "${step}"; then
      ok=$((ok + 1))
    else
      log "Eval failed: ${exp} step ${step}"
      fail=$((fail + 1))
    fi
  done
done

log "Done. ok=${ok}, fail=${fail}, skip=${skip}. log=${MAIN_LOG}"
exit $(( fail > 0 ? 1 : 0 ))
