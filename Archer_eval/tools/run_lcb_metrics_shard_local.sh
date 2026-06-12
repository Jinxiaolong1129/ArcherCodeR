#!/usr/bin/env bash
set -euo pipefail

# Run local LCB metrics on a deterministic shard of tasks.
# - Scans output/self-rl-jxl for v5/v6 parquet outputs
# - Skips tasks that already have .pass.lcb.csv
# - Splits remaining tasks by SHARD_ID / NUM_SHARDS
# - Runs tasks sequentially on current node

PROJ_DIR="${PROJ_DIR:-/data_storage/wyj/jxl/ArcherCodeR}"
EVAL_DIR="${EVAL_DIR:-${PROJ_DIR}/Archer_eval}"
LCB_DIR="${LCB_DIR:-${EVAL_DIR}/LiveCodeBench}"
MODEL_ROOT="${MODEL_ROOT:-${PROJ_DIR}/output/self-rl-jxl}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SHARD_ID="${SHARD_ID:-0}"        # 0-based shard index
NUM_SHARDS="${NUM_SHARDS:-2}"    # total shards
NUM_WORKERS="${NUM_WORKERS:-96}" # cpu workers for metric script
TASK_LIMIT="${TASK_LIMIT:-0}"    # 0 means no limit
DRY_RUN="${DRY_RUN:-0}"          # 1 means only print selected tasks
USE_HF_CACHE="${USE_HF_CACHE:-0}" # 1 means use HF cache benchmark loader

PROJECT_NAME="${PROJECT_NAME:-ArcherCodeR_Eval}"
BENCHMARK_V5_JSON="${BENCHMARK_V5_JSON:-${EVAL_DIR}/data/test/livecodebench_v5.json}"
BENCHMARK_V6_JSON="${BENCHMARK_V6_JSON:-${EVAL_DIR}/data/test/livecodebench_v6.json}"

LOG_DIR="${LOG_DIR:-${EVAL_DIR}/tools/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/lcb_metrics_shard${SHARD_ID}of${NUM_SHARDS}_${TS}.log"
mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${RUN_LOG}"
}

if [[ ! -d "${MODEL_ROOT}" ]]; then
  log "MODEL_ROOT not found: ${MODEL_ROOT}"
  exit 1
fi

if [[ "${USE_HF_CACHE}" != "1" ]]; then
  if [[ ! -f "${BENCHMARK_V5_JSON}" ]]; then
    log "Missing v5 benchmark json: ${BENCHMARK_V5_JSON}"
    exit 1
  fi
  if [[ ! -f "${BENCHMARK_V6_JSON}" ]]; then
    log "Missing v6 benchmark json: ${BENCHMARK_V6_JSON}"
    exit 1
  fi
fi

if [[ "${SHARD_ID}" -lt 0 || "${SHARD_ID}" -ge "${NUM_SHARDS}" ]]; then
  log "Invalid shard config: SHARD_ID=${SHARD_ID}, NUM_SHARDS=${NUM_SHARDS}"
  exit 1
fi

TASK_FILE="$(mktemp)"
SELECTED_FILE="$(mktemp)"
cleanup() {
  rm -f "${TASK_FILE}" "${SELECTED_FILE}"
}
trap cleanup EXIT

log "Scanning parquet tasks under ${MODEL_ROOT}"
log "Shard: ${SHARD_ID}/${NUM_SHARDS}, workers=${NUM_WORKERS}, dry_run=${DRY_RUN}, use_hf_cache=${USE_HF_CACHE}"

export MODEL_ROOT
"${PYTHON_BIN}" - <<'PY' > "${TASK_FILE}"
import os
import glob

model_root = os.environ["MODEL_ROOT"]

patterns = [
    "**/output_livecodebench_v5_8k_n8/livecodebench_v5.parquet",
    "**/output_livecodebench_v6_8k_n8/livecodebench_v6.parquet",
    "**/output_16k/livecodebench_v5_n8/livecodebench_v5.parquet",
    "**/output_16k/livecodebench_v6_n8/livecodebench_v6.parquet",
    "**/output_v6_8k/livecodebench_v6.parquet",
    "**/output_v6_16k/livecodebench_v6.parquet",
    "**/output/livecodebench_v5.parquet",
    "**/output/livecodebench_v6.parquet",
]

files = set()
for p in patterns:
    files.update(glob.glob(os.path.join(model_root, p), recursive=True))

pending = []
for f in sorted(files):
    csv = f + ".pass.lcb.csv"
    if not os.path.isfile(csv):
        pending.append(f)

for f in pending:
    print(f)
PY

TOTAL_PENDING="$(wc -l < "${TASK_FILE}")"
log "Total pending tasks found: ${TOTAL_PENDING}"

if [[ "${TOTAL_PENDING}" -eq 0 ]]; then
  log "No pending task. Exit."
  exit 0
fi

awk -v sid="${SHARD_ID}" -v nsh="${NUM_SHARDS}" '((NR-1) % nsh) == sid {print $0}' "${TASK_FILE}" > "${SELECTED_FILE}"
TOTAL_SELECTED="$(wc -l < "${SELECTED_FILE}")"
log "Tasks selected for this shard: ${TOTAL_SELECTED}"

if [[ "${TOTAL_SELECTED}" -eq 0 ]]; then
  log "This shard has no task to run. Exit."
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1, selected tasks:"
  nl -ba "${SELECTED_FILE}" | tee -a "${RUN_LOG}"
  exit 0
fi

cd "${LCB_DIR}"
export PYTHONPATH="${EVAL_DIR}:${LCB_DIR}:${PYTHONPATH:-}"

ok=0
fail=0
idx=0

while IFS= read -r parquet; do
  idx=$((idx + 1))
  if [[ "${TASK_LIMIT}" -gt 0 && "${idx}" -gt "${TASK_LIMIT}" ]]; then
    log "Hit TASK_LIMIT=${TASK_LIMIT}, stop."
    break
  fi

  if [[ "${parquet}" == *"livecodebench_v6.parquet" ]]; then
    lcb_version="v6"
    benchmark_json="${BENCHMARK_V6_JSON}"
    dataset_tag="lcbv6"
  else
    lcb_version="v5"
    benchmark_json="${BENCHMARK_V5_JSON}"
    dataset_tag="lcbv5"
  fi

  exp_name="$(echo "${parquet}" | sed -n 's#.*/output/self-rl-jxl/\([^/]*\)/global_step_.*#\1#p')"
  step="$(echo "${parquet}" | sed -n 's#.*global_step_\([0-9][0-9]*\).*#\1#p')"
  if [[ -z "${exp_name}" ]]; then
    exp_name="unknown_exp"
  fi
  if [[ -z "${step}" ]]; then
    step="0"
  fi
  wandb_exp="${exp_name}_${dataset_tag}_s${step}_metric_local"

  log "[$idx/${TOTAL_SELECTED}] Start ${dataset_tag} step=${step} exp=${exp_name}"
  log "Eval file: ${parquet}"

  if [[ "${USE_HF_CACHE}" == "1" ]]; then
    if "${PYTHON_BIN}" -m lcb_runner.evaluation.compute_code_generation_metrics_online \
      --eval_file "${parquet}" \
      --project_name "${PROJECT_NAME}" \
      --experiment_name "${wandb_exp}" \
      --global_step "${step}" \
      --lcb_version "${lcb_version}" \
      --num_workers "${NUM_WORKERS}" >> "${RUN_LOG}" 2>&1 < /dev/null; then
      ok=$((ok + 1))
      log "Done: ${parquet}.pass.lcb.csv"
    else
      fail=$((fail + 1))
      log "FAILED: ${parquet}"
    fi
  elif "${PYTHON_BIN}" -m lcb_runner.evaluation.compute_code_generation_metrics_online \
    --eval_file "${parquet}" \
    --project_name "${PROJECT_NAME}" \
    --experiment_name "${wandb_exp}" \
    --global_step "${step}" \
    --lcb_version "${lcb_version}" \
    --benchmark_json "${benchmark_json}" \
    --num_workers "${NUM_WORKERS}" >> "${RUN_LOG}" 2>&1 < /dev/null; then
    ok=$((ok + 1))
    log "Done: ${parquet}.pass.lcb.csv"
  else
    fail=$((fail + 1))
    log "FAILED: ${parquet}"
  fi
done < "${SELECTED_FILE}"

log "Finished shard ${SHARD_ID}/${NUM_SHARDS}: ok=${ok}, fail=${fail}, selected=${TOTAL_SELECTED}"
