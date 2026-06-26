#!/usr/bin/env bash
# ============================================================
# LTP eval launcher: merge verl ckpt -> HF, generate on a benchmark (verl main_generation),
# then score with LiveCodeBench lcb_runner. Runs INSIDE the GPU container (1 node x 8 GPU).
# Faithful port of Archer_eval/tools/batch_merge_eval_lcb_v6_16k_part1.sh + run_lcb_eval_v5.sh.
#
# Required env:
#   CKPT_DIR        full path to a global_step_<N> dir (must contain actor/)
#   DATASET         basename of the json under data/test, e.g. livecodebench_v5 | livecodebench_v6
#   LCB_VERSION     v5 | v6  (for the scorer)
# Optional env:
#   EVAL_RESP_LEN   response length (default 16384). 8K models -> 8192.
#   N_SAMPLES       pass@k samples (default 8 for LCB)
#   TEMP            sampling temperature (default 0.6)
# ============================================================
set -xeuo pipefail

CEPHFS=${CEPHFS:-/mnt/cephfs/data/processing/xiaolong.jin}
REPO_DIR=${REPO_DIR:-$CEPHFS/code/ArcherCodeR}
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/Archer_eval/LiveCodeBench:${PYTHONPATH:-}"

CKPT_DIR=${CKPT_DIR:-}   # optional: omitted for step0 (RAW_MODEL_DIR set instead)
DATASET=${DATASET:-livecodebench_v5}
LCB_VERSION=${LCB_VERSION:-v5}
EVAL_RESP_LEN=${EVAL_RESP_LEN:-16384}
EVAL_PROMPT_LEN=${EVAL_PROMPT_LEN:-4096}
N_SAMPLES=${N_SAMPLES:-8}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
N_GPUS=${N_GPUS:-8}
TP=${TP:-1}
BATCH_SIZE=${BATCH_SIZE:-2048}
NUM_WORKERS=${NUM_WORKERS:-160}

RAW_MODEL_DIR=${RAW_MODEL_DIR:-}
if [ -n "$RAW_MODEL_DIR" ]; then
  # step0 baseline: eval the raw pretrained HF model directly (no actor, no merge needed)
  HF_DIR="$RAW_MODEL_DIR"
  STEP=0
  EXP="${STEP0_NAME:-base}"
  OUT_DIR="$REPO_DIR/output/step0/${EXP}/eval_out_${EVAL_RESP_LEN}"
else
  ACTOR_DIR="$CKPT_DIR/actor"
  HF_DIR="$ACTOR_DIR/hf_model"
  OUT_DIR="$HF_DIR/eval_out_${EVAL_RESP_LEN}"
  STEP=$(basename "$CKPT_DIR" | grep -oE "[0-9]+$" || echo 0)
  EXP=$(basename "$(dirname "$CKPT_DIR")")
fi
mkdir -p "$OUT_DIR"

MODE=${MODE:-both}   # gen | score | both
DONE_MARK="$OUT_DIR/${DATASET}.eval.done"
PARQUET="$OUT_DIR/${DATASET}.parquet"
# resume
if [ "$MODE" != "gen" ] && [ -f "$DONE_MARK" ]; then echo "✅ scored, skip: $EXP s$STEP $DATASET @${EVAL_RESP_LEN}"; exit 0; fi
if [ "$MODE" = "gen" ] && [ -f "$PARQUET" ]; then echo "✅ generated, skip: $EXP s$STEP $DATASET @${EVAL_RESP_LEN}"; exit 0; fi

echo "🔎 EVAL[$MODE] | exp=$EXP step=$STEP | dataset=$DATASET | resp_len=$EVAL_RESP_LEN n=$N_SAMPLES"

if [ "$MODE" != "score" ]; then
  # ===== merge verl FSDP shards -> HF (skip if already merged) =====
  if [ ! -f "$HF_DIR/config.json" ]; then
    echo "=== merging $ACTOR_DIR -> $HF_DIR ==="
    t0=$(date +%s)
    python -m tools.model_merge merge --backend fsdp --local_dir "$ACTOR_DIR" --target_dir "$HF_DIR"
    echo "merge took $(( $(date +%s) - t0 ))s"
  else
    echo "=== HF model exists, skip merge ==="
  fi
  # ===== generate (skip if parquet exists) =====
  if [ -f "$PARQUET" ]; then
    echo "=== parquet exists, skip generation ==="
  else
    echo "=== generation start $(date) ==="
    g0=$(date +%s)
    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${N_GPUS} \
        +trainer.project_name=ArcherCodeR_Eval \
        +trainer.experiment_name=${EXP} \
        +trainer.task_name=${DATASET} \
        +trainer.global_step=${STEP} \
        +trainer.use_wandb=False \
        model.path=${HF_DIR} \
        data.path=${REPO_DIR}/data/test/${DATASET}.json \
        data.output_path=${PARQUET} \
        data.batch_size=${BATCH_SIZE} \
        data.n_samples=${N_SAMPLES} \
        rollout.name=vllm \
        rollout.gpu_memory_utilization=0.9 \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.tensor_model_parallel_size=${TP} \
        rollout.temperature=${TEMP} \
        rollout.top_k=-1 \
        rollout.top_p=${TOP_P} \
        rollout.prompt_length=${EVAL_PROMPT_LEN} \
        rollout.response_length=${EVAL_RESP_LEN} \
        rollout.max_num_batched_tokens=$((EVAL_PROMPT_LEN + EVAL_RESP_LEN))
    echo "=== generation took $(( $(date +%s) - g0 ))s ==="
  fi
fi

if [ "$MODE" != "gen" ]; then
  # ===== score =====
  echo "=== scoring start $(date) ==="
  s0=$(date +%s)
  case "$DATASET" in
    livecodebench*)
      python Archer_eval/LiveCodeBench/lcb_runner/evaluation/compute_code_generation_metrics_local.py \
          --eval_file "${PARQUET}" \
          --testcase_file "${REPO_DIR}/data/test/${DATASET}.json" \
          --project_name ArcherEval --experiment_name "${EXP}" --global_step "${STEP}" \
          --num_workers ${NUM_WORKERS}
      ;;
    aime*)
      python tools/compute_math_metrics_local.py \
          --eval_file "${PARQUET}" \
          --testcase_file "${REPO_DIR}/data/test/${DATASET}.json"
      ;;
    *) echo "=== unknown dataset ($DATASET): scoring skipped ===" ;;
  esac
  echo "=== scoring took $(( $(date +%s) - s0 ))s ==="
  touch "$DONE_MARK"
fi
echo "FINISHED[$MODE] eval $EXP step=$STEP dataset=$DATASET"
