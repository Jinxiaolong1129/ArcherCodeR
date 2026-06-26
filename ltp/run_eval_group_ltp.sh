#!/usr/bin/env bash
# ============================================================
# Group eval launcher: one 8-GPU job evaluates a whole GROUP of models, serially looping
#   model -> ckpt {20,50,80,105} -> benchmark {v5,v6,aime24,aime25} -> eval-length.
# Each (ckpt,dataset,len) is run via run_eval_ltp.sh (merge cached per ckpt; resume via .done marker).
# GROUP env selects the model list + eval length(s):
#   7b-8k | 7b-16k | 1.5b | lp | kl
# Failures of one eval do NOT abort the group (logged as WARN).
# ============================================================
set -uo pipefail

CEPHFS=${CEPHFS:-/mnt/cephfs/data/processing/xiaolong.jin}
REPO_DIR=${REPO_DIR:-$CEPHFS/code/ArcherCodeR}
OUT="$REPO_DIR/output"
GROUP=${GROUP:?need GROUP=7b-8k|7b-16k|1.5b|lp|kl}
cd "$REPO_DIR"

SEVENB="$OUT/swe-rl/swe-rl-grpo $OUT/swe-rl/swe-rl-intuitor $OUT/swe-rl/swe-rl-token_entropy $OUT/swe-rl/swe-rl-trajectory_entropy $OUT/swe-rl/swe-rl-prob_disparity"
case "$GROUP" in
  7b-8k)  MODELS="$SEVENB"; LENS="8192" ;;
  7b-16k) MODELS="$SEVENB"; LENS="16384" ;;
  1.5b)   MODELS="$OUT/r1-1.5b-16k/grpo $OUT/r1-1.5b-16k/in $OUT/r1-1.5b-16k/tok $OUT/r1-1.5b-16k/traj $OUT/r1-1.5b-16k/prob"; LENS="16384" ;;
  lp)     MODELS="$OUT/r1-1.5b-16k-lp/in-lp $OUT/r1-1.5b-16k-lp/tok-lp $OUT/r1-1.5b-16k-lp/traj-lp $OUT/r1-1.5b-16k-lp/prob-lp"; LENS="16384" ;;
  kl)     MODELS="$OUT/r1-1.5b-16k-kl/in-kl $OUT/r1-1.5b-16k-kl/tok-kl $OUT/r1-1.5b-16k-kl/traj-kl $OUT/r1-1.5b-16k-kl/prob-kl"; LENS="16384" ;;
  4b)     MODELS="$OUT/qwen3-4b-16k/grpo $OUT/qwen3-4b-16k/in $OUT/qwen3-4b-16k/tok $OUT/qwen3-4b-16k/traj $OUT/qwen3-4b-16k/prob"; LENS="16384" ;;
  # --- new (8K-trained 7B KL + 7B/1.5B random): 8K-trained -> eval 8K+16K; 16K-trained -> 16K only ---
  7b-new-8k)  MODELS="$OUT/swe-rl-kl/swe-rl-intuitor-kl $OUT/swe-rl-kl/swe-rl-token_entropy-kl $OUT/swe-rl-kl/swe-rl-trajectory_entropy-kl $OUT/swe-rl-kl/swe-rl-prob_disparity-kl $OUT/swe-rl-random/grpo-random"; LENS="8192" ;;
  7b-new-16k) MODELS="$OUT/swe-rl-kl/swe-rl-intuitor-kl $OUT/swe-rl-kl/swe-rl-token_entropy-kl $OUT/swe-rl-kl/swe-rl-trajectory_entropy-kl $OUT/swe-rl-kl/swe-rl-prob_disparity-kl $OUT/swe-rl-random/grpo-random"; LENS="16384" ;;
  1.5b-8krand-8k)  MODELS="$OUT/r1-1.5b-8k-random/grpo-random"; LENS="8192" ;;
  1.5b-8krand-16k) MODELS="$OUT/r1-1.5b-8k-random/grpo-random"; LENS="16384" ;;
  1.5b-16krand)    MODELS="$OUT/r1-1.5b-16k-random/grpo-random"; LENS="16384" ;;
  # --- step0 baselines: eval the raw pretrained model (no training); both eval lengths ---
  step0-1.5b)      RAW_MODEL="$CEPHFS/models/DeepSeek-R1-Distill-Qwen-1.5B"; STEP0_NAME="r1-1.5b-base"; LENS="${STEP0_LENS:-8192 16384}"; RAW=1 ;;
  step0-7b)        RAW_MODEL="$CEPHFS/models/Qwen2.5-Coder-7B-Instruct"; STEP0_NAME="coder-7b-base"; LENS="${STEP0_LENS:-8192 16384}"; RAW=1 ;;
  *) echo "unknown GROUP=$GROUP"; exit 1 ;;
esac

# optional: restrict to a single model (by basename) for finer per-model parallel jobs
if [ -n "${MODEL_FILTER:-}" ]; then
  FILTERED=""
  for m in $MODELS; do [ "$(basename "$m")" = "$MODEL_FILTER" ] && FILTERED="$FILTERED $m"; done
  MODELS="$FILTERED"
  echo "## MODEL_FILTER=$MODEL_FILTER -> $(echo $MODELS | wc -w) model(s)"
fi

BENCHES="${BENCHES:-livecodebench_v5:v5:8 livecodebench_v6:v6:8 aime2024:v5:32 aime2025:v5:32}"

# --- step0 path: raw pretrained model, single "ckpt"=0, loops lengths x benches ---
if [ "${RAW:-0}" = 1 ]; then
  echo "######## STEP0=$STEP0_NAME | lens=$LENS ########"
  for len in $LENS; do
    for b in $BENCHES; do
      ds="${b%%:*}"; rest="${b#*:}"; ver="${rest%%:*}"; n="${rest##*:}"
      echo ">>> EVAL[${MODE:-gen}] STEP0 $STEP0_NAME bench=$ds len=$len n=$n  [$(date +%H:%M:%S)]"
      RAW_MODEL_DIR="$RAW_MODEL" STEP0_NAME="$STEP0_NAME" DATASET="$ds" LCB_VERSION="$ver" EVAL_RESP_LEN="$len" N_SAMPLES="$n" MODE="${MODE:-gen}" \
        bash "$REPO_DIR/ltp/run_eval_ltp.sh" || echo "WARN: step0 eval FAILED $STEP0_NAME $ds len=$len"
    done
  done
  echo "######## STEP0 $STEP0_NAME DONE ########"
  exit 0
fi

STEPS="20 50 80 105"
# benchmark spec: "dataset:lcb_version:n_samples" (env-overridable, e.g. score AIME only)
BENCHES="${BENCHES:-livecodebench_v5:v5:8 livecodebench_v6:v6:8 aime2024:v5:32 aime2025:v5:32}"

echo "######## GROUP=$GROUP | lens=$LENS | $(echo $MODELS | wc -w) models ########"
t_group=$(date +%s)
for md in $MODELS; do
  for st in $STEPS; do
    ckpt="$md/global_step_$st"
    if [ ! -d "$ckpt/actor" ]; then echo "-- skip missing $ckpt"; continue; fi
    for len in $LENS; do
      for b in $BENCHES; do
        ds="${b%%:*}"; rest="${b#*:}"; ver="${rest%%:*}"; n="${rest##*:}"
        echo ">>> EVAL[${MODE:-gen}] $(basename $md) step=$st bench=$ds len=$len n=$n  [$(date +%H:%M:%S)]"
        CKPT_DIR="$ckpt" DATASET="$ds" LCB_VERSION="$ver" EVAL_RESP_LEN="$len" N_SAMPLES="$n" MODE="${MODE:-gen}" \
          bash "$REPO_DIR/ltp/run_eval_ltp.sh" || echo "WARN: eval FAILED $(basename $md) step=$st $ds len=$len"
      done
    done
  done
done
echo "######## GROUP $GROUP DONE in $(( $(date +%s) - t_group ))s ########"
