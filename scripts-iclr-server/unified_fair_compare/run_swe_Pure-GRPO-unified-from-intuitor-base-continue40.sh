#!/usr/bin/env bash
set -euo pipefail

# Continue Pure-GRPO from Intuitor base checkpoints for +40 steps.
# Base Intuitor experiment:
#   ./output/self-rl-jxl/Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl

INTUITOR_BASE_DIR="${INTUITOR_BASE_DIR:-./output/self-rl-jxl/Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl}"
if [ -n "${START_STEPS_OVERRIDE:-}" ]; then
  read -r -a START_STEPS <<< "${START_STEPS_OVERRIDE}"
else
  START_STEPS=(50 80 105)
fi

for start_step in "${START_STEPS[@]}"; do
  target_step=$((start_step + 40))
  resume_ckpt="${INTUITOR_BASE_DIR}/global_step_${start_step}"
  exp_suffix="from-intuitor-base-step${start_step}-plus40-to${target_step}"

  echo "============================================================"
  echo "🚀 Launch Pure-GRPO continuation from Intuitor step ${start_step}"
  echo "   Resume checkpoint: ${resume_ckpt}"
  echo "   Target total_training_steps: ${target_step}"
  echo "   EXP_SUFFIX: ${exp_suffix}"
  echo "============================================================"

  if [ ! -d "${resume_ckpt}" ]; then
    echo "❌ Checkpoint not found: ${resume_ckpt}"
    echo "   You can override base dir via:"
    echo "   INTUITOR_BASE_DIR=/your/path bash $0"
    exit 1
  fi

  EXP_SUFFIX="${exp_suffix}" \
  bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${resume_ckpt}" \
    actor_rollout_ref.model.path="${resume_ckpt}/actor/hf_model" \
    trainer.total_training_steps="${target_step}" \
    "$@"
done
