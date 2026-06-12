#!/usr/bin/env bash
set -euo pipefail

# Wrapper for unified Pure-GRPO:
# - enable token-entropy-separated per-token clipping
# - set PPO epochs to 3
# - keep clip_ratio_c at 3
EXP_SUFFIX="${EXP_SUFFIX:-token-separate-c3-e3}" \
CLIP_RATIO_C_OVERRIDE="${CLIP_RATIO_C_OVERRIDE:-3.0}" \
PPO_EPOCHS_OVERRIDE="${PPO_EPOCHS_OVERRIDE:-3}" \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh \
  +actor_rollout_ref.actor.use_token_entropy_separate=True \
  "$@"
