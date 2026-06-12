#!/usr/bin/env bash
set -euo pipefail

# Wrapper for unified Pure-GRPO:
# force dual-clip coefficient and PPO epochs while reusing the base script.
EXP_SUFFIX="${EXP_SUFFIX:-clip_ratio_c10-ppo_epoch3}" \
CLIP_RATIO_C_OVERRIDE=10.0 \
PPO_EPOCHS_OVERRIDE=3 \
bash scripts-iclr-server/unified_fair_compare/run_swe_Pure-GRPO-unified.sh "$@"
