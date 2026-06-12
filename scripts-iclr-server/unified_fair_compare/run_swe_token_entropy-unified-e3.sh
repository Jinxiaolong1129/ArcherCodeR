#!/usr/bin/env bash
set -euo pipefail

# Wrapper for unified Token Entropy:
# force PPO epochs to 3 while reusing the base script.
EXP_SUFFIX="${EXP_SUFFIX:-ppo_epoch3}" \
bash scripts-iclr-server/unified_fair_compare/run_swe_token_entropy-unified.sh \
  actor_rollout_ref.actor.ppo_epochs=3 \
  "$@"
