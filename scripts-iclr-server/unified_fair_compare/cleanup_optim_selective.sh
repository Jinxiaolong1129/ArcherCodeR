#!/usr/bin/env bash
# =============================================================================
# cleanup_optim_selective.sh
#
# Deletes ONLY the FSDP shard + optimizer files from specified steps:
#   actor/model_world_size_*.pt
#   actor/optim_world_size_*.pt
#   actor/extra_state_world_size_*.pt
#
# Preserved (never touched):
#   actor/hf_model/        ← merged model weights
#   actor/config.json
#   actor/tokenizer*.json
#   actor/generation_config.json
#   actor/special_tokens_map.json
#   eval/                  ← evaluation results
#
# Part 1 – continue-40 runs: delete shards+optim from all steps EXCEPT last
#   - from-*-step105→145: keep 145, delete 110,120,130,140
#   - from-*-step50→90:   keep 90,  delete 60,70,80
#
# Part 2 – n/kl/temp ablation runs: delete shards+optim from step_20 only
#
# Part 3 – ppo_epoch3 runs: keep 10,50,80,105; delete shards+optim from rest
#
# Part 4 – long baseline runs: delete shards+optim from specified steps
#   TrajectoryEntropy-no-kl: 110,120,130,140
#   TokenEntropy-no-kl:      110,120,130,140
#   Intuitor-no-kl:          110,120,130,140,160,170,180,190
#
# Usage:
#   bash cleanup_optim_selective.sh          # DRY RUN (default)
#   bash cleanup_optim_selective.sh --execute
# =============================================================================

set -uo pipefail

EXECUTE=false
[[ "${1:-}" == "--execute" ]] && EXECUTE=true

BASE="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'
info() { echo -e "${BLU}[INFO]${RST}  $*"; }
ok()   { echo -e "${GRN}[SKIP]${RST}  $*"; }
warn() { echo -e "${YLW}[WARN]${RST}  $*"; }
act()  { echo -e "${RED}[DEL]${RST}   $*"; }
dry()  { echo -e "${CYN}[DRY]${RST}   $*"; }

total_kb=0

# ── helper: delete only *_world_size_*.pt files (shards + optim + extra_state)
#            hf_model/, config.json, tokenizer files etc. are UNTOUCHED
del_shards_and_optim() {
    local run="$1" step="$2"
    local actor_dir="$BASE/$run/global_step_${step}/actor"

    if [[ ! -d "$actor_dir" ]]; then
        ok "not found: $run / step_$step"
        return
    fi

    # collect all three categories of shard files
    local files
    mapfile -t files < <(ls \
        "$actor_dir"/model_world_size_*.pt \
        "$actor_dir"/optim_world_size_*.pt \
        "$actor_dir"/extra_state_world_size_*.pt \
        2>/dev/null || true)

    if [[ ${#files[@]} -eq 0 ]]; then
        ok "no shards/optim: $run / step_$step"
        return
    fi

    local kb=0
    for f in "${files[@]}"; do
        kb=$((kb + $(du -sk "$f" 2>/dev/null | cut -f1)))
    done
    total_kb=$((total_kb + kb))
    local gb; gb=$(awk "BEGIN{printf \"%.1f\", $kb/1024/1024}")
    local n=${#files[@]}

    if $EXECUTE; then
        act "DELETE shards+optim ($n files, ~${gb}G)  $run / step_$step"
        rm -f "${files[@]}"
    else
        dry "WOULD DELETE shards+optim ($n files, ~${gb}G)  $run / step_$step"
    fi
}

# ==============================================================================
if ! $EXECUTE; then
    warn "DRY RUN – pass --execute to actually delete"
    echo
fi

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: continue-40 runs
# ─────────────────────────────────────────────────────────────────────────────
echo
info "════ PART 1: continue-40 runs (keep only last step's shards+optim) ════"

# step105→145: delete 110,120,130,140  (keep 145)
for src in intuitor tokenentropy probdisparity trajentropy; do
    run="Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-${src}-base-step105-plus40-to145"
    for step in 110 120 130 140; do
        del_shards_and_optim "$run" "$step"
    done
done

# step50→90: delete 60,70,80  (keep 90)
for src in intuitor tokenentropy probdisparity trajentropy; do
    run="Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-${src}-base-step50-plus40-to90"
    for step in 60 70 80; do
        del_shards_and_optim "$run" "$step"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: ablation runs – delete shards+optim from step_20 only
# ─────────────────────────────────────────────────────────────────────────────
echo
info "════ PART 2: ablation runs (delete shards+optim step_20 only) ════"

STEP20_RUNS=(
    "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n8"
    "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
    "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
    "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
    "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
    "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
    "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
    "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08"
    "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
    "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-kl0005"
)

for run in "${STEP20_RUNS[@]}"; do
    del_shards_and_optim "$run" "20"
done

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: ppo_epoch3 runs – keep 10,50,80,105; delete shards+optim from rest
# ─────────────────────────────────────────────────────────────────────────────
echo
info "════ PART 3: ppo_epoch3 runs (delete shards+optim for steps 20,30,40,60,70,90,100) ════"

PPO3_RUNS=(
    "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-ppo_epoch3"
    "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-ppo_epoch3"
    "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-ppo_epoch3"
)

for run in "${PPO3_RUNS[@]}"; do
    for step in 20 30 40 60 70 90 100; do
        del_shards_and_optim "$run" "$step"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: long baseline runs
# ─────────────────────────────────────────────────────────────────────────────
echo
info "════ PART 4: long baseline runs (delete shards+optim from specified steps) ════"

echo
info "  TrajectoryEntropy-no-kl: steps 110,120,130,140"
for step in 110 120 130 140; do
    del_shards_and_optim "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl" "$step"
done

echo
info "  TokenEntropy-no-kl: steps 110,120,130,140"
for step in 110 120 130 140; do
    del_shards_and_optim "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl" "$step"
done

echo
info "  Intuitor-no-kl: steps 110,120,130,140,160,170,180,190"
for step in 110 120 130 140 160 170 180 190; do
    del_shards_and_optim "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl" "$step"
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo
freed_gb=$(awk "BEGIN{printf \"%.1f\", $total_kb/1024/1024}")
info "════════════════════════════════════════════════════════════"
if $EXECUTE; then
    info "Done. Freed: ~${freed_gb}G"
else
    info "DRY RUN complete. Would free: ~${freed_gb}G"
    warn "Run with --execute to apply changes"
fi
info "════════════════════════════════════════════════════════════"
