#!/usr/bin/env bash
# =============================================================================
# merge_ckpts_to_hf.sh
#
# For each run and its "keep" steps:
#   1. If hf_model/model.safetensors is missing → run model_merge.py (FSDP→HF)
#   2. Delete optim_world_size_*.pt + model_world_size_*.pt from actor/
#   3. Delete every global_step_N that is NOT in the keep list
#
# Usage:
#   bash merge_ckpts_to_hf.sh           # DRY RUN (prints actions only)
#   bash merge_ckpts_to_hf.sh --execute  # actually execute
# =============================================================================

set -euo pipefail

EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then
    EXECUTE=true
fi

BASE="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl"
MERGE_SCRIPT="/data_storage/wyj/jxl/ArcherCodeR/tools/model_merge.py"
PYTHON="${PYTHON:-python}"

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; RST='\033[0m'

info()  { echo -e "${BLU}[INFO]${RST}  $*"; }
ok()    { echo -e "${GRN}[OK]${RST}    $*"; }
warn()  { echo -e "${YLW}[WARN]${RST}  $*"; }
act()   { echo -e "${RED}[ACT]${RST}   $*"; }

dry_or_run() {
    # $1 = description, rest = command
    local desc="$1"; shift
    if $EXECUTE; then
        act "$desc"
        "$@"
    else
        act "[DRY] $desc"
        echo "       CMD: $*"
    fi
}

# ── run catalogue ─────────────────────────────────────────────────────────────
# Format: "RUN_NAME:step1,step2,..."  (steps to KEEP)

declare -a RUNS=(
    # ── Group 1: Pure-GRPO-n12 (keep 80,100,105) ──────────────────────────
    "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-n12:80,100,105"

    # ── Group 2: A-pattern runs (keep 80,105) ─────────────────────────────
    "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl:80,105"
    "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08:80,105"
    "Unified-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12:80,105"
    "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl:80,105"
    "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08:80,105"
    "Unified-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12:80,105"
    "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl:80,105"
    "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-clip_ratio_c10-ppo_epoch3:80,105"
    "Unified-Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-token-separate-c3-e3:80,105"
    "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl:80,105"
    "Unified-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12:80,105"
    "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl:80,105"
    "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp08:80,105"
    "Unified-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp12:80,105"
)

# ── helpers ───────────────────────────────────────────────────────────────────

has_safetensors() {
    # $1 = actor_dir
    [[ -f "$1/hf_model/model.safetensors" ]]
}

has_model_shards() {
    # $1 = actor_dir  – check if model_world_size_*.pt exists
    ls "$1"/model_world_size_*_rank_0.pt 2>/dev/null | grep -q .
}

delete_training_state() {
    # Delete optim_*.pt and model_world_size_*.pt under actor/
    local actor_dir="$1"
    local count
    count=$(ls "$actor_dir"/optim_world_size_*.pt "$actor_dir"/model_world_size_*.pt 2>/dev/null | wc -l)
    if [[ $count -gt 0 ]]; then
        dry_or_run \
            "  DELETE $count training-state .pt files in $(basename "$actor_dir")" \
            bash -c "rm -f '$actor_dir'/optim_world_size_*.pt '$actor_dir'/model_world_size_*.pt"
    else
        ok "  No training-state .pt in $(basename "$actor_dir") – already clean"
    fi
}

merge_step() {
    local actor_dir="$1"
    local hf_dir="$actor_dir/hf_model"
    dry_or_run \
        "  MERGE FSDP shards → $hf_dir" \
        "$PYTHON" "$MERGE_SCRIPT" merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$hf_dir"
}

# ── main loop ─────────────────────────────────────────────────────────────────

if ! $EXECUTE; then
    warn "DRY RUN mode – pass --execute to actually run"
    echo
fi

total_freed_estimate=0

for entry in "${RUNS[@]}"; do
    run_name="${entry%%:*}"
    keep_str="${entry##*:}"
    IFS=',' read -ra KEEP_STEPS <<< "$keep_str"

    run_path="$BASE/$run_name"

    if [[ ! -d "$run_path" ]]; then
        warn "SKIP (not found): $run_name"
        continue
    fi

    echo
    info "════════════════════════════════════════════════════════════"
    info "RUN: $run_name"
    info "KEEP steps: ${KEEP_STEPS[*]}"
    info "════════════════════════════════════════════════════════════"

    # collect all existing global_step dirs
    declare -a ALL_STEPS=()
    for d in "$run_path"/global_step_*/; do
        [[ -d "$d" ]] || continue
        step_num=$(basename "$d" | sed 's/global_step_//')
        ALL_STEPS+=("$step_num")
    done

    # build keep set for fast lookup
    declare -A KEEP_SET=()
    for s in "${KEEP_STEPS[@]}"; do
        KEEP_SET["$s"]=1
    done

    # find the last (largest) keep step – preserve its training state for resumption
    last_keep_step=0
    for s in "${KEEP_STEPS[@]}"; do
        [[ $s -gt $last_keep_step ]] && last_keep_step=$s
    done

    for step in "${ALL_STEPS[@]}"; do
        step_dir="$run_path/global_step_$step"
        actor_dir="$step_dir/actor"

        if [[ -n "${KEEP_SET[$step]+x}" ]]; then
            # ── KEEP this step ───────────────────────────────────────
            if [[ $step -eq $last_keep_step ]]; then
                info "  [KEEP-LAST] step $step  ← training state preserved for resumption"
            else
                info "  [KEEP] step $step"
            fi

            if [[ ! -d "$actor_dir" ]]; then
                warn "  actor dir missing for step $step – skip"
                continue
            fi

            # 1. Merge if needed
            if has_safetensors "$actor_dir"; then
                ok "  hf_model/model.safetensors already exists – skip merge"
            elif has_model_shards "$actor_dir"; then
                merge_step "$actor_dir"
            else
                warn "  No model shards found in $actor_dir – cannot merge"
            fi

            # 2. Delete raw training state ONLY for non-last steps
            if [[ $step -eq $last_keep_step ]]; then
                ok "  Keeping optim + model_shard (last step, needed for resumption)"
            else
                delete_training_state "$actor_dir"
            fi

        else
            # ── DELETE this step ──────────────────────────────────────
            step_size_kb=$(du -sk "$step_dir" 2>/dev/null | awk '{print $1}')
            step_size_gb=$(awk "BEGIN{printf \"%.1f\", $step_size_kb/1024/1024}")
            total_freed_estimate=$((total_freed_estimate + step_size_kb))
            dry_or_run \
                "  DELETE step $step (${step_size_gb} GB)" \
                rm -rf "$step_dir"
        fi
    done

    unset KEEP_SET
    unset ALL_STEPS
done

echo
freed_gb=$(awk "BEGIN{printf \"%.1f\", $total_freed_estimate/1024/1024}")
info "Estimated space to be freed by step deletion: ~${freed_gb} GB"

if ! $EXECUTE; then
    echo
    warn "This was a DRY RUN. Run with --execute to apply changes."
fi
