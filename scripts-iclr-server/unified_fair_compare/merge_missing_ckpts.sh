#!/usr/bin/env bash
# =============================================================================
# merge_missing_ckpts.sh
#
# Scan ALL runs under self-rl-jxl and merge FSDP shards → HuggingFace format
# for every global_step_N that:
#   - has model_world_size_8_rank_0.pt  (shards exist)
#   - does NOT yet have actor/hf_model/model.safetensors (not yet merged)
#
# NO files are deleted. This script is purely additive and idempotent.
#
# Usage:
#   bash merge_missing_ckpts.sh                        # dry run
#   bash merge_missing_ckpts.sh --execute              # serial (safe default)
#   bash merge_missing_ckpts.sh --execute --jobs 8     # 8-way parallel (recommended)
#   bash merge_missing_ckpts.sh --execute --jobs 16    # 16-way parallel
#   bash merge_missing_ckpts.sh --execute \
#       --run "Unified-Intuitor-*"                     # filter to one run (glob)
#
# Parallel notes:
#   - Each merge uses ~10-15G RAM  (loads 8×848M FP32 shards)
#   - This machine has 1.8T RAM / 128 cores → --jobs 16 is comfortably safe
#   - Bottleneck is usually storage I/O; 8-16 jobs is a good sweet spot
#   - Logs for each step written to:  output/self-rl-jxl/.merge_logs/<run>__<step>.log
# =============================================================================

set -uo pipefail

EXECUTE=false
JOBS=1
RUN_FILTER="*"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) EXECUTE=true ;;
        --jobs)    JOBS="$2"; shift ;;
        --run)     RUN_FILTER="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

BASE="/data_storage/wyj/jxl/ArcherCodeR/output/self-rl-jxl"
MERGE_SCRIPT="/data_storage/wyj/jxl/ArcherCodeR/tools/model_merge.py"
PYTHON="${PYTHON:-python}"
LOG_DIR="$BASE/.merge_logs"
mkdir -p "$LOG_DIR"

# ── colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'
info()  { echo -e "${BLU}[INFO]${RST}  $*"; }
ok()    { echo -e "${GRN}[OK]${RST}    $*"; }
warn()  { echo -e "${YLW}[WARN]${RST}  $*"; }
act()   { echo -e "${RED}[ACT]${RST}   $*"; }
dry()   { echo -e "${CYN}[DRY]${RST}   $*"; }

if ! $EXECUTE; then
    warn "DRY RUN mode – pass --execute to actually merge"
    echo
fi

# ── collect all unmerged steps ─────────────────────────────────────────────────
declare -a ACTOR_DIRS=()   # each element: actor_dir (contains run info via path)

for run_dir in "$BASE"/$RUN_FILTER/; do
    [[ -d "$run_dir" ]] || continue
    for step_dir in "$run_dir"global_step_*/; do
        [[ -d "$step_dir" ]] || continue
        actor_dir="$step_dir/actor"
        # skip if already merged
        [[ -f "$actor_dir/hf_model/model.safetensors" ]] && continue
        # skip if no model shards (nothing to merge)
        ls "$actor_dir"/model_world_size_*_rank_0.pt 2>/dev/null | grep -q . || continue
        ACTOR_DIRS+=("$actor_dir")
    done
done

total=${#ACTOR_DIRS[@]}

if [[ $total -eq 0 ]]; then
    ok "All checkpoints already merged. Nothing to do."
    exit 0
fi

info "Found $total unmerged steps  (--jobs $JOBS)"
echo

# ── dry run: just print and exit ───────────────────────────────────────────────
if ! $EXECUTE; then
    for actor_dir in "${ACTOR_DIRS[@]}"; do
        run=$(basename "$(dirname "$(dirname "$actor_dir")")")
        step=$(basename "$(dirname "$actor_dir")")
        dry "WOULD MERGE: $run / $step  →  $actor_dir/hf_model"
    done
    echo
    info "DRY RUN summary: would merge $total steps"
    warn "Run with --execute to actually perform merges"
    exit 0
fi

# ── write a tiny single-step worker script ────────────────────────────────────
mkdir -p "$LOG_DIR"
WORKER="$LOG_DIR/_worker.sh"
cat > "$WORKER" <<'WORKER_EOF'
#!/usr/bin/env bash
set -uo pipefail
PYTHON="${PYTHON:-python}"
MERGE_SCRIPT="$1"
actor_dir="$2"
LOG_DIR="$3"

run=$(basename "$(dirname "$(dirname "$actor_dir")")")
step=$(basename "$(dirname "$actor_dir")")
hf_dir="$actor_dir/hf_model"
log="$LOG_DIR/${run}__${step}.log"

# idempotency check
if [[ -f "$hf_dir/model.safetensors" ]]; then
    echo "[SKIP]  $run / $step  (already done)"
    exit 0
fi

echo "[START] $(date '+%H:%M:%S')  $run / $step"
if "$PYTHON" "$MERGE_SCRIPT" merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$hf_dir" \
        > "$log" 2>&1; then
    echo "[OK]    $(date '+%H:%M:%S')  $run / $step"
    exit 0
else
    echo "[FAIL]  $(date '+%H:%M:%S')  $run / $step  →  see $log"
    exit 1
fi
WORKER_EOF
chmod +x "$WORKER"

# ── run: serial (JOBS=1) or parallel (JOBS>1) ─────────────────────────────────
if [[ $JOBS -le 1 ]]; then
    # serial
    ok_count=0; fail_count=0
    for actor_dir in "${ACTOR_DIRS[@]}"; do
        if bash "$WORKER" "$MERGE_SCRIPT" "$actor_dir" "$LOG_DIR"; then
            ok_count=$((ok_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
else
    # parallel via background jobs + semaphore
    info "Launching parallel merges  (max $JOBS concurrent)…"
    echo

    ok_count=0; fail_count=0
    declare -a pids=()
    declare -A pid_status=()   # pid → pending

    # Wait until at least one slot is free, reaping any finished PIDs
    reap_one() {
        while true; do
            for i in "${!pids[@]}"; do
                local pid="${pids[$i]}"
                if ! kill -0 "$pid" 2>/dev/null; then
                    wait "$pid" 2>/dev/null && ok_count=$((ok_count + 1)) || fail_count=$((fail_count + 1))
                    unset 'pids[$i]'
                    return
                fi
            done
            sleep 0.3
        done
    }

    for actor_dir in "${ACTOR_DIRS[@]}"; do
        # throttle: reap until a slot is free
        while [[ ${#pids[@]} -ge $JOBS ]]; do
            reap_one
        done
        bash "$WORKER" "$MERGE_SCRIPT" "$actor_dir" "$LOG_DIR" &
        pids+=($!)
    done

    # drain all remaining background jobs
    while [[ ${#pids[@]} -gt 0 ]]; do
        reap_one
    done
fi

# ── summary ────────────────────────────────────────────────────────────────────
echo
info "══════════════════════════════════════════════════════════"
info "Done.  total=$total  ok=$ok_count  fail=$fail_count"
[[ $fail_count -gt 0 ]] && warn "Check failed logs in: $LOG_DIR/"
info "Per-step logs: $LOG_DIR/"
info "══════════════════════════════════════════════════════════"
