#!/bin/bash
# Part 1: GPU 0,1,2,3
# Experiments: Archer-Intuitor (3) + Archer-ProbDisparity-n12
# Total: 84 files
#
# Priority: 8k > 16k, aime2024 > lcb_v5 > lcb_v6 > aime2025
#
# Usage: bash tools/eval_certainty/bash_certainty_data4_part1.sh

# 不使用 set -e，让脚本在单个任务失败时继续执行

# ============ 环境配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 让 Ray 不重写 CUDA_VISIBLE_DEVICES，避免物理 GPU id 索引错误
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_DISABLE_IMPORT_WARNING=1
export PYTHONPATH=/data3/user/jin509/Archer_eval:$PYTHONPATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# Part 1 使用独立的 Ray 目录，避免与 Part 2 冲突
export RAY_TMPDIR=/data3/user/jin509/ray/part1
mkdir -p $RAY_TMPDIR

# 清理其他 GPU 环境变量
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

PYTHON=/data3/user/jin509/conda_envs/archer/bin/python
BASE_DIR=/data3/user/jin509/Archer_eval

cd ${BASE_DIR}
mkdir -p tools/eval_certainty/logs

ray stop --force 2>/dev/null || true

# Part 1 experiments (84 files)
EXPERIMENTS=(
    "Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-kl005"
    "Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
    "Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp0.8"
    "Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
)

BASE_OUTPUT_DIR="/data4/user/jin509/Archer_eval/output/ArcherCodeR"

# ============ 日志配置 ============
LOG_DIR=${BASE_DIR}/tools/eval_certainty/logs
MAIN_LOG=${LOG_DIR}/data4_part1_$(date +%Y%m%d_%H%M%S).log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

# ============ CUDA 预检 ============
preflight_cuda() {
    log "🔎 CUDA 预检开始"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "✗ 未找到 nvidia-smi，请确认驱动已安装"
        return 1
    fi

    if ! nvidia-smi >/dev/null 2>&1; then
        log "✗ nvidia-smi 运行失败，请检查驱动/权限"
        return 1
    fi

    $PYTHON - <<'PY' 2>&1 | tee -a ${MAIN_LOG}
import os
import sys
import torch

print("torch_version:", torch.__version__)
print("torch_cuda_version:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda_is_available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    sys.exit(2)

print("cuda_device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        log "✗ CUDA 预检失败，请检查 CUDA 驱动与 PyTorch 版本匹配"
        return 1
    fi

    log "✓ CUDA 预检通过"
    return 0
}

# Priority function: returns sort key for priority ordering
# Priority: 8k > 16k, aime2024 > lcb_v5 > lcb_v6 > aime2025
get_priority() {
    local path="$1"
    local priority=0
    
    # 16k gets lower priority (higher number = lower priority)
    if [[ "$path" == *"output_16k"* ]]; then
        priority=$((priority + 100))
    fi
    
    # Dataset priority: aime2024=0, lcb_v5=10, lcb_v6=20, aime2025=30
    if [[ "$path" == *"aime2024"* ]]; then
        priority=$((priority + 0))
    elif [[ "$path" == *"livecodebench_v5"* ]]; then
        priority=$((priority + 10))
    elif [[ "$path" == *"livecodebench_v6"* ]]; then
        priority=$((priority + 20))
    elif [[ "$path" == *"aime2025"* ]]; then
        priority=$((priority + 30))
    else
        priority=$((priority + 50))
    fi
    
    echo "$priority"
}

# Format seconds to human readable time
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    if [[ $hours -gt 0 ]]; then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif [[ $minutes -gt 0 ]]; then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

process_file() {
    local PARQUET_PATH="$1"
    local FILENAME=$(basename "$PARQUET_PATH")
    local DIRNAME=$(dirname "$PARQUET_PATH")
    local SUBDIR=$(basename "$DIRNAME")
    
    # Determine response_length based on directory
    if [[ "$SUBDIR" == *"16k"* ]]; then
        RESP_LEN=16384
    else
        RESP_LEN=8192
    fi
    
    # Determine prompt_length based on dataset type
    # AIME: 2048 (1024*2), LCB: 4096 (1024*4)
    if [[ "$FILENAME" == *"aime"* ]]; then
        PROMPT_LEN=2048
    else
        PROMPT_LEN=4096
    fi
    
    # Model path: go up from output/output_16k directory
    MODEL_PATH=$(dirname "$DIRNAME")
    OUTPUT_PATH="${PARQUET_PATH%.parquet}_certainty.parquet"
    
    # Skip if already done
    if [[ -f "$OUTPUT_PATH" ]]; then
        log "SKIP (already done): $PARQUET_PATH"
        return 0
    fi
    
    log ""
    log "============================================================"
    log "Processing: $FILENAME"
    log "  Input: $PARQUET_PATH"
    log "  Model: $MODEL_PATH"
    log "  Prompt length: $PROMPT_LEN"
    log "  Response length: $RESP_LEN"
    log "  Output: $OUTPUT_PATH"
    log "============================================================"
    
    local START_TIME=$(date +%s)
    
    $PYTHON -m verl.trainer.compute_certainty \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        model.path="$MODEL_PATH" \
        data.input_path="$PARQUET_PATH" \
        data.output_path="$OUTPUT_PATH" \
        data.batch_size=32 \
        rollout.prompt_length=$PROMPT_LEN \
        rollout.response_length=$RESP_LEN \
        rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor.ppo_mini_batch_size=16 \
        actor.ppo_micro_batch_size_per_gpu=1 \
        2>&1 | tee -a ${MAIN_LOG}
    
    local exit_code=${PIPESTATUS[0]}
    
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    local DURATION_STR=$(format_duration $DURATION)
    
    ray stop --force 2>/dev/null || true
    sleep 5
    
    if [ $exit_code -eq 0 ] && [ -f "$OUTPUT_PATH" ]; then
        log ""
        log "✓ Completed: $FILENAME"
        log ">>> Duration: $DURATION_STR"
        log ""
        return 0
    else
        log ""
        log "✗ Failed: $FILENAME (exit_code: $exit_code)"
        log ">>> Duration: $DURATION_STR"
        log ""
        return 1
    fi
}

# ============ 主流程 ============

main() {
    log "################################################################"
    log "Part 1 - Certainty Computation (GPU 0,1,2,3)"
    log "################################################################"
    log "Experiments: ${#EXPERIMENTS[@]}"
    log "GPU: ${CUDA_VISIBLE_DEVICES}"
    log "Priority: 8k > 16k, aime2024 > lcb_v5 > lcb_v6 > aime2025"
    log "================================================================"

    # CUDA 预检
    if ! preflight_cuda; then
        log "⚠ CUDA 预检未通过，终止运行"
        return 1
    fi

    # Collect all files with priority
    declare -a ALL_FILES_WITH_PRIORITY

    for EXP in "${EXPERIMENTS[@]}"; do
        PARQUET_FILES=$(find "$BASE_OUTPUT_DIR/$EXP" -name "*.parquet" -type f 2>/dev/null | grep -E "(aime2024|aime2025|livecodebench)" | grep -v "_certainty")
        
        for PARQUET_PATH in $PARQUET_FILES; do
            PRIORITY=$(get_priority "$PARQUET_PATH")
            ALL_FILES_WITH_PRIORITY+=("$PRIORITY:$PARQUET_PATH")
        done
    done

    # Sort by priority
    IFS=$'\n' SORTED_FILES=($(sort -t: -k1 -n <<<"${ALL_FILES_WITH_PRIORITY[*]}"))
    unset IFS

    TOTAL_FILES=${#SORTED_FILES[@]}

    log ""
    log "Total files: $TOTAL_FILES"
    log ""

    # Count done and todo
    DONE_COUNT=0
    TODO_COUNT=0
    for item in "${SORTED_FILES[@]}"; do
        path="${item#*:}"
        output="${path%.parquet}_certainty.parquet"
        if [[ -f "$output" ]]; then
            DONE_COUNT=$((DONE_COUNT + 1))
        else
            TODO_COUNT=$((TODO_COUNT + 1))
        fi
    done

    log "Status: $DONE_COUNT done, $TODO_COUNT remaining"
    log ""
    log "Processing order:"
    log "================================================================"
    for item in "${SORTED_FILES[@]}"; do
        path="${item#*:}"
        output="${path%.parquet}_certainty.parquet"
        if [[ -f "$output" ]]; then
            log "  [DONE] $path"
        else
            log "  [TODO] $path"
        fi
    done
    log "================================================================"
    log ""

    # Main loop
    SCRIPT_START_TIME=$(date +%s)
    PROCESSED=0
    ACTUALLY_PROCESSED=0
    FAILED_COUNT=0
    SUCCESS_COUNT=0

    for item in "${SORTED_FILES[@]}"; do
        PARQUET_PATH="${item#*:}"
        PROCESSED=$((PROCESSED + 1))
        
        OUTPUT_PATH="${PARQUET_PATH%.parquet}_certainty.parquet"
        if [[ -f "$OUTPUT_PATH" ]]; then
            log "[$PROCESSED/$TOTAL_FILES] SKIP (already done): $(basename $PARQUET_PATH)"
            continue
        fi
        
        ACTUALLY_PROCESSED=$((ACTUALLY_PROCESSED + 1))
        log ""
        log ">>> Progress: $PROCESSED / $TOTAL_FILES (processed: $ACTUALLY_PROCESSED)"
        
        # 使用 || true 确保单个失败不会中断脚本
        if process_file "$PARQUET_PATH"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            log "⚠ 任务失败，继续下一个: $(basename $PARQUET_PATH)"
        fi
    done

    SCRIPT_END_TIME=$(date +%s)
    TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
    TOTAL_DURATION_STR=$(format_duration $TOTAL_DURATION)

    log ""
    log "============================================================"
    log "Part 1 Complete!"
    log "  Total files: $TOTAL_FILES"
    log "  Actually processed: $ACTUALLY_PROCESSED"
    log "  Success: $SUCCESS_COUNT"
    log "  Failed: $FAILED_COUNT"
    log "  Total time: $TOTAL_DURATION_STR"
    log "  Log file: $MAIN_LOG"
    log "============================================================"
}

main
