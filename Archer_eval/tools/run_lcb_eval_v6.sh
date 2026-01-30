#!/bin/bash
#
# LCB v6 自动化评估脚本
# 扫描 output/ArcherCodeR/ 下的模型，自动评估 livecodebench_v6
# 支持 output_v6_8k/ 和 output_v6_16k/ 两种目录
#
# LiveCodeBench v6: 2025-02-01 ~ 最新
#
# 用法:
#   bash run_lcb_eval_v6.sh                    # 单次扫描并评估
#   DRY_RUN=1 bash run_lcb_eval_v6.sh          # 干运行，只显示待评估任务
#   MODE=watch bash run_lcb_eval_v6.sh         # 持续监控模式
#   TARGET_DIRS="exp1 exp2" bash run_lcb_eval_v6.sh  # 指定实验目录
#   NUM_WORKERS=64 bash run_lcb_eval_v6.sh     # 设置 CPU workers 数量
#

set -e

BASE_DIR=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval
cd ${BASE_DIR}
export PYTHONPATH=${BASE_DIR}:${BASE_DIR}/LiveCodeBench:$PYTHONPATH

PYTHON=/data/xuandong_zhao/anaconda3/envs/archer/bin/python
MODEL_ROOT="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"
LOG_FILE="${BASE_DIR}/tools/logs/lcb_v6_evaluated_tasks.log"
LCB_DIR=${BASE_DIR}/LiveCodeBench

# 一级目录筛选配置 (空格分隔)
# 留空则检测所有一级目录
TARGET_DIRS=${TARGET_DIRS:-""}

# Dry Run 模式: 只显示待评估任务，不实际执行
DRY_RUN=${DRY_RUN:-0}

# CPU workers 数量 (默认 64)
NUM_WORKERS=${NUM_WORKERS:-64}

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# ============ 时间日志函数 ============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============ 检查 LiveCodeBench ============
if [ ! -d "$LCB_DIR" ]; then
    log "LiveCodeBench 不存在，正在克隆..."
    git clone https://github.com/wizard-III/LiveCodeBench.git ${LCB_DIR}
fi

# 函数：记录任务完成状态
log_task_completion() {
    local eval_file="$1"
    local duration="$2"
    echo "$(date +%s)|${duration}s|$eval_file" >> "$LOG_FILE"
}

# 函数：执行单个评测任务
run_evaluation() {
    local eval_file="$1"
    local model_path="$2"
    
    log "=============================================="
    log "开始评测 (v6): $eval_file"
    
    # 提取实验名和步数
    # 路径格式: .../EXP_NAME/global_step_XXX/actor/hf_model/output_v6_[8k|16k]/livecodebench_v6.parquet
    local exp_name=$(echo "$model_path" | awk -F'/' '{for(i=1;i<=NF;i++) if($i ~ /global_step_/) print $(i-1)}')
    local global_step=$(echo "$model_path" | grep -oP 'global_step_\K\d+')
    local output_type=$(basename "$(dirname "$eval_file")")  # output_v6_8k or output_v6_16k
    
    # 根据输出目录添加后缀
    if [[ "$output_type" == "output_v6_16k" ]]; then
        exp_name="${exp_name}_v6_16k"
    else
        exp_name="${exp_name}_v6_8k"
    fi
    
    log "实验名: ${exp_name}"
    log "步数: ${global_step}"
    log "输出目录: ${output_type}"
    log "LCB 版本: v6 (2025-02-01 ~ 最新)"
    
    local start_time=$(date +%s)
    
    $PYTHON ${LCB_DIR}/lcb_runner/evaluation/compute_code_generation_metrics_online.py \
        --eval_file "${eval_file}" \
        --project_name ArcherEval_v6 \
        --experiment_name "${exp_name}" \
        --global_step "${global_step}" \
        --lcb_version v6 \
        --num_workers ${NUM_WORKERS}
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    # 检查评测是否成功
    local result_file="${eval_file}.pass.lcb.csv"
    if [ $exit_code -eq 0 ] && [ -f "$result_file" ]; then
        log_task_completion "$eval_file" "$duration"
        log "✓ 评测成功! 耗时: ${minutes}分${seconds}秒 (${duration}s)"
        log "结果文件: $result_file"
    else
        log "✗ 错误：评测失败或结果文件未生成"
        return 1
    fi
    log "=============================================="
}

# 函数：检查是否需要评估
is_eval_needed() {
    local eval_file="$1"
    # 存在 parquet 文件但不存在结果 csv 文件
    [ -f "$eval_file" ] && [ ! -f "${eval_file}.pass.lcb.csv" ]
}

# 函数：显示任务详情 (用于 dry run)
show_task_info() {
    local eval_file="$1"
    local model_path="$2"
    
    local exp_name=$(echo "$model_path" | awk -F'/' '{for(i=1;i<=NF;i++) if($i ~ /global_step_/) print $(i-1)}')
    local global_step=$(echo "$model_path" | grep -oP 'global_step_\K\d+')
    local output_type=$(basename "$(dirname "$eval_file")")
    
    # 根据输出目录添加后缀
    if [[ "$output_type" == "output_v6_16k" ]]; then
        exp_name="${exp_name}_v6_16k"
    else
        exp_name="${exp_name}_v6_8k"
    fi
    
    log "  📋 待评估任务 (v6):"
    log "     文件: $eval_file"
    log "     实验名: ${exp_name}"
    log "     步数: ${global_step}"
    log "     输出目录: ${output_type}"
    echo ""
}

# 函数：处理单个模型目录
process_model() {
    local model_path="$1"
    
    # 检查两种可能的输出目录 (v6 使用 output_v6_8k 和 output_v6_16k)
    for output_dir in "output_v6_8k" "output_v6_16k"; do
        local eval_file="${model_path}/actor/hf_model/${output_dir}/livecodebench_v6.parquet"
        
        if is_eval_needed "$eval_file"; then
            ((PENDING_COUNT++)) || true
            
            if [[ "$DRY_RUN" == "1" ]]; then
                show_task_info "$eval_file" "$model_path"
            else
                log "发现新任务: $eval_file"
                run_evaluation "$eval_file" "$model_path"
            fi
        fi
    done
}

# 函数：扫描并处理所有模型
scan_and_process() {
    log "=============================================="
    if [[ "$DRY_RUN" == "1" ]]; then
        log "🔍 DRY RUN 模式 - 只扫描不执行"
    fi
    log "开始扫描新任务 (LCB v6)..."
    log "扫描目录: ${MODEL_ROOT}"
    
    local found_count=0
    PENDING_COUNT=0  # 待评估任务计数
    
    if [[ -n "$TARGET_DIRS" ]]; then
        log "使用筛选目录: $TARGET_DIRS"
        IFS=' ' read -ra TARGET_DIR_ARRAY <<< "$TARGET_DIRS"
        for dir in "${TARGET_DIR_ARRAY[@]}"; do
            local full_path="${MODEL_ROOT}/${dir}"
            if [[ -d "$full_path" ]]; then
                log "扫描: $full_path"
                for model_path in "${full_path}"/global_step_*; do
                    if [[ -d "$model_path" ]]; then
                        process_model "$model_path"
                        ((found_count++)) || true
                    fi
                done
            else
                log "警告：目录不存在，跳过: $full_path"
            fi
        done
    else
        # 无筛选时扫描所有目录
        log "扫描所有目录 (无筛选)"
        for exp_dir in "${MODEL_ROOT}"/*; do
            if [[ -d "$exp_dir" ]]; then
                for model_path in "${exp_dir}"/global_step_*; do
                    if [[ -d "$model_path" ]]; then
                        process_model "$model_path"
                        ((found_count++)) || true
                    fi
                done
            fi
        done
    fi
    
    log "=============================================="
    log "扫描完成，共检查 ${found_count} 个 checkpoint"
    log "待评估任务 (v6): ${PENDING_COUNT} 个"
    
    if [[ "$DRY_RUN" == "1" ]] && [[ "$PENDING_COUNT" -gt 0 ]]; then
        log ""
        log "💡 要执行评估，请运行:"
        log "   bash tools/run_lcb_eval_v6.sh"
    fi
    log "=============================================="
}

# ============ 主程序 ============
log "LCB v6 评估脚本启动"
log "模型根目录: ${MODEL_ROOT}"
log "LCB 版本: v6 (2025-02-01 ~ 最新)"

# 运行模式：单次扫描 或 持续监控
MODE=${MODE:-"once"}  # once | watch | dry

# 支持 MODE=dry 作为 DRY_RUN=1 的简写
if [[ "$MODE" == "dry" ]]; then
    DRY_RUN=1
    MODE="once"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    log "=============================================="
    log "🔍 DRY RUN 模式"
    log "   只显示待评估任务，不实际执行"
    log "=============================================="
fi

if [[ "$MODE" == "watch" ]]; then
    log "模式: 持续监控 (每60秒扫描一次)"
    while true; do
        scan_and_process
        log "等待60秒后再次扫描..."
        sleep 60
    done
else
    log "模式: 单次扫描"
    scan_and_process
    if [[ "$DRY_RUN" != "1" ]]; then
        log "评估完成!"
    fi
fi


