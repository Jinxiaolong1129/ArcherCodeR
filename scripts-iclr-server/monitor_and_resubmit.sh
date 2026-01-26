#!/usr/bin/env bash
# 监控 xuandong_zhao 的作业，如果失败则自动重新提交
# 用法: ./monitor_and_resubmit.sh [检查间隔秒数，默认300]

set -euo pipefail

# 配置
USER="xuandong_zhao"
PARTITION="schmidt_sciences"
CHECK_INTERVAL=${1:-300}  # 默认5分钟检查一次
MAX_RETRIES=3  # 每个作业最大重试次数

# 作业名称到提交脚本的映射
declare -A JOB_SCRIPTS=(
    ["intuitor"]="scripts-iclr-server/intuitor/slurm-Intuitor-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray_v2-detailed.sh"
    # ["token-en"]="scripts-iclr-server/token_entropy/slurm-TokenEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray-detailed.sh"
    ["traj-ent"]="scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray-detailed.sh"
    ["prob-dis"]="scripts-iclr-server/prob_disparity/slurm-ProbDisparity-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray-detailed.sh"
)

# 作业描述（用于日志输出）
declare -A JOB_NAMES=(
    ["intuitor"]="Intuitor"
    # ["token-en"]="Token Entropy"
    ["traj-ent"]="Trajectory Entropy"
    ["prob-dis"]="Prob Disparity"
)

# 重试计数器
declare -A RETRY_COUNT=(
    ["intuitor"]=0
    ["token-en"]=0
    ["traj-ent"]=0
    ["prob-dis"]=0
)

# 记录已完成的作业（成功完成，不需要再监控）
declare -A COMPLETED_JOBS=(
    ["intuitor"]=0
    ["token-en"]=0
    ["traj-ent"]=0
    ["prob-dis"]=0
)

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 获取用户当前运行的作业
get_running_jobs() {
    squeue -u "$USER" -p "$PARTITION" -h -o "%j" 2>/dev/null || echo ""
}

# 检查作业是否在运行或等待中
is_job_running() {
    local job_prefix="$1"
    local running_jobs
    running_jobs=$(get_running_jobs)
    echo "$running_jobs" | grep -q "^${job_prefix}" && return 0 || return 1
}

# 检查作业是否成功完成（通过检查输出目录的checkpoint）
# 这个函数可以根据实际情况修改
is_job_completed_successfully() {
    local job_prefix="$1"
    # 可以添加检查逻辑，比如检查checkpoint文件是否存在
    # 目前简单返回false，表示未完成
    return 1
}

# 提交作业
submit_job() {
    local job_prefix="$1"
    local script="${JOB_SCRIPTS[$job_prefix]}"
    local job_name="${JOB_NAMES[$job_prefix]}"
    
    if [[ ! -f "$script" ]]; then
        log "❌ 脚本不存在: $script"
        return 1
    fi
    
    log "🚀 正在提交 ${job_name} 作业..."
    if sbatch "$script"; then
        log "✅ ${job_name} 作业提交成功"
        return 0
    else
        log "❌ ${job_name} 作业提交失败"
        return 1
    fi
}

# 主监控循环
main() {
    log "=========================================="
    log "🔍 开始监控作业 (用户: $USER, 分区: $PARTITION)"
    log "📋 监控的作业: ${!JOB_SCRIPTS[*]}"
    log "⏰ 检查间隔: ${CHECK_INTERVAL}秒"
    log "🔄 最大重试次数: ${MAX_RETRIES}"
    log "=========================================="
    
    while true; do
        local all_completed=true
        
        for job_prefix in "${!JOB_SCRIPTS[@]}"; do
            job_name="${JOB_NAMES[$job_prefix]}"
            
            # 跳过已标记为完成的作业
            if [[ ${COMPLETED_JOBS[$job_prefix]} -eq 1 ]]; then
                continue
            fi
            
            all_completed=false
            
            if is_job_running "$job_prefix"; then
                log "✅ ${job_name} 正在运行或等待中"
            else
                # 作业不在运行，检查是否成功完成
                if is_job_completed_successfully "$job_prefix"; then
                    log "🎉 ${job_name} 已成功完成"
                    COMPLETED_JOBS[$job_prefix]=1
                else
                    # 作业失败，需要重新提交
                    current_retries=${RETRY_COUNT[$job_prefix]}
                    
                    if [[ $current_retries -lt $MAX_RETRIES ]]; then
                        log "⚠️  ${job_name} 未在运行，尝试重新提交 (重试 $((current_retries + 1))/$MAX_RETRIES)"
                        
                        if submit_job "$job_prefix"; then
                            RETRY_COUNT[$job_prefix]=$((current_retries + 1))
                        fi
                    else
                        log "❌ ${job_name} 已达到最大重试次数 ($MAX_RETRIES)，停止重试"
                        COMPLETED_JOBS[$job_prefix]=1  # 标记为完成，不再监控
                    fi
                fi
            fi
        done
        
        # 检查是否所有作业都已完成
        if $all_completed; then
            log "🎉 所有作业都已完成或达到最大重试次数"
            break
        fi
        
        log "💤 等待 ${CHECK_INTERVAL} 秒后再次检查..."
        log "------------------------------------------"
        sleep "$CHECK_INTERVAL"
    done
    
    log "=========================================="
    log "📊 监控结束，作业状态汇总:"
    for job_prefix in "${!JOB_SCRIPTS[@]}"; do
        job_name="${JOB_NAMES[$job_prefix]}"
        retries=${RETRY_COUNT[$job_prefix]}
        log "   - ${job_name}: 重试次数 $retries"
    done
    log "=========================================="
}

# 运行主函数
main

