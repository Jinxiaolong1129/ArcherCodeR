#!/bin/bash
#
# 补充缺失的 LiveCodeBench v6 16K 评估脚本
# 共 3 个缺失的评估任务 (TrajectoryEntropy-temp1.2 的 Steps 50, 80, 100)
#

set -e

# ============ 环境配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval:$PYTHONPATH
PYTHON=/data/xuandong_zhao/anaconda3/envs/archer/bin/python

# 导入环境变量
if [ -f /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env ]; then
    export $(grep -v '^#' /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval/.env | xargs)
    echo "Loaded environment variables from .env"
fi

# Clear Ray
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

ray stop --force 2>/dev/null || true

# ============ 路径配置 ============
BASE_DIR=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR
EVAL_DIR=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval
OUTPUT_ROOT=${BASE_DIR}/output/ArcherCodeR
DATA_DIR=${EVAL_DIR}/data/test

# ============ 评估参数 ============
n_gpus=4
tp_size=1
n_samples=8
temperature=0.6
top_p=0.95
max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 16))    # 16K response
batch_size=2048

OUTPUT_SUBDIR="output_v6_16k"

# ============ 日志配置 ============
LOG_DIR=${EVAL_DIR}/tools/logs
mkdir -p ${LOG_DIR}
MAIN_LOG=${LOG_DIR}/batch_eval_missing_lcbv6_16k_$(date +%Y%m%d_%H%M%S).log

# ============ 缺失的评估任务列表 ============
MISSING_TASKS=(
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2:50"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2:80"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2:100"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

check_eval_exists() {
    local exp_name=$1
    local step=$2
    local result_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model/${OUTPUT_SUBDIR}/livecodebench_v6.parquet"
    [ -f "$result_path" ]
}

run_eval() {
    local exp_name=$1
    local step=$2
    local model_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    local output_dir="${model_path}/${OUTPUT_SUBDIR}"
    
    log "📊 开始 LCB v6 16K 评估: ${exp_name} step ${step}"
    
    if check_eval_exists "$exp_name" "$step"; then
        log "✓ 评估结果已存在，跳过"
        return 0
    fi
    
    mkdir -p "${output_dir}"
    
    local start_time=$(date +%s)
    
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u AMD_VISIBLE_DEVICES \
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=ArcherCodeR_Eval \
        +trainer.experiment_name=${exp_name} \
        +trainer.task_name=livecodebench_v6 \
        +trainer.global_step=${step} \
        +trainer.use_wandb=False \
        model.path=${model_path} \
        data.path=${DATA_DIR}/livecodebench_v6.json \
        data.output_path=${output_dir}/livecodebench_v6.parquet \
        data.batch_size=${batch_size} \
        data.n_samples=${n_samples} \
        rollout.name=vllm \
        rollout.gpu_memory_utilization=0.9 \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.disable_log_stats=False \
        rollout.tensor_model_parallel_size=${tp_size} \
        rollout.temperature=${temperature} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.prompt_length=${max_prompt_length} \
        rollout.response_length=${max_response_length} \
        rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        2>&1 | tee -a ${MAIN_LOG}
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ] && [ -f "${output_dir}/livecodebench_v6.parquet" ]; then
        log "✓ 评估完成 (耗时: ${duration}s)"
        return 0
    else
        log "✗ 评估失败"
        return 1
    fi
}

# ============ 主流程 ============
main() {
    log "============================================================"
    log "补充缺失的 LCB v6 16K 评估"
    log "共 ${#MISSING_TASKS[@]} 个任务"
    log "============================================================"
    
    local completed=0
    local failed=0
    
    for task in "${MISSING_TASKS[@]}"; do
        exp_name="${task%%:*}"
        step="${task##*:}"
        
        log ""
        log "处理: ${exp_name} Step ${step}"
        
        if run_eval "$exp_name" "$step"; then
            ((completed++))
        else
            ((failed++))
        fi
    done
    
    log ""
    log "============================================================"
    log "🎉 完成! 成功: ${completed}, 失败: ${failed}"
    log "============================================================"
}

cd ${EVAL_DIR}
main

