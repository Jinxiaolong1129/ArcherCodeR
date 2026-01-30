#!/bin/bash
#
# 批量模型合并与 LiveCodeBench v6 评估脚本 (SLURM版本 - 4 GPU)
# 8K 输出长度版本 - Part 4
#

set -e

# ============ 获取脚本自身路径 ============
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_NAME="$(basename "$0")"

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

# 关键修复: 彻底清除 AMD GPU 环境变量
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

ray stop --force 2>/dev/null || true

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

echo "Environment check:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"

# ============ 路径配置 ============
BASE_DIR=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR
EVAL_DIR=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval
OUTPUT_ROOT=${BASE_DIR}/output/ArcherCodeR
DATA_DIR=${EVAL_DIR}/data/test

# ============ 评估参数 ============
n_gpus=4
tp_size=1
temperature=0.6
top_p=0.95
batch_size=2048

# LCB 参数
lcb_n_samples=8             # pass@8
lcb_max_prompt_length=$((1024 * 4))       # 4K prompt
lcb_max_response_length=$((1024 * 8))     # 8K response

# 输出目录名称 (区分不同配置)
OUTPUT_SUBDIR="output_v6_8k"

# ============ 日志配置 ============
LOG_DIR=${EVAL_DIR}/tools/logs
mkdir -p ${LOG_DIR}
MAIN_LOG=${LOG_DIR}/batch_merge_eval_lcb_v6_8k_part4_$(date +%Y%m%d_%H%M%S).log

# ============ 实验列表 Part 4 (2个实验) ============
EXPERIMENTS=(
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
)

# Checkpoint steps
STEPS=(10 50 80 100 105)

# ============ 辅助函数 ============

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

# 检查 checkpoint 是否存在 actor 目录
check_checkpoint_exists() {
    local exp_name=$1
    local step=$2
    local ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
    
    if [ -d "$ckpt_path" ] && [ -f "$ckpt_path/config.json" ]; then
        return 0
    else
        return 1
    fi
}

# 检查 HF 模型是否已存在
check_hf_model_exists() {
    local exp_name=$1
    local step=$2
    local hf_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    
    if [ -d "$hf_path" ] && [ -f "$hf_path/config.json" ]; then
        return 0
    else
        return 1
    fi
}

# 检查评估结果是否已存在
check_eval_exists() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local result_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model/${OUTPUT_SUBDIR}/${dataset}.parquet"
    
    if [ -f "$result_path" ]; then
        return 0
    else
        return 1
    fi
}

# 合并模型
merge_model() {
    local exp_name=$1
    local step=$2
    local ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
    local hf_path="${ckpt_path}/hf_model"
    
    log "🔧 开始合并模型: ${exp_name} step ${step}"
    
    # 检查是否已合并
    if check_hf_model_exists "$exp_name" "$step"; then
        log "✓ HF模型已存在，跳过合并: ${hf_path}"
        return 0
    fi
    
    # 执行合并
    local start_time=$(date +%s)
    
    $PYTHON -m tools.model_merge merge \
        --backend fsdp \
        --local_dir "${ckpt_path}" \
        --target_dir "${hf_path}" 2>&1 | tee -a ${MAIN_LOG}
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ] && [ -f "${hf_path}/config.json" ]; then
        log "✓ 模型合并成功 (耗时: ${duration}s): ${exp_name} step ${step}"
        return 0
    else
        log "✗ 模型合并失败: ${exp_name} step ${step}"
        # 清理失败的合并结果
        [ -d "${hf_path}" ] && rm -rf "${hf_path}"
        return 1
    fi
}

# 复制脚本到输出目录用于追踪
copy_script_to_output() {
    local exp_name=$1
    local step=$2
    local output_dir="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model/${OUTPUT_SUBDIR}"
    
    mkdir -p "${output_dir}"
    cp "${SCRIPT_PATH}" "${output_dir}/${SCRIPT_NAME}"
    log "📋 已复制评估脚本到: ${output_dir}/${SCRIPT_NAME}"
}

# 运行 LCB 评估
run_eval_lcb() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local model_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    local output_dir="${model_path}/${OUTPUT_SUBDIR}"
    
    log "📊 开始LCB评估: ${exp_name} step ${step} - ${dataset}"
    
    # 检查是否已评估
    if check_eval_exists "$exp_name" "$step" "$dataset"; then
        log "✓ 评估结果已存在，跳过: ${output_dir}/${dataset}.parquet"
        return 0
    fi
    
    # 创建输出目录
    mkdir -p "${output_dir}"
    
    local start_time=$(date +%s)
    
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u AMD_VISIBLE_DEVICES \
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=ArcherCodeR_Eval \
        +trainer.experiment_name=${exp_name} \
        +trainer.task_name=${dataset} \
        +trainer.global_step=${step} \
        +trainer.use_wandb=False \
        model.path=${model_path} \
        data.path=${DATA_DIR}/${dataset}.json \
        data.output_path=${output_dir}/${dataset}.parquet \
        data.batch_size=${batch_size} \
        data.n_samples=${lcb_n_samples} \
        rollout.name=vllm \
        rollout.gpu_memory_utilization=0.9 \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.disable_log_stats=False \
        rollout.tensor_model_parallel_size=${tp_size} \
        rollout.temperature=${temperature} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.prompt_length=${lcb_max_prompt_length} \
        rollout.response_length=${lcb_max_response_length} \
        rollout.max_num_batched_tokens=$((lcb_max_prompt_length + lcb_max_response_length)) \
        2>&1 | tee -a ${MAIN_LOG}
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ] && [ -f "${output_dir}/${dataset}.parquet" ]; then
        log "✓ LCB评估完成 (耗时: ${duration}s): ${exp_name} step ${step} - ${dataset}"
        return 0
    else
        log "✗ LCB评估失败: ${exp_name} step ${step} - ${dataset}"
        return 1
    fi
}

# 处理单个 checkpoint - LCB
process_checkpoint_lcb() {
    local exp_name=$1
    local step=$2
    
    log "============================================================"
    log "处理LCB: ${exp_name} - global_step_${step}"
    log "============================================================"
    
    # 检查 checkpoint 是否存在
    if ! check_checkpoint_exists "$exp_name" "$step"; then
        log "⚠ Checkpoint 不存在，跳过: ${exp_name} step ${step}"
        return 0
    fi
    
    # 检查 HF 模型
    if ! check_hf_model_exists "$exp_name" "$step"; then
        log "⚠ HF模型不存在，尝试合并: ${exp_name} step ${step}"
        if ! merge_model "$exp_name" "$step"; then
            log "⚠ 合并失败，跳过评估: ${exp_name} step ${step}"
            return 1
        fi
    fi
    
    # 复制脚本到输出目录
    copy_script_to_output "$exp_name" "$step"
    
    # 评估 LiveCodeBench v6
    run_eval_lcb "$exp_name" "$step" "livecodebench_v6"
    
    log "✓ 完成LCB处理: ${exp_name} - global_step_${step}"
}

# ============ 主流程 ============

main() {
    log "============================================================"
    log "批量模型合并与 LiveCodeBench v6 评估脚本 (SLURM - 4 GPU)"
    log "8K 输出长度版本 - Part 4"
    log "============================================================"
    log "实验数量: ${#EXPERIMENTS[@]}"
    log "Checkpoint steps: ${STEPS[*]}"
    log "GPU: ${CUDA_VISIBLE_DEVICES} (${n_gpus} GPUs)"
    log "LCB参数: n_samples=${lcb_n_samples}, prompt=${lcb_max_prompt_length}, response=${lcb_max_response_length} (8K)"
    log "输出目录: ${OUTPUT_SUBDIR}"
    log "============================================================"
    
    # 统计计数
    local lcb_total=0
    local lcb_completed=0
    local lcb_failed=0
    
    # LCB 评估
    for step in "${STEPS[@]}"; do
        log ""
        log "########################################################"
        log "LCB - 开始处理 Step ${step}"
        log "########################################################"
        
        for exp_name in "${EXPERIMENTS[@]}"; do
            ((lcb_total+=1))
            
            if process_checkpoint_lcb "$exp_name" "$step"; then
                ((lcb_completed+=1))
            else
                ((lcb_failed+=1))
            fi
            
            log ""
        done
    done
    
    # 打印最终统计
    log ""
    log "============================================================"
    log "🎉 LCB v6 评估完成!"
    log "============================================================"
    log "总任务: ${lcb_total}, 成功: ${lcb_completed}, 失败: ${lcb_failed}"
    log "日志文件: ${MAIN_LOG}"
    log "============================================================"
}

# 运行主流程
cd ${EVAL_DIR}
main






