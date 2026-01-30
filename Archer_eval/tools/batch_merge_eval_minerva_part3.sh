#!/bin/bash
#
# 批量模型合并与 MinervaMAth 评估脚本 (Part 3 - 4 GPU)
# 8K 输出长度, Pass@8
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

# 清除 AMD GPU 环境变量
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
n_samples=8              # pass@8
temperature=0.6
top_p=0.95
max_prompt_length=$((1024 * 2))       # 2K prompt
max_response_length=$((1024 * 8))     # 8K response
batch_size=2048

# ============ 日志配置 ============
LOG_DIR=${EVAL_DIR}/tools/logs
mkdir -p ${LOG_DIR}
MAIN_LOG=${LOG_DIR}/batch_merge_eval_minerva_part3_$(date +%Y%m%d_%H%M%S).log

# ============ 实验列表 Part 3 ============
EXPERIMENTS=(
    "Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp0.8"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp0.8"
)

# Checkpoint steps
STEPS=(10 50 80 100 105)

# ============ 辅助函数 ============

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

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

check_eval_exists() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local result_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model/output/${dataset}.parquet"
    
    if [ -f "$result_path" ]; then
        return 0
    else
        return 1
    fi
}

merge_model() {
    local exp_name=$1
    local step=$2
    local ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
    local hf_path="${ckpt_path}/hf_model"
    
    log "🔧 开始合并模型: ${exp_name} step ${step}"
    
    if check_hf_model_exists "$exp_name" "$step"; then
        log "✓ HF模型已存在，跳过合并: ${hf_path}"
        return 0
    fi
    
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
        [ -d "${hf_path}" ] && rm -rf "${hf_path}"
        return 1
    fi
}

run_eval() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local model_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    local output_dir="${model_path}/output"
    
    log "📊 开始MinervaMAth评估: ${exp_name} step ${step} - ${dataset}"
    
    if check_eval_exists "$exp_name" "$step" "$dataset"; then
        log "✓ 评估结果已存在，跳过: ${output_dir}/${dataset}.parquet"
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
        +trainer.task_name=${dataset} \
        +trainer.global_step=${step} \
        +trainer.use_wandb=False \
        model.path=${model_path} \
        data.path=${DATA_DIR}/${dataset}.parquet \
        data.output_path=${output_dir}/${dataset}.parquet \
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
    
    if [ $exit_code -eq 0 ] && [ -f "${output_dir}/${dataset}.parquet" ]; then
        log "✓ MinervaMAth评估完成 (耗时: ${duration}s): ${exp_name} step ${step} - ${dataset}"
        return 0
    else
        log "✗ MinervaMAth评估失败: ${exp_name} step ${step} - ${dataset}"
        return 1
    fi
}

process_checkpoint() {
    local exp_name=$1
    local step=$2
    
    log "============================================================"
    log "处理: ${exp_name} - global_step_${step}"
    log "============================================================"
    
    if ! check_checkpoint_exists "$exp_name" "$step"; then
        log "⚠ Checkpoint 不存在，跳过: ${exp_name} step ${step}"
        return 0
    fi
    
    if ! merge_model "$exp_name" "$step"; then
        log "⚠ 合并失败，跳过评估: ${exp_name} step ${step}"
        return 1
    fi
    
    run_eval "$exp_name" "$step" "minervamath"
    
    log "✓ 完成处理: ${exp_name} - global_step_${step}"
}

# ============ 主流程 ============

main() {
    log "============================================================"
    log "MinervaMAth 评估脚本 (Part 3 - 4 GPU)"
    log "8K 输出长度, Pass@8"
    log "============================================================"
    log "实验数量: ${#EXPERIMENTS[@]}"
    log "Checkpoint steps: ${STEPS[*]}"
    log "GPU: ${CUDA_VISIBLE_DEVICES} (${n_gpus} GPUs)"
    log "n_samples: ${n_samples} (pass@8)"
    log "max_response_length: ${max_response_length} (8K)"
    log "============================================================"
    
    local total_tasks=0
    local completed_tasks=0
    local failed_tasks=0
    
    for step in "${STEPS[@]}"; do
        log ""
        log "########################################################"
        log "开始处理 Step ${step}"
        log "########################################################"
        
        for exp_name in "${EXPERIMENTS[@]}"; do
            ((total_tasks+=1))
            
            if process_checkpoint "$exp_name" "$step"; then
                ((completed_tasks+=1))
            else
                ((failed_tasks+=1))
            fi
            
            log ""
        done
    done
    
    log "============================================================"
    log "🎉 评估完成!"
    log "============================================================"
    log "总任务数: ${total_tasks}"
    log "成功: ${completed_tasks}"
    log "失败: ${failed_tasks}"
    log "日志文件: ${MAIN_LOG}"
    log "============================================================"
}

cd ${EVAL_DIR}
main

