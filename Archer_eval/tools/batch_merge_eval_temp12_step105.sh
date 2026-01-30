#!/bin/bash
#
# TrajectoryEntropy-temp1.2 Step 105 的 Model Merge + 全量评估脚本
# 需要先 merge，然后运行全部 9 种评估
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

# ============ 目标实验 ============
EXP_NAME="Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2"
STEP=105

CKPT_PATH="${OUTPUT_ROOT}/${EXP_NAME}/global_step_${STEP}/actor"
HF_PATH="${CKPT_PATH}/hf_model"

# ============ 评估参数 ============
n_gpus=4
tp_size=1
temperature=0.6
top_p=0.95
batch_size=2048

# ============ 日志配置 ============
LOG_DIR=${EVAL_DIR}/tools/logs
mkdir -p ${LOG_DIR}
MAIN_LOG=${LOG_DIR}/batch_merge_eval_temp12_step105_$(date +%Y%m%d_%H%M%S).log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

# ============ Step 1: Model Merge ============
merge_model() {
    log "============================================================"
    log "Step 1: Model Merge"
    log "============================================================"
    
    if [ -d "$HF_PATH" ] && [ -f "$HF_PATH/config.json" ]; then
        log "✓ HF模型已存在，跳过合并"
        return 0
    fi
    
    log "🔧 开始合并模型..."
    
    $PYTHON -m tools.model_merge merge \
        --backend fsdp \
        --local_dir "${CKPT_PATH}" \
        --target_dir "${HF_PATH}" 2>&1 | tee -a ${MAIN_LOG}
    
    if [ -f "${HF_PATH}/config.json" ]; then
        log "✓ 模型合并成功"
        return 0
    else
        log "✗ 模型合并失败"
        return 1
    fi
}

# ============ Step 2: 运行评估 ============
run_eval() {
    local dataset=$1
    local data_file=$2
    local output_dir=$3
    local n_samples=$4
    local max_prompt=$5
    local max_response=$6
    
    local output_file="${output_dir}/${dataset}.parquet"
    
    if [ -f "$output_file" ]; then
        log "✓ ${dataset} 已存在，跳过"
        return 0
    fi
    
    mkdir -p "${output_dir}"
    
    log "📊 开始评估: ${dataset}"
    
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u AMD_VISIBLE_DEVICES \
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=ArcherCodeR_Eval \
        +trainer.experiment_name=${EXP_NAME} \
        +trainer.task_name=${dataset} \
        +trainer.global_step=${STEP} \
        +trainer.use_wandb=False \
        model.path=${HF_PATH} \
        data.path=${data_file} \
        data.output_path=${output_file} \
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
        rollout.prompt_length=${max_prompt} \
        rollout.response_length=${max_response} \
        rollout.max_num_batched_tokens=$((max_prompt + max_response)) \
        2>&1 | tee -a ${MAIN_LOG}
    
    if [ -f "$output_file" ]; then
        log "✓ ${dataset} 评估完成"
        return 0
    else
        log "✗ ${dataset} 评估失败"
        return 1
    fi
}

# ============ 主流程 ============
main() {
    log "============================================================"
    log "TrajectoryEntropy-temp1.2 Step 105 全量评估"
    log "实验: ${EXP_NAME}"
    log "Step: ${STEP}"
    log "============================================================"
    
    # 检查 checkpoint
    if [ ! -d "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH/config.json" ]; then
        log "✗ Checkpoint 不存在: ${CKPT_PATH}"
        exit 1
    fi
    
    # Step 1: Merge
    if ! merge_model; then
        log "✗ Merge 失败，退出"
        exit 1
    fi
    
    # Step 2: 评估
    log ""
    log "============================================================"
    log "Step 2: 运行全部 9 种评估"
    log "============================================================"
    
    local completed=0
    local failed=0
    
    # output/ 目录评估 (4种, 8K response)
    log ""
    log "--- output/ 目录 (8K response) ---"
    
    # AIME 2024 (8K)
    if run_eval "aime2024" "${DATA_DIR}/aime2024.json" "${HF_PATH}/output" 32 $((1024*2)) $((1024*8)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # AIME 2025 (8K)
    if run_eval "aime2025" "${DATA_DIR}/aime2025.json" "${HF_PATH}/output" 32 $((1024*2)) $((1024*8)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # LCB v5 (8K)
    if run_eval "livecodebench_v5" "${DATA_DIR}/livecodebench_v5.json" "${HF_PATH}/output" 8 $((1024*4)) $((1024*8)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # Minerva (8K)
    if run_eval "minervamath" "${DATA_DIR}/minervamath.parquet" "${HF_PATH}/output" 8 $((1024*2)) $((1024*8)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # output_16k/ 目录评估 (3种, 16K response)
    log ""
    log "--- output_16k/ 目录 (16K response) ---"
    
    # AIME 2024 (16K)
    if run_eval "aime2024" "${DATA_DIR}/aime2024.json" "${HF_PATH}/output_16k" 32 $((1024*2)) $((1024*16)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # AIME 2025 (16K)
    if run_eval "aime2025" "${DATA_DIR}/aime2025.json" "${HF_PATH}/output_16k" 32 $((1024*2)) $((1024*16)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # LCB v5 (16K)
    if run_eval "livecodebench_v5" "${DATA_DIR}/livecodebench_v5.json" "${HF_PATH}/output_16k" 8 $((1024*4)) $((1024*16)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # output_v6_8k/ 目录评估 (1种, 8K response)
    log ""
    log "--- output_v6_8k/ 目录 ---"
    
    if run_eval "livecodebench_v6" "${DATA_DIR}/livecodebench_v6.json" "${HF_PATH}/output_v6_8k" 8 $((1024*4)) $((1024*8)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    # output_v6_16k/ 目录评估 (1种, 16K response)
    log ""
    log "--- output_v6_16k/ 目录 ---"
    
    if run_eval "livecodebench_v6" "${DATA_DIR}/livecodebench_v6.json" "${HF_PATH}/output_v6_16k" 8 $((1024*4)) $((1024*16)); then
        ((completed++))
    else
        ((failed++))
    fi
    
    log ""
    log "============================================================"
    log "🎉 完成! 成功: ${completed}/9, 失败: ${failed}/9"
    log "============================================================"
}

cd ${EVAL_DIR}
main

