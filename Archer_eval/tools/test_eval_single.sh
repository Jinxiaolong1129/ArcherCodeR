#!/bin/bash
#
# 测试脚本 - 单个实验评估 (4 GPU) - pass@1 快速测试版
# 实验: Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12
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

# 关键修复: 彻底清除 AMD GPU 环境变量，避免与 CUDA 冲突
# 必须在 ray stop 之前 unset，确保新 ray 进程不会继承这些变量
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

# 停止可能已有的 Ray 进程
ray stop --force 2>/dev/null || true

# 再次确认清除（有时 ray stop 后环境变量会重置）
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

echo "Environment check:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
echo "  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"

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

# ============ 单个实验配置 ============
exp_name="Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
step=100  # 使用 step 100 测试

# ============ 日志配置 ============
LOG_DIR=${EVAL_DIR}/tools/logs
mkdir -p ${LOG_DIR}
MAIN_LOG=${LOG_DIR}/test_eval_single_$(date +%Y%m%d_%H%M%S).log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

cd ${EVAL_DIR}

log "============================================================"
log "测试评估脚本"
log "实验: ${exp_name}"
log "Step: ${step}"
log "GPU: ${CUDA_VISIBLE_DEVICES}"
log "============================================================"

# 检查 checkpoint
ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
hf_path="${ckpt_path}/hf_model"
output_dir="${hf_path}/output"

log "Checkpoint路径: ${ckpt_path}"

if [ ! -d "$ckpt_path" ] || [ ! -f "$ckpt_path/config.json" ]; then
    log "❌ Checkpoint不存在: ${ckpt_path}"
    log "可用的steps:"
    ls -la "${OUTPUT_ROOT}/${exp_name}/" 2>/dev/null || echo "实验目录不存在"
    exit 1
fi

log "✓ Checkpoint存在"

# 合并模型
if [ -d "$hf_path" ] && [ -f "$hf_path/config.json" ]; then
    log "✓ HF模型已存在，跳过合并"
else
    log "🔧 开始合并模型..."
    $PYTHON -m tools.model_merge merge \
        --backend fsdp \
        --local_dir "${ckpt_path}" \
        --target_dir "${hf_path}" 2>&1 | tee -a ${MAIN_LOG}
    
    if [ ! -f "${hf_path}/config.json" ]; then
        log "❌ 模型合并失败"
        exit 1
    fi
    log "✓ 模型合并成功"
fi

mkdir -p "${output_dir}"

# ========== 评估 AIME2025 ==========
log ""
log "📊 开始评估 AIME2025..."

if [ -f "${output_dir}/aime2025.parquet" ]; then
    log "✓ aime2025 结果已存在，跳过"
else
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u AMD_VISIBLE_DEVICES \
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=ArcherCodeR_Eval \
        +trainer.experiment_name=${exp_name} \
        +trainer.task_name=aime2025 \
        +trainer.global_step=${step} \
        +trainer.use_wandb=False \
        model.path=${hf_path} \
        data.path=${DATA_DIR}/aime2025.json \
        data.output_path=${output_dir}/aime2025.parquet \
        data.batch_size=${batch_size} \
        data.n_samples=1 \
        rollout.name=vllm \
        rollout.gpu_memory_utilization=0.9 \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.disable_log_stats=False \
        rollout.tensor_model_parallel_size=${tp_size} \
        rollout.temperature=${temperature} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.prompt_length=$((1024 * 2)) \
        rollout.response_length=$((1024 * 8)) \
        rollout.max_num_batched_tokens=$((1024 * 10)) \
        2>&1 | tee -a ${MAIN_LOG}
    
    if [ -f "${output_dir}/aime2025.parquet" ]; then
        log "✓ AIME2025 评估完成"
    else
        log "❌ AIME2025 评估失败"
    fi
fi

# ========== 评估 LiveCodeBench ==========
log ""
log "📊 开始评估 LiveCodeBench v5..."

if [ -f "${output_dir}/livecodebench_v5.parquet" ]; then
    log "✓ livecodebench_v5 结果已存在，跳过"
else
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u AMD_VISIBLE_DEVICES \
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=ArcherCodeR_Eval \
        +trainer.experiment_name=${exp_name} \
        +trainer.task_name=livecodebench_v5 \
        +trainer.global_step=${step} \
        +trainer.use_wandb=False \
        model.path=${hf_path} \
        data.path=${DATA_DIR}/livecodebench_v5.json \
        data.output_path=${output_dir}/livecodebench_v5.parquet \
        data.batch_size=${batch_size} \
        data.n_samples=1 \
        rollout.name=vllm \
        rollout.gpu_memory_utilization=0.9 \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.disable_log_stats=False \
        rollout.tensor_model_parallel_size=${tp_size} \
        rollout.temperature=${temperature} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.prompt_length=$((1024 * 4)) \
        rollout.response_length=$((1024 * 8)) \
        rollout.max_num_batched_tokens=$((1024 * 12)) \
        2>&1 | tee -a ${MAIN_LOG}
    
    if [ -f "${output_dir}/livecodebench_v5.parquet" ]; then
        log "✓ LiveCodeBench v5 评估完成"
    else
        log "❌ LiveCodeBench v5 评估失败"
    fi
fi

log ""
log "============================================================"
log "🎉 测试完成!"
log "输出目录: ${output_dir}"
log "日志文件: ${MAIN_LOG}"
log "============================================================"

