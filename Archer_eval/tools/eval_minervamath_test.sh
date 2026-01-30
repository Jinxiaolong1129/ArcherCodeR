#!/bin/bash
#
# MinervaMAth 评估测试脚本
# 测试1: 16K Pass@1
# 测试2: 8K Pass@8
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

# ============ 通用配置 ============
nnodes=1
n_gpus=4
tp_size=1
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

base_dir=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval
data_dir=${base_dir}/data/test
dataset=minervamath

max_prompt_length=$((1024 * 2))      # 2K prompt
batch_size=2048
gpu_memory_utilization=0.9

project_name=MinervaMAth_Eval
experiment_name=DeepSeek-R1-Distill-Qwen-1.5B

# ============ 评估函数 ============
run_eval() {
    local n_samples=$1
    local max_response_length=$2
    local temperature=$3
    local top_p=$4
    local desc=$5
    
    local response_len_k=$((max_response_length / 1024))
    local output_dir=${base_dir}/output/eval_results/${dataset}_${response_len_k}k_n${n_samples}
    
    # 清理之前的结果
    rm -rf ${output_dir}
    mkdir -p ${output_dir}
    
    echo ""
    echo "============================================================"
    echo "MinervaMAth Evaluation - ${desc}"
    echo "============================================================"
    echo "Model: ${model_path}"
    echo "Dataset: ${data_dir}/${dataset}.parquet"
    echo "Output: ${output_dir}/${dataset}.parquet"
    echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${n_gpus} GPUs)"
    echo "n_samples: ${n_samples}"
    echo "max_response_length: ${max_response_length} (${response_len_k}K)"
    echo "temperature: ${temperature}"
    echo "top_p: ${top_p}"
    echo "============================================================"
    
    local start_time=$(date +%s)
    
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u AMD_VISIBLE_DEVICES \
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=${nnodes} \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=${project_name} \
        +trainer.experiment_name=${experiment_name} \
        +trainer.task_name=${dataset} \
        +trainer.global_step=0 \
        +trainer.use_wandb=False \
        model.path=${model_path} \
        data.path=${data_dir}/${dataset}.parquet \
        data.output_path=${output_dir}/${dataset}.parquet \
        data.batch_size=${batch_size} \
        data.n_samples=${n_samples} \
        rollout.name=vllm \
        rollout.gpu_memory_utilization=${gpu_memory_utilization} \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.disable_log_stats=False \
        rollout.tensor_model_parallel_size=${tp_size} \
        rollout.temperature=${temperature} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.prompt_length=${max_prompt_length} \
        rollout.response_length=${max_response_length} \
        rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "============================================================"
    echo "✓ ${desc} 完成!"
    echo "耗时: ${duration} 秒"
    echo "结果: ${output_dir}/${dataset}.parquet"
    echo "============================================================"
}

# ============ 主流程 ============
echo ""
echo "########################################################################"
echo "MinervaMAth 评估测试"
echo "########################################################################"
echo ""


# 测试2: 8K Pass@8 (Sampling)
run_eval 8 $((1024 * 8)) 0.6 0.95 "8K Pass@8 (Sampling)"

echo ""
echo "########################################################################"
echo "🎉 所有测试完成!"
echo "########################################################################"

