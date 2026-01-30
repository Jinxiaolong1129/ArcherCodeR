#!/bin/bash
#
# 多卡 GPU 评估脚本 - Pass@1
# Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Dataset: MinervaMAth (272 problems)
#

set -e

# ============ 环境配置 ============
# 设置 CUDA 设备 (使用 GPU 0, 1, 2, 3)
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

# ============ 配置 ============
nnodes=1
n_gpus=4   # 使用 4 张 GPU
tp_size=1  # 数据并行模式 (每张GPU独立处理，速度更快)

# Model: 直接使用 HuggingFace 模型
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 数据路径
base_dir=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval
data_dir=${base_dir}/data/test
dataset=minervamath

# 评估参数 (MinervaMAth math task - Pass@1)
n_samples=1          # Pass@1 只需要 1 个回答
temperature=0.0      # Greedy decoding for deterministic results
top_p=1.0            # 不使用 top_p sampling
max_prompt_length=$((1024 * 2))      # 2K prompt
max_response_length=$((1024 * 8))    # 8K response
batch_size=2048        # MinervaMAth有272题，适当增大batch_size
gpu_memory_utilization=0.9

# 输出路径 (包含响应长度标识)
response_len_k=$((max_response_length / 1024))
output_dir=${base_dir}/output/eval_results/${dataset}_${response_len_k}k_n${n_samples}
# 删除之前的结果
rm -rf ${output_dir}
mkdir -p ${output_dir}

# 项目配置
project_name=MinervaMAth_Eval
experiment_name=DeepSeek-R1-Distill-Qwen-1.5B

echo "============================================================"
echo "MinervaMAth Evaluation - Pass@1"
echo "============================================================"
echo "Model: ${model_path}"
echo "Dataset: ${data_dir}/${dataset}.parquet"
echo "Output: ${output_dir}/${dataset}.parquet"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${n_gpus} GPUs, Data Parallel)"
echo "n_samples: ${n_samples} (Pass@1)"
echo "batch_size: ${batch_size}"
echo "max_prompt_length: ${max_prompt_length}"
echo "max_response_length: ${max_response_length}"
echo "temperature: ${temperature} (Greedy)"
echo "gpu_memory_utilization: ${gpu_memory_utilization}"
echo "============================================================"

# 运行评估
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

echo ""
echo "============================================================"
echo "✓ Evaluation complete!"
echo "Results saved to: ${output_dir}/${dataset}.parquet"
echo "============================================================"

