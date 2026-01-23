#!/bin/bash
#
# Compute certainty metrics for existing responses
# This script skips vLLM generation and only uses FSDP Actor for forward pass
#

set -e

# 使用 archer 环境的 Python
PYTHON=/data/xuandong_zhao/anaconda3/envs/archer/bin/python
export PYTHONPATH=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR:$PYTHONPATH

# ============ 配置 ============
nnodes=1
n_gpus=8  # 使用的 GPU 数量

# 默认参数 - 请根据需要修改
INPUT_PATH=${1:-"/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp1.2-v2/global_step_105/actor/hf_model/output/aime2024.parquet"}
MODEL_PATH=${2:-"/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp1.2-v2/global_step_105/actor/hf_model"}
OUTPUT_DIR=$(dirname ${INPUT_PATH})
OUTPUT_PATH=${OUTPUT_DIR}/$(basename ${INPUT_PATH} .parquet)_certainty.parquet

# 处理参数
batch_size=32
max_prompt_length=$((1024 * 2))      # 2K prompt
max_response_length=$((1024 * 8))    # 8K response

echo "============================================================"
echo "Compute Certainty Metrics"
echo "============================================================"
echo "Input: ${INPUT_PATH}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "GPU: ${n_gpus} GPUs"
echo "batch_size: ${batch_size}"
echo "max_prompt_length: ${max_prompt_length}"
echo "max_response_length: ${max_response_length}"
echo "============================================================"

# 运行 certainty 计算
$PYTHON -m verl.trainer.compute_certainty \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus} \
    model.path=${MODEL_PATH} \
    data.input_path=${INPUT_PATH} \
    data.output_path=${OUTPUT_PATH} \
    data.batch_size=${batch_size} \
    rollout.prompt_length=${max_prompt_length} \
    rollout.response_length=${max_response_length} \
    rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor.ppo_mini_batch_size=${batch_size} \
    actor.ppo_micro_batch_size_per_gpu=2

echo ""
echo "============================================================"
echo "✓ Certainty computation complete!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "============================================================"


