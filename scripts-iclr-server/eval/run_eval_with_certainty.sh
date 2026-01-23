#!/bin/bash
set -x

export PYTHONPATH=`pwd`:$PYTHONPATH

# ============ Cluster Configuration ============
nnodes=1
n_gpus_per_node=8
tp_size=1

# ============ Generation Configuration ============
n_samples=1          # Number of responses per prompt (1 for greedy eval)
temperature=0.0      # 0.0 for greedy decoding
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

# ============ Path Configuration ============
base_dir=.

# Model checkpoint path (change this to your checkpoint)
# Example: ./output/ArcherCodeR/Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64/global_step_100/actor
model_path=${1:-"${base_dir}/model/WizardCodeR-1.5B-DAPO"}

# Output directory
output_dir=${model_path}/eval_certainty

# Dataset
data_dir=${base_dir}/data/test
dataset=${2:-"livecodebench_v5"}

# Training method (determines which certainty metric to use for classification)
# Options: INTUITOR, TRAJECTORY_ENTROPY, TOKEN_ENTROPY, PROB_DISPARITY
training_method=${3:-"INTUITOR"}

# ============ Derived Configuration ============
max_model_len=$((max_prompt_length + max_response_length))

# ============ Memory Configuration ============
# Since we run both vLLM AND FSDP, we need to be careful with memory
# gpu_memory_utilization for vLLM (lower than usual)
gpu_memory_utilization=0.4

# Batch sizes (smaller than usual)
batch_size=128
log_prob_micro_batch_size_per_gpu=4

echo "============================================"
echo "Evaluation with Certainty Analysis"
echo "============================================"
echo "Model: ${model_path}"
echo "Dataset: ${dataset}"
echo "Training Method: ${training_method}"
echo "Output: ${output_dir}/${dataset}.parquet"
echo "============================================"

# Create output directory
mkdir -p ${output_dir}

# Run evaluation
python -m verl.trainer.main_generation_with_certainty \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    model.path=${model_path} \
    data.path=${data_dir}/${dataset}.parquet \
    data.output_path=${output_dir}/${dataset}.parquet \
    data.batch_size=${batch_size} \
    data.n_samples=${n_samples} \
    data.trust_remote_code=true \
    analysis.training_method=${training_method} \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    rollout.enforce_eager=True \
    rollout.free_cache_engine=True \
    rollout.tensor_model_parallel_size=${tp_size} \
    rollout.temperature=${temperature} \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=${max_prompt_length} \
    rollout.response_length=${max_response_length} \
    rollout.max_num_batched_tokens=${max_model_len} \
    rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor.fsdp_config.model_dtype=bfloat16

echo "============================================"
echo "Evaluation Complete!"
echo "Results saved to: ${output_dir}/${dataset}_with_certainty.parquet"
echo "Summary saved to: ${output_dir}/${dataset}_with_certainty_summary.json"
echo "============================================"

