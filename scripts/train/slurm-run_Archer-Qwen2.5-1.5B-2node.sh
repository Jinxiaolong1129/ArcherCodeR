#!/usr/bin/env bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=512GB
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -xeuo pipefail

export HF_TOKEN=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
export WANDB_API_KEY=5c271ef60b4c4753def92be733cf80487f0c7e78

# ==============================================
# Ray Cluster Setup
# ==============================================

# Create log directory and files
JOB_ID=${SLURM_JOB_ID}
LOG_DIR="./logs/${JOB_ID}"
mkdir -p ${LOG_DIR}
RAY_LOG="${LOG_DIR}/ray_setup.log"

echo "=== Ray Cluster Setup Started at $(date) ===" | tee -a ${RAY_LOG}
echo "Job ID: ${JOB_ID}" | tee -a ${RAY_LOG}
echo "Allocated nodes: ${SLURM_JOB_NODELIST}" | tee -a ${RAY_LOG}

# Getting the node names
echo "Getting node information..." | tee -a ${RAY_LOG}
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Nodes array: ${nodes_array[@]}" | tee -a ${RAY_LOG}
echo "Head node: ${head_node}" | tee -a ${RAY_LOG}
echo "Head node IP (raw): ${head_node_ip}" | tee -a ${RAY_LOG}

# Handle potential IPv6/IPv4 mixed addresses
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPv6 address detected. Using IPv4 address: $head_node_ip" | tee -a ${RAY_LOG}
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "Final IP Head: $ip_head" | tee -a ${RAY_LOG}
echo "Ray dashboard will be available at: http://${head_node_ip}:8265" | tee -a ${RAY_LOG}

# Start Ray head node
echo "Starting Ray HEAD node at $head_node..." | tee -a ${RAY_LOG}
echo "Command: ray start --head --node-ip-address=\"$head_node_ip\" --port=$port --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE} --dashboard-host=0.0.0.0" | tee -a ${RAY_LOG}

# Method: Use srun to execute ray start and let it complete normally (no background &)
srun --nodes=1 --ntasks=1 -w "$head_node" \
    /data/xuandong_zhao/anaconda3/envs/archer/bin/ray start --head \
    --node-ip-address="$head_node_ip" \
    --port=$port \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${SLURM_GPUS_PER_NODE}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --disable-usage-stats \
    2>&1 | tee -a ${RAY_LOG}

echo "Ray head node start command completed, waiting for GCS to stabilize..." | tee -a ${RAY_LOG}

# Wait for Ray to stabilize (reduced time since we know it works fast)
sleep 20
echo "Wait completed, verifying head node status..." | tee -a ${RAY_LOG}


# Start Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Number of worker nodes to start: $worker_num" | tee -a ${RAY_LOG}

if [ $worker_num -gt 0 ]; then
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray WORKER $i at $node_i..." | tee -a ${RAY_LOG}
        echo "Command: ray start --address \"$ip_head\" --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE}" | tee -a ${RAY_LOG}
        
        # Remove background & and use full path to ray
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            /data/xuandong_zhao/anaconda3/envs/archer/bin/ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" \
            --disable-usage-stats \
            2>&1 | tee -a ${RAY_LOG}
        
        echo "Worker $i start command completed, waiting 5 seconds before next worker..." | tee -a ${RAY_LOG}
        sleep 5
    done
else
    echo "No worker nodes to start (single node setup)" | tee -a ${RAY_LOG}
fi

# Wait for all workers to connect (reduced time)
echo "Waiting 10 seconds for all workers to connect..." | tee -a ${RAY_LOG}
sleep 10

# Final verification of Ray cluster
echo "=== Final Ray Cluster Verification ===" | tee -a ${RAY_LOG}
echo "Running 'ray status' to verify complete cluster..." | tee -a ${RAY_LOG}
srun --nodes=1 --ntasks=1 -w "$head_node" /data/xuandong_zhao/anaconda3/envs/archer/bin/ray status 2>&1 | tee -a ${RAY_LOG}

echo "=== Ray Cluster Setup Completed at $(date) ===" | tee -a ${RAY_LOG}
echo "Ray cluster logs saved to: ${RAY_LOG}"

# ==============================================
# Training Configuration
# ==============================================

nnodes=2

project_name='ArcherCodeR'
exp_name='Archer-Qwen2.5-1.5B-2node'

adv_estimator=grpo

# kl config
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# clip
clip_ratio_low=0.2
clip_ratio_high=0.2
loss_agg_mode=token-mean

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))
enable_overlong_buffer=False
overlong_buffer_len=16
overlong_penalty_factor=1.0
v_max_response_length=$((1024 * 32))

train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=32

# Paths
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
CKPTS_DIR=./output/${project_name}/${exp_name}
data_dir=./data
TRAIN_FILE=$data_dir/train/archercoder-1.5b-train.json
TEST_FILE=$data_dir/test/livecodebench_v5.json

# Algorithm
n_resp_per_prompt=16
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
v_n=4
v_temperature=0.8
v_top_p=1.0
v_top_k=-1

# Performance Related Parameter
sp_size=1
gen_tp=1
use_dynamic_bsz=False
micro_batch_size_per_gpu=1
actor_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + v_max_response_length))
offload=False

# Token Mask
use_token_entropy_separate=True
token_entropy_quantile=0.8
high_entropy_kl_loss_scale_coef=0.0
low_entropy_clip_ratio_low=0.2
low_entropy_clip_ratio_high=0.2
high_entropy_clip_ratio_low=0.5
high_entropy_clip_ratio_high=0.5

# Trainer
use_overlong_filter=False

mkdir -p ${CKPTS_DIR}

# ==============================================
# Start Training on Ray Cluster
# ==============================================

echo "Starting training on Ray cluster..."

# Run training on head node with Ray cluster
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    /data/xuandong_zhao/anaconda3/envs/archer/bin/python -m dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    +actor_rollout_ref.actor.use_token_entropy_separate=${use_token_entropy_separate} \
    +actor_rollout_ref.actor.high_entropy_kl_loss_scale_coef=${high_entropy_kl_loss_scale_coef} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_low=${low_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_high=${low_entropy_clip_ratio_high} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_low=${high_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_high=${high_entropy_clip_ratio_high} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + v_max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + v_max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${v_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${v_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${v_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${v_n} \
    +actor_rollout_ref.rollout.val_kwargs.response_length=${v_max_response_length} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=wizard \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${nnodes}" \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.max_actor_ckpt_to_keep=3 \
    +trainer.max_critic_ckpt_to_keep=3 \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    +trainer.enable_overlong_filter=${use_overlong_filter} \
    +trainer.rejection_sample=True $@ 2>&1 | tee ${CKPTS_DIR}/${project_name}_${exp_name}_grpo.log

# ==============================================
# Cleanup
# ==============================================

echo "Training completed. Stopping Ray cluster..."

# Stop Ray on all nodes (use full path)
for node in "${nodes_array[@]}"; do
    echo "Stopping Ray on $node"
    srun --nodes=1 --ntasks=1 -w "$node" /data/xuandong_zhao/anaconda3/envs/archer/bin/ray stop &
done

wait

echo "Ray cluster stopped. Job finished."