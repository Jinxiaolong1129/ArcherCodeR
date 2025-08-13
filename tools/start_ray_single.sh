#!/bin/bash

echo "启动单节点Ray集群..."

# 设置环境变量
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_MIN_NCHANNELS=16
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1

# Check if sensitive environment variables are set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY environment variable is not set"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set"
fi

export WANDB_API_KEY=$WANDB_API_KEY
export HF_TOKEN=$HF_TOKEN

# 停止已存在的Ray集群
ray stop

# 启动单节点Ray集群
ray start --head --port=6379 --dashboard-port=8265

echo "单节点Ray集群启动完成！"
echo "Dashboard 地址: http://localhost:8265"
echo "Ray集群信息:"
ray status 


# bash tools/start_ray_single.sh