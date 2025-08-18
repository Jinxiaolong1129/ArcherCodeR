#!/bin/bash

echo "🚀 启动单节点Ray集群..."

# 🔧 首先清理所有GPU相关的冲突环境变量
echo "🧹 清理GPU环境变量冲突..."
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true
unset AMD_VISIBLE_DEVICES 2>/dev/null || true

# 确保这些变量不会被导出到子进程
export -n ROCR_VISIBLE_DEVICES 2>/dev/null || true
export -n HIP_VISIBLE_DEVICES 2>/dev/null || true
export -n AMD_VISIBLE_DEVICES 2>/dev/null || true

echo "✅ GPU环境变量冲突已清理"

# 设置训练所需的环境变量
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
export HF_TOKEN=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
export WANDB_API_KEY=5c271ef60b4c4753def92be733cf80487f0c7e78

# 停止已存在的Ray集群
echo "🛑 停止现有Ray集群..."
ray stop --force

# 清理Ray临时文件（可选，确保干净启动）
echo "🧹 清理Ray临时文件..."
rm -rf /tmp/ray* 2>/dev/null || true

# 再次确认GPU环境变量已清理
echo "🔍 最终环境变量检查："
env | grep -E "(CUDA|ROCR|HIP|AMD)_VISIBLE_DEVICES" || echo "✅ 没有GPU环境变量冲突"

# 启动单节点Ray集群
echo "🚀 启动Ray集群..."
ray start --head --port=6379 --dashboard-port=8265 --disable-usage-stats

echo "✅ 单节点Ray集群启动完成！"
echo "📊 Dashboard 地址: http://localhost:8265"
echo "📋 Ray集群信息:"
ray status