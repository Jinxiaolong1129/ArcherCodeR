#!/bin/bash

# 使用SLURM环境变量获取节点信息
if [ -n "$SLURM_JOB_NODELIST" ]; then
    # SLURM环境：从SLURM_JOB_NODELIST解析节点列表
    echo "SLURM环境检测到，节点列表: $SLURM_JOB_NODELIST"
    
    # 解析SLURM节点列表并获取IP地址
    NODE_LIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    NODE_ARRAY=($NODE_LIST)
    
    # 获取第一个节点作为head节点的IP
    HEAD_NODE=${NODE_ARRAY[0]}
    HEAD_IP=$(getent hosts $HEAD_NODE | awk '{print $1}')
    
    # 获取worker节点IP列表
    WORKER_IPS=()
    for ((i=1; i<${#NODE_ARRAY[@]}; i++)); do
        WORKER_NODE=${NODE_ARRAY[i]}
        WORKER_IP=$(getent hosts $WORKER_NODE | awk '{print $1}')
        WORKER_IPS+=($WORKER_IP)
    done
    
elif [ -f "/etc/mpi/hostfile" ]; then
    # MPI环境：使用hostfile
    HOSTFILE="/etc/mpi/hostfile"
    echo "MPI环境检测到，使用hostfile: $HOSTFILE"
    
    # 解析 Head 节点 IP（取第一个IP）
    HEAD_IP=$(head -n 1 "$HOSTFILE" | awk '{print $1}')
    
    # 解析 Worker 节点 IP（排除第一行后取所有IP）
    mapfile -t WORKER_IPS < <(tail -n +2 "$HOSTFILE" | awk '{print $1}')
else
    echo "错误：既没有SLURM环境变量也没有MPI hostfile"
    exit 1
fi

# 打印解析结果
echo "Head 节点: $HEAD_IP"
echo "Worker 节点: ${WORKER_IPS[*]}"

# 启动 Head 节点
echo "正在启动 Head 节点 ($HEAD_IP)..."

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 检测GPU
echo "检测Head节点GPU..."
nvidia-smi -L || echo "警告: nvidia-smi 不可用"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

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
export WANDB_API_KEY=$WANDB_API_KEY

# 停止任何现有的Ray进程
ray stop

# 启动Head节点，确保GPU可见
ray start --head --port=6379 --dashboard-port=8265 --node-ip-address=$HEAD_IP --num-gpus=8
HEAD_ADDRESS="$HEAD_IP:6379"

# 等待Head节点完全启动
echo "等待Head节点启动完成..."
sleep 10

# 启动Worker节点
echo "启动Worker节点..."
for WORKER_IP in "${WORKER_IPS[@]}"; do
    echo "正在启动Worker节点: $WORKER_IP"
    
    # 测试SSH连接
    echo "测试SSH连接到 $WORKER_IP..."
    if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$WORKER_IP" "echo 'SSH连接成功到 $WORKER_IP'" 2>/dev/null; then
        echo "✅ SSH连接到 $WORKER_IP 成功"
    else
        echo "❌ SSH连接到 $WORKER_IP 失败"
        continue
    fi
    
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$WORKER_IP" bash -l <<EOF &
echo "========== Worker节点 $WORKER_IP 启动日志 =========="
echo "当前时间: \$(date)"
echo "当前工作目录: \$(pwd)"
echo "切换到工作目录: $PWD"
cd $PWD
echo "新工作目录: \$(pwd)"

echo "加载环境..."
source ~/.bashrc
source /data/xuandong_zhao/anaconda3/etc/profile.d/conda.sh
conda activate archer

echo "验证环境..."
echo "Python路径: \$(which python)"
echo "Ray路径: \$(which ray)"

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 检测Worker节点GPU
echo "检测Worker节点GPU..."
nvidia-smi -L || echo "警告: nvidia-smi 不可用"
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"

echo "设置环境变量..."
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
export WANDB_API_KEY=$WANDB_API_KEY

# 停止任何现有的Ray进程
echo "停止现有Ray进程..."
ray stop

# 连接到Head节点，明确指定GPU数量
echo "Worker节点 $WORKER_IP 连接到 $HEAD_ADDRESS"
echo "执行命令: ray start --address="$HEAD_ADDRESS" --node-ip-address=$WORKER_IP --num-gpus=8"
ray start --address="$HEAD_ADDRESS" --node-ip-address=$WORKER_IP --num-gpus=8

if [ \$? -eq 0 ]; then
    echo "✅ Worker节点 $WORKER_IP 启动成功"
else
    echo "❌ Worker节点 $WORKER_IP 启动失败"
fi

echo "========== Worker节点 $WORKER_IP 启动完成 =========="
EOF

done

# 等待所有Worker节点启动完成
echo "等待所有Worker节点启动完成..."
wait

# 额外等待时间确保集群完全就绪
sleep 20

# 检查Ray集群状态
echo "检查Ray集群状态..."
ray status

echo "Ray 集群启动完成！"
echo "Dashboard 地址: http://$HEAD_IP:8265"

# 验证集群中的节点数量 - 增加重试机制
EXPECTED_NODES=$((1 + ${#WORKER_IPS[@]}))
echo "期望节点数: $EXPECTED_NODES"

# 等待所有节点注册到Ray集群
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "检查Ray集群状态 (尝试 $((RETRY_COUNT + 1))/$MAX_RETRIES)..."
    
    # 获取当前Ray状态
    RAY_STATUS=$(ray status)
    echo "$RAY_STATUS"
    
    # 计算节点数量
    ACTUAL_NODES=$(echo "$RAY_STATUS" | grep -c "node_" || echo "0")
    
    # 计算GPU数量 - 改进解析方法
    # 首先尝试解析 "0.0/16.0 GPU" 格式
    TOTAL_GPUS=$(echo "$RAY_STATUS" | grep -o "[0-9]*\.[0-9]*/[0-9]*\.[0-9]* GPU" | head -1 | cut -d'/' -f2 | cut -d' ' -f1 | cut -d'.' -f1)
    
    # 如果上面的方法没有找到GPU，尝试其他格式
    if [ -z "$TOTAL_GPUS" ] || [ "$TOTAL_GPUS" -eq 0 ]; then
        TOTAL_GPUS=$(echo "$RAY_STATUS" | grep -o "GPU: [0-9]*" | awk '{sum += $2} END {print sum+0}')
    fi
    
    # 如果还是没有找到，设置为0
    if [ -z "$TOTAL_GPUS" ]; then
        TOTAL_GPUS=0
    fi
    
    echo "当前状态: $ACTUAL_NODES 个节点, $TOTAL_GPUS 个GPU"
    
    if [ "$ACTUAL_NODES" -eq "$EXPECTED_NODES" ] && [ "$TOTAL_GPUS" -eq 16 ]; then
        echo "✅ Ray集群完全就绪: $ACTUAL_NODES 个节点, $TOTAL_GPUS 个GPU"
        break
    else
        echo "⏳ Ray集群还未完全就绪，等待..."
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 15
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ 警告: Ray集群在 $((MAX_RETRIES * 15)) 秒后仍未完全就绪"
    echo "当前状态: $ACTUAL_NODES 个节点 (期望 $EXPECTED_NODES), $TOTAL_GPUS 个GPU (期望 16)"
    echo "继续执行，但可能会遇到资源不足的问题"
fi

# 最终状态检查
echo "========== 最终Ray集群状态 =========="
ray status