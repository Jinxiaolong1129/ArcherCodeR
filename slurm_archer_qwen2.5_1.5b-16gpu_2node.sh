#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=256GB
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=slurm_archer_qwen2.5_1.5b_16gpu_2node_out_%j.txt
#SBATCH --error=slurm_archer_qwen2.5_1.5b_16gpu_2node_error_%j.txt
#SBATCH --job-name=archer-qwen2.5-1.5b-16gpu-2node

echo "Starting Archer Qwen2.5 1.5B 2-Node training job"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "Date: $(date)"

# 初始化conda环境
source ~/.bashrc
source /data/xuandong_zhao/anaconda3/etc/profile.d/conda.sh

# 激活conda环境
conda activate archer
echo "Activated conda environment: $(conda info --envs | grep '*')"

# 验证Python路径
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# 确保路径正确
export PATH=/data/xuandong_zhao/anaconda3/envs/archer/bin:$PATH

# 设置CUDA环境变量
echo "设置CUDA环境..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# 验证GPU可见性
echo "验证GPU可见性..."
nvidia-smi -L || echo "警告: 无法检测到GPU"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "检测到的GPU数量: $(nvidia-smi -L | wc -l)"

# 设置SSH密钥（如果需要）
echo "设置SSH连接..."
# 确保SSH agent正在运行
eval $(ssh-agent -s) 2>/dev/null || true

# 测试节点间连接
if [[ $SLURM_PROCID -eq 0 ]] && [ -n "$SLURM_JOB_NODELIST" ]; then
    echo "测试节点间网络连接..."
    NODE_LIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    for node in $NODE_LIST; do
        if [ "$node" != "$(hostname)" ]; then
            echo "测试到 $node 的连接..."
            ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$node" "echo '连接到 $node 成功'" 2>/dev/null || echo "警告: 无法SSH到 $node"
        fi
    done
fi

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Download flash-attention wheel file if it doesn't exist
FLASH_ATTN_WHEEL="flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
    echo "Downloading flash-attention wheel file..."
    wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/$FLASH_ATTN_WHEEL
else
    echo "Flash-attention wheel file already exists, skipping download."
fi
pip install --no-cache-dir $FLASH_ATTN_WHEEL

pip install -e .


# Enable debug mode
set -xeuo pipefail

# Clear conflicting AMD GPU environment variables
unset ROCR_VISIBLE_DEVICES

# Set ulimits and environment variables
ulimit -n 1048576 2>/dev/null || echo "Warning: Could not set ulimit for open files (insufficient permissions)"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

export HF_TOKEN=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
export WANDB_API_KEY=5c271ef60b4c4753def92be733cf80487f0c7e78

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. Continuing without loading .env"
fi

# Only run setup tasks on the main node
if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "Step 1: Downloading datasets..."
    /data/xuandong_zhao/anaconda3/envs/archer/bin/python tools/download_datasets.py

    echo "Step 2: Starting Ray cluster..."
    bash ./tools/start_ray.sh
    
    # 检查Ray集群状态
    echo "验证Ray集群状态..."
    ray status
    
    # 等待额外时间确保所有节点完全就绪
    echo "等待Ray集群完全就绪..."
    sleep 30
    
    # 再次检查Ray集群状态
    echo "最终Ray集群状态："
    ray status
    
    # 检查GPU数量 - 修复解析逻辑
    # 首先尝试解析 "0.0/16.0 GPU" 格式
    TOTAL_GPUS=$(ray status | grep -o "[0-9]*\.[0-9]*/[0-9]*\.[0-9]* GPU" | head -1 | cut -d'/' -f2 | cut -d' ' -f1 | cut -d'.' -f1)
    
    # 如果上面的方法没有找到GPU，尝试其他格式
    if [ -z "$TOTAL_GPUS" ] || [ "$TOTAL_GPUS" -eq 0 ]; then
        TOTAL_GPUS=$(ray status | grep -o "GPU: [0-9]*" | awk '{sum += $2} END {print sum+0}')
    fi
    
    # 如果还是没有找到，设置为0
    if [ -z "$TOTAL_GPUS" ]; then
        TOTAL_GPUS=0
    fi
    
    echo "Ray集群中检测到的总GPU数量: $TOTAL_GPUS"
    
    # 验证GPU数量是否符合预期
    if [ "$TOTAL_GPUS" -eq 16 ]; then
        echo "✅ GPU数量正确: $TOTAL_GPUS 个GPU"
    else
        echo "⚠️  警告: GPU数量不符合预期 (期望16个, 实际$TOTAL_GPUS个)"
    fi
    
else
    echo "Worker node $SLURM_PROCID waiting for setup to complete..."
    # 增加等待时间让Ray集群完全启动
    echo "等待Ray集群设置完成..."
    sleep 120  # 增加到2分钟
    
    # 在worker节点上也检查Ray状态
    echo "Worker节点检查Ray集群状态..."
    ray status || echo "Worker节点无法检查Ray状态，这是正常的"
fi

echo "Step 3: Starting training..."
if [[ $SLURM_PROCID -eq 0 ]]; then
    echo "主节点开始训练..."
    bash ./scripts/train/run_archer_qwen2.5_1.5b_code_2node.sh
    
    # Capture the exit status
    EXIT_STATUS=$?
    
    echo "Training completed with exit status: $EXIT_STATUS"
    echo "End time: $(date)"
    
    # 创建一个文件来通知worker节点训练完成
    echo "$EXIT_STATUS" > /tmp/training_complete_${SLURM_JOB_ID}
    
else
    # Worker nodes wait for training to complete
    echo "Worker节点等待训练完成..."
    while [ ! -f /tmp/training_complete_${SLURM_JOB_ID} ]; do
        echo "等待主节点完成训练..."
        sleep 30
    done
    
    # 读取训练结果
    EXIT_STATUS=$(cat /tmp/training_complete_${SLURM_JOB_ID})
    echo "训练完成，退出状态: $EXIT_STATUS"
fi

# Clean up Ray cluster on all nodes
echo "清理Ray集群..."
ray stop

# 清理临时文件
if [[ $SLURM_PROCID -eq 0 ]]; then
    # 清理训练状态文件
    rm -f /tmp/training_complete_${SLURM_JOB_ID}
    
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "Training completed successfully!"
    else
        echo "Training failed with exit status: $EXIT_STATUS"
        exit $EXIT_STATUS
    fi
fi

echo "作业清理完成，节点 $SLURM_PROCID 退出"

# Usage: sbatch slurm_archer_qwen2.5_1.5b-8gpu.sh

# sbatch --nodelist=compute-891 run_simple_nccl_test.sh