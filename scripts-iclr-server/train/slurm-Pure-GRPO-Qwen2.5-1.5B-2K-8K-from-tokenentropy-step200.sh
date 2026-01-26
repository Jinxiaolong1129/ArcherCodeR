#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=512GB
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=./output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-tokenentropy-step200/slurm_out_%j.txt
#SBATCH --error=./output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-tokenentropy-step200/slurm_error_%j.txt
#SBATCH --job-name=grpo-from-tokenentropy-step200


# 导入环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env"
    echo "Your WANDB_API_KEY is: $WANDB_API_KEY"
    echo "Your HF_TOKEN is: $HF_TOKEN"
else
    echo "Warning: .env file not found. Please create .env file with WANDB_API_KEY and HF_TOKEN"
fi


# Clear Ray environment variables to force local cluster creation
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1

# Clear AMD GPU environment variables to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

# 创建输出目录
mkdir -p ./output/ArcherCodeR/Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-tokenentropy-step200

# ============================================
# Step 1: Merge FSDP checkpoint to HF format
# ============================================
TOKENENTROPY_CHECKPOINT_PATH=./output/ArcherCodeR/Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple/global_step_200

# Check if hf_model already exists
if [ -d "${TOKENENTROPY_CHECKPOINT_PATH}/actor/hf_model" ]; then
    echo "✅ hf_model already exists, skipping merge"
else
    echo "🔄 Merging FSDP checkpoint to HuggingFace format..."
    /data/xuandong_zhao/anaconda3/envs/archer/bin/python -m tools.model_merge merge \
        --backend fsdp \
        --local_dir ${TOKENENTROPY_CHECKPOINT_PATH}/actor \
        --target_dir ${TOKENENTROPY_CHECKPOINT_PATH}/actor/hf_model
    echo "✅ Model merge completed"
fi

# ============================================
# Step 2: Start training
# ============================================
ray stop --force 2>/dev/null || true

bash scripts-iclr-server/train/run_Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-tokenentropy-step200.sh




