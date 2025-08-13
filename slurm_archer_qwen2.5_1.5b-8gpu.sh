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
#SBATCH --output=slurm_archer_qwen2.5_1.5b_single_out_%j.txt
#SBATCH --error=slurm_archer_qwen2.5_1.5b_single_error_%j.txt
#SBATCH --job-name=archer-qwen2.5-1.5b-single

echo "Starting Archer Qwen2.5 1.5B Single Node training job"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "Date: $(date)"


# conda create -n archer python=3.10 -y
# conda activate archer


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

echo "Step 1: Downloading datasets..."
/data/xuandong_zhao/anaconda3/envs/archer/bin/python tools/download_datasets.py

echo "Step 2: Starting Ray cluster..."
bash ./tools/start_ray_single.sh

echo "Step 3: Starting training..."
bash ./scripts/train/run_archer_qwen2.5_1.5b_code-8gpu.sh

# Capture the exit status
EXIT_STATUS=$?

echo "Training completed with exit status: $EXIT_STATUS"
echo "End time: $(date)"

# Clean up Ray cluster
ray stop

if [ $EXIT_STATUS -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit status: $EXIT_STATUS"
    exit $EXIT_STATUS
fi

# Usage: sbatch slurm_archer_qwen2.5_1.5b-8gpu.sh