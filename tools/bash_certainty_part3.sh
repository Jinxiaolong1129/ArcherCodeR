#!/bin/bash
# Part 3: TrajectoryEntropy-n8 (6) + TrajectoryEntropy-temp1.2 (25) + Pure-GRPO (15) = 46 files
# Usage: bash tools/bash_certainty_part3.sh

set -e
cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

mkdir -p logs

# Clear environment
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

ray stop --force 2>/dev/null || true

export PYTHONPATH=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR:$PYTHONPATH
PYTHON="/data/xuandong_zhao/anaconda3/envs/archer/bin/python"

# Experiments to process in Part 3
EXPERIMENTS=(
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp1.2-v2"
)

BASE_DIR="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"

process_file() {
    local PARQUET_PATH="$1"
    local FILENAME=$(basename "$PARQUET_PATH")
    local DIRNAME=$(dirname "$PARQUET_PATH")
    local SUBDIR=$(basename "$DIRNAME")
    
    # Determine response_length based on directory
    if [[ "$SUBDIR" == *"16k"* ]]; then
        RESP_LEN=16384
    else
        RESP_LEN=8192
    fi
    
    # Determine prompt_length based on dataset type
    # AIME: 2048 (1024*2), LCB: 4096 (1024*4)
    if [[ "$FILENAME" == *"aime"* ]]; then
        PROMPT_LEN=2048
    else
        PROMPT_LEN=4096
    fi
    
    # Model path
    MODEL_PATH=$(dirname "$(dirname "$PARQUET_PATH")")
    OUTPUT_PATH="${PARQUET_PATH%.parquet}_certainty.parquet"
    
    # Skip if already done
    if [[ -f "$OUTPUT_PATH" ]]; then
        echo "SKIP (already done): $PARQUET_PATH"
        return
    fi
    
    echo ""
    echo "============================================================"
    echo "Processing: $FILENAME"
    echo "  Model: $MODEL_PATH"
    echo "  Prompt length: $PROMPT_LEN"
    echo "  Response length: $RESP_LEN"
    echo "============================================================"
    
    $PYTHON -m verl.trainer.compute_certainty \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        model.path="$MODEL_PATH" \
        data.input_path="$PARQUET_PATH" \
        data.output_path="$OUTPUT_PATH" \
        data.batch_size=32 \
        rollout.prompt_length=$PROMPT_LEN \
        rollout.response_length=$RESP_LEN \
        rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor.ppo_mini_batch_size=16 \
        actor.ppo_micro_batch_size_per_gpu=1
    
    ray stop --force 2>/dev/null || true
    sleep 5
}

# Main loop
for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "################################################################"
    echo "Processing experiment: $EXP"
    echo "################################################################"
    
    PARQUET_FILES=$(find "$BASE_DIR/$EXP" -name "*.parquet" -type f 2>/dev/null | grep -E "(aime2024|livecodebench)" | grep -v "_certainty" | sort)
    
    for PARQUET_PATH in $PARQUET_FILES; do
        process_file "$PARQUET_PATH"
    done
done

echo ""
echo "============================================================"
echo "Part 3 Complete!"
echo "============================================================"






