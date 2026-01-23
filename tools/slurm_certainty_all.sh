#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256GB
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/logs/slurm_certainty_all_%j.txt
#SBATCH --error=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/logs/slurm_certainty_all_error_%j.txt
#SBATCH --job-name=cert-all
#SBATCH --exclusive

# All experiments combined - 8 GPUs, single node
# Experiments: TokenEntropy + TrajectoryEntropy (all variants) + Pure-GRPO

mkdir -p /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/logs

# Clear environment
unset RAY_ADDRESS
unset RAY_HEAD_NODE_HOST
export RAY_DISABLE_IMPORT_WARNING=1
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

cd /data/xuandong_zhao/mnt/xiaolong/ArcherCodeR

ray stop --force 2>/dev/null || true

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset AMD_VISIBLE_DEVICES

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "All Certainty Experiments - 8 GPUs"
echo "Environment check:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"

export PYTHONPATH=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR:$PYTHONPATH
PYTHON="/data/xuandong_zhao/anaconda3/envs/archer/bin/python"

# All experiments to process
EXPERIMENTS=(
    "Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp0.8"
    "Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp0.8"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n8"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-temp1.2"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp1.2-v2"
)

BASE_DIR="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"

# Counter for processed files
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
TOTAL_TIME=0
FAILED_FILES=()

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
        # livecodebench (v5 or v6)
        PROMPT_LEN=4096
    fi
    
    # Model path
    MODEL_PATH=$(dirname "$(dirname "$PARQUET_PATH")")
    OUTPUT_PATH="${PARQUET_PATH%.parquet}_certainty.parquet"
    
    # Skip if already done
    if [[ -f "$OUTPUT_PATH" ]]; then
        echo "SKIP (already done): $PARQUET_PATH"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        return 0
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo ""
    echo "============================================================"
    echo "Processing: $(basename $PARQUET_PATH)"
    echo "  Model: $MODEL_PATH"
    echo "  Prompt length: $PROMPT_LEN"
    echo "  Response length: $RESP_LEN"
    echo "  Start time: $START_TIME_STR"
    echo "============================================================"
    
    # Run with 8 GPUs
    $PYTHON -m verl.trainer.compute_certainty \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        model.path="$MODEL_PATH" \
        data.input_path="$PARQUET_PATH" \
        data.output_path="$OUTPUT_PATH" \
        data.batch_size=64 \
        rollout.prompt_length=$PROMPT_LEN \
        rollout.response_length=$RESP_LEN \
        rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor.ppo_mini_batch_size=32 \
        actor.ppo_micro_batch_size_per_gpu=1
    
    EXIT_CODE=$?
    
    # Record end time and calculate duration
    END_TIME=$(date +%s)
    END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    TOTAL_TIME=$((TOTAL_TIME + DURATION))
    
    echo ""
    echo "------------------------------------------------------------"
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "SUCCESS: $(basename $PARQUET_PATH)"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_FILES+=("$PARQUET_PATH")
        echo "FAILED (exit code $EXIT_CODE): $(basename $PARQUET_PATH)"
        echo "  >> Skipping to next file..."
    fi
    
    echo "  End time: $END_TIME_STR"
    echo "  Duration: ${DURATION_MIN}m ${DURATION_SEC}s (${DURATION}s total)"
    echo "  Success: $SUCCESS_COUNT | Failed: $FAILED_COUNT | Skipped: $SKIPPED_COUNT"
    echo "  Total time so far: $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
    echo "------------------------------------------------------------"
    
    # Stop ray between runs to clean up
    ray stop --force 2>/dev/null || true
    sleep 5
    
    return 0  # Always return success to continue loop
}

# Main loop
JOB_START_TIME=$(date +%s)
JOB_START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "================================================================"
echo "Job started at: $JOB_START_TIME_STR"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "================================================================"

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "################################################################"
    echo "Processing experiment: $EXP"
    echo "################################################################"
    
    # Find all parquet files for this experiment
    PARQUET_FILES=$(find "$BASE_DIR/$EXP" -name "*.parquet" -type f 2>/dev/null | grep -E "(aime2024|livecodebench)" | grep -v "_certainty" | sort)
    
    for PARQUET_PATH in $PARQUET_FILES; do
        process_file "$PARQUET_PATH"
    done
done

JOB_END_TIME=$(date +%s)
JOB_END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
JOB_DURATION=$((JOB_END_TIME - JOB_START_TIME))

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "  Job started: $JOB_START_TIME_STR"
echo "  Job ended: $JOB_END_TIME_STR"
echo "  Total duration: $((JOB_DURATION / 3600))h $((JOB_DURATION % 3600 / 60))m $((JOB_DURATION % 60))s"
echo ""
echo "  Summary:"
echo "    Success: $SUCCESS_COUNT"
echo "    Failed:  $FAILED_COUNT"
echo "    Skipped: $SKIPPED_COUNT"
echo "============================================================"

if [[ ${#FAILED_FILES[@]} -gt 0 ]]; then
    echo ""
    echo "FAILED FILES:"
    for f in "${FAILED_FILES[@]}"; do
        echo "  - $f"
    done
fi

