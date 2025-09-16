#!/usr/bin/env bash
set -euo pipefail

# 🚀 Unified Alternating Training Script
# 支持 Intuitor 和 GRPO 算法的交替训练

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ALGORITHMS="intuitor,grpo"
STEPS_PER_PHASE=50
START_ALGORITHM="intuitor"
TOTAL_EPOCHS=4
PROJECT_NAME="ArcherCodeR"
KL_MODE="no-kl"  # Options: no-kl, kl005, kl01, etc.
TEST_MODE=false
DATASET_LIMIT=""
OUTPUT_DIR=""  # Will be generated based on parameters
CONFIG_NAME="alternating_official"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        --steps-per-phase)
            STEPS_PER_PHASE="$2"
            shift 2
            ;;
        --start-with)
            START_ALGORITHM="$2"
            shift 2
            ;;
        --total-epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --kl-mode)
            KL_MODE="$2"
            shift 2
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        --dataset-limit)
            DATASET_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo -e "${GREEN}🚀 Unified Alternating Training Script${NC}"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --algorithms ALGOS        Comma-separated list of algorithms (default: intuitor,grpo)"
            echo "  --steps-per-phase STEPS   Steps per algorithm phase (default: 50)"
            echo "  --start-with ALGO         Starting algorithm (default: intuitor)"
            echo "  --total-epochs EPOCHS     Total training epochs (default: 4)"
            echo "  --output-dir DIR          Output directory (default: auto-generated)"
            echo "  --config CONFIG           Config name (default: alternating_official)"
            echo "  --project-name PROJECT    Wandb project name (default: ArcherCodeR)"
            echo "  --kl-mode MODE            KL loss mode: no-kl, kl005, kl01 (default: no-kl)"
            echo "  --test-mode               Enable test mode (1-step switching, limited data)"
            echo "  --dataset-limit N         Limit dataset to N samples (for testing)"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default settings"
            echo "  $0 --steps-per-phase 25              # 25 steps per phase"
            echo "  $0 --algorithms grpo,intuitor         # Start with GRPO"
            echo "  $0 --start-with grpo --total-epochs 6 # Custom start and epochs"
            echo "  $0 --kl-mode kl005 --project-name MyProject # With KL loss"
            echo "  $0 --test-mode --dataset-limit 1000  # Test mode with limited data"
            echo ""
            echo -e "${BLUE}Supported algorithms: intuitor, grpo${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate algorithms
IFS=',' read -ra ALGO_ARRAY <<< "$ALGORITHMS"
for algo in "${ALGO_ARRAY[@]}"; do
    if [[ "$algo" != "intuitor" && "$algo" != "grpo" ]]; then
        echo -e "${RED}❌ Unsupported algorithm: $algo${NC}"
        echo -e "${BLUE}Supported algorithms: intuitor, grpo${NC}"
        exit 1
    fi
done

# Validate start algorithm
if [[ ! " ${ALGO_ARRAY[@]} " =~ " ${START_ALGORITHM} " ]]; then
    echo -e "${RED}❌ Start algorithm '$START_ALGORITHM' not in algorithms list: $ALGORITHMS${NC}"
    exit 1
fi

# Validate KL mode
case "$KL_MODE" in
    no-kl|kl005|kl01|kl02|kl05)
        ;;
    *)
        echo -e "${RED}❌ Unsupported KL mode: $KL_MODE${NC}"
        echo -e "${BLUE}Supported KL modes: no-kl, kl005, kl01, kl02, kl05${NC}"
        exit 1
        ;;
esac

# Generate output directory if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    ALGO_STR=$(echo "$ALGORITHMS" | tr ',' '-')
    
    if [[ "$TEST_MODE" == "true" ]]; then
        OUTPUT_DIR="./output/${PROJECT_NAME}/Test-${ALGO_STR}-${KL_MODE}-steps${STEPS_PER_PHASE}-${TIMESTAMP}"
        if [[ -n "$DATASET_LIMIT" ]]; then
            OUTPUT_DIR="${OUTPUT_DIR}-limit${DATASET_LIMIT}"
        fi
    else
        OUTPUT_DIR="./output/${PROJECT_NAME}/Alternating-${ALGO_STR}-${KL_MODE}-steps${STEPS_PER_PHASE}-epochs${TOTAL_EPOCHS}-${TIMESTAMP}"
    fi
fi

# Adjust parameters for test mode
if [[ "$TEST_MODE" == "true" ]]; then
    STEPS_PER_PHASE=1  # Switch every step for testing
    if [[ -z "$DATASET_LIMIT" ]]; then
        DATASET_LIMIT=1000  # Default limit for test mode
    fi
    if [[ "$TOTAL_EPOCHS" -gt 2 ]]; then
        TOTAL_EPOCHS=2  # Limit epochs in test mode
    fi
    echo -e "${YELLOW}🧪 Test mode enabled: 1-step switching, limited to ${DATASET_LIMIT} samples${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/eval"

# Display configuration
echo -e "${GREEN}🚀 UNIFIED ALTERNATING TRAINING CONFIGURATION${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "📋 Algorithms: ${YELLOW}$ALGORITHMS${NC}"
echo -e "🔄 Steps per phase: ${YELLOW}$STEPS_PER_PHASE${NC}"
echo -e "🎯 Starting algorithm: ${YELLOW}$START_ALGORITHM${NC}"
echo -e "📊 Total epochs: ${YELLOW}$TOTAL_EPOCHS${NC}"
echo -e "🏷️  Project name: ${YELLOW}$PROJECT_NAME${NC}"
echo -e "🔧 KL mode: ${YELLOW}$KL_MODE${NC}"
if [[ "$TEST_MODE" == "true" ]]; then
    echo -e "🧪 Test mode: ${YELLOW}ENABLED${NC}"
    if [[ -n "$DATASET_LIMIT" ]]; then
        echo -e "📊 Dataset limit: ${YELLOW}$DATASET_LIMIT${NC}"
    fi
fi
echo -e "📁 Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "⚙️  Configuration: ${YELLOW}$CONFIG_NAME${NC}"
echo -e "🐍 Python: ${YELLOW}/home/ec2-user/miniconda3/envs/archer/bin/python${NC}"
echo -e "${BLUE}=================================================${NC}"

# Confirm execution
echo -e "${YELLOW}⏳ Starting training in 3 seconds... (Ctrl+C to cancel)${NC}"
sleep 3

# 导入环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}✅ Loaded environment variables from .env${NC}"
    echo -e "🔑 WANDB_API_KEY: ${YELLOW}${WANDB_API_KEY:0:8}...${NC}"
    echo -e "🔑 HF_TOKEN: ${YELLOW}${HF_TOKEN:0:8}...${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: .env file not found. Please create .env file with WANDB_API_KEY and HF_TOKEN${NC}"
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Convert algorithms to list format for Hydra
ALGO_LIST="[$(echo "$ALGORITHMS" | sed 's/,/,/g' | sed 's/\([^,]*\)/"\1"/g')]"

# Launch training
echo -e "${GREEN}🚀 Launching unified alternating training...${NC}"

# Build experiment name with key parameters
EXPERIMENT_NAME="alternating-$(echo "$ALGORITHMS" | tr ',' '-')-${KL_MODE}-steps${STEPS_PER_PHASE}"
if [[ "$TEST_MODE" == "true" ]]; then
    EXPERIMENT_NAME="test-${EXPERIMENT_NAME}"
fi

# Build additional parameters for the training command
EXTRA_PARAMS=""
if [[ -n "$DATASET_LIMIT" ]]; then
    EXTRA_PARAMS="$EXTRA_PARAMS data.dataset_limit=$DATASET_LIMIT"
fi

# Add KL mode configuration
case "$KL_MODE" in
    kl005)
        EXTRA_PARAMS="$EXTRA_PARAMS alternating.kl_mode=kl005"
        ;;
    kl01)
        EXTRA_PARAMS="$EXTRA_PARAMS alternating.kl_mode=kl01"
        ;;
    kl02)
        EXTRA_PARAMS="$EXTRA_PARAMS alternating.kl_mode=kl02"
        ;;
    kl05)
        EXTRA_PARAMS="$EXTRA_PARAMS alternating.kl_mode=kl05"
        ;;
    no-kl)
        EXTRA_PARAMS="$EXTRA_PARAMS alternating.kl_mode=no-kl"
        ;;
esac

/home/ec2-user/miniconda3/envs/archer/bin/python -m verl.trainer.main_alternating \
    --config-name="$CONFIG_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.validation_data_dir="$OUTPUT_DIR/eval" \
    alternating.algorithms="$ALGO_LIST" \
    alternating.steps_per_phase="$STEPS_PER_PHASE" \
    alternating.start_algorithm="$START_ALGORITHM" \
    alternating.test_mode="$TEST_MODE" \
    data.train_files=./data/train/archercoder-1.5b-train.json \
    data.val_files=./data/test/livecodebench_v5.json \
    $EXTRA_PARAMS \
    "$@" 2>&1 | tee "$OUTPUT_DIR/training.log"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "📁 Results saved to: ${YELLOW}$OUTPUT_DIR${NC}"
    echo -e "📋 Training log: ${YELLOW}$OUTPUT_DIR/training.log${NC}"
else
    echo -e "${RED}❌ Training failed with exit code: $EXIT_CODE${NC}"
    echo -e "📋 Check log for details: ${YELLOW}$OUTPUT_DIR/training.log${NC}"
    exit $EXIT_CODE
fi

# Display summary
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}🎉 TRAINING SUMMARY${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "📋 Configuration used: ${YELLOW}$CONFIG_NAME${NC}"
echo -e "🔄 Algorithms: ${YELLOW}$ALGORITHMS${NC}"
echo -e "📊 Steps per phase: ${YELLOW}$STEPS_PER_PHASE${NC}"
echo -e "📁 Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "📋 Full log: ${YELLOW}$OUTPUT_DIR/training.log${NC}"
echo -e "${BLUE}=================================================${NC}"

# Optional: Show recent checkpoints
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}📂 Recent checkpoints:${NC}"
    find "$OUTPUT_DIR" -name "global_step_*" -type d | sort -V | tail -3 | while read -r checkpoint; do
        echo -e "   📌 $(basename "$checkpoint")"
    done
fi

echo -e "${GREEN}🎯 Training completed! Check the output directory for results.${NC}"



# bash scripts/train/run_alternating_unified.sh     --test-mode     --dataset-limit 500     --kl-mode no-kl