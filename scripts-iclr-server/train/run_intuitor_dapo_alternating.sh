#!/usr/bin/env bash
set -euo pipefail

# 🚀 Intuitor-DAPO Alternating Training Script
# 在 Intuitor 算法和 DAPO 训练方式之间交替

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
MODES="intuitor,dapo"
STEPS_PER_PHASE=50
START_MODE="intuitor"
TOTAL_EPOCHS=4
OUTPUT_DIR="./output/ArcherCodeR/Intuitor-DAPO-Alternating-$(date +%Y%m%d-%H%M%S)"
CONFIG_NAME="intuitor_dapo_alternating"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --modes)
            MODES="$2"
            shift 2
            ;;
        --steps-per-phase)
            STEPS_PER_PHASE="$2"
            shift 2
            ;;
        --start-with)
            START_MODE="$2"
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
        --help|-h)
            echo -e "${GREEN}🚀 Intuitor-DAPO Alternating Training Script${NC}"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --modes MODES             Comma-separated list of modes (default: intuitor,dapo)"
            echo "  --steps-per-phase STEPS   Steps per mode phase (default: 50)"
            echo "  --start-with MODE         Starting mode (default: intuitor)"
            echo "  --total-epochs EPOCHS     Total training epochs (default: 4)"
            echo "  --output-dir DIR          Output directory (default: auto-generated)"
            echo "  --config CONFIG           Config name (default: intuitor_dapo_alternating)"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default settings"
            echo "  $0 --steps-per-phase 25              # 25 steps per phase"
            echo "  $0 --modes dapo,intuitor              # Start with DAPO"
            echo "  $0 --start-with dapo --total-epochs 6 # Custom start and epochs"
            echo ""
            echo -e "${BLUE}Supported modes:${NC}"
            echo "  - intuitor: Self-certainty based algorithm"
            echo "  - dapo: Rejection sampling + external rewards"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate modes
IFS=',' read -ra MODE_ARRAY <<< "$MODES"
for mode in "${MODE_ARRAY[@]}"; do
    if [[ "$mode" != "intuitor" && "$mode" != "dapo" ]]; then
        echo -e "${RED}❌ Unsupported mode: $mode${NC}"
        echo -e "${BLUE}Supported modes: intuitor, dapo${NC}"
        exit 1
    fi
done

# Validate start mode
if [[ ! " ${MODE_ARRAY[@]} " =~ " ${START_MODE} " ]]; then
    echo -e "${RED}❌ Start mode '$START_MODE' not in modes list: $MODES${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/eval"

# Display configuration
echo -e "${GREEN}🚀 INTUITOR-DAPO ALTERNATING TRAINING CONFIGURATION${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "🎯 Modes: ${YELLOW}$MODES${NC}"
echo -e "🔄 Steps per phase: ${YELLOW}$STEPS_PER_PHASE${NC}"
echo -e "🎯 Starting mode: ${YELLOW}$START_MODE${NC}"
echo -e "📊 Total epochs: ${YELLOW}$TOTAL_EPOCHS${NC}"
echo -e "📁 Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "⚙️  Configuration: ${YELLOW}$CONFIG_NAME${NC}"
echo -e "🐍 Python: ${YELLOW}/data/xuandong_zhao/anaconda3/envs/archer/bin/python${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""
echo -e "${YELLOW}📝 Training Overview:${NC}"
echo -e "   🧠 Intuitor phases: Use model self-certainty as reward signal"
echo -e "   🎯 DAPO phases: Use rejection sampling + external rewards"
echo -e "   🔄 Switch every $STEPS_PER_PHASE steps"

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

# Convert modes to list format for Hydra
MODE_LIST="[$(echo "$MODES" | sed 's/,/,/g' | sed 's/\([^,]*\)/"\1"/g')]"

# Launch training
echo -e "${GREEN}🚀 Launching Intuitor-DAPO alternating training...${NC}"

/data/xuandong_zhao/anaconda3/envs/archer/bin/python -m verl.trainer.main_intuitor_dapo_alternating \
    --config-name="$CONFIG_NAME" \
    trainer.experiment_name="intuitor-dapo-alternating-$(date +%Y%m%d-%H%M%S)" \
    trainer.project_name=ArcherCodeR \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.validation_data_dir="$OUTPUT_DIR/eval" \
    alternating.modes="$MODE_LIST" \
    alternating.steps_per_phase="$STEPS_PER_PHASE" \
    alternating.start_mode="$START_MODE" \
    data.train_files=./data/train/archercoder-1.5b-train.json \
    data.val_files=./data/test/livecodebench_v5.json \
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
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}🎉 TRAINING SUMMARY${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "📋 Configuration used: ${YELLOW}$CONFIG_NAME${NC}"
echo -e "🎯 Modes: ${YELLOW}$MODES${NC}"
echo -e "📊 Steps per phase: ${YELLOW}$STEPS_PER_PHASE${NC}"
echo -e "📁 Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "📋 Full log: ${YELLOW}$OUTPUT_DIR/training.log${NC}"
echo -e "${BLUE}======================================================${NC}"

# Optional: Show recent checkpoints
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}📂 Recent checkpoints:${NC}"
    find "$OUTPUT_DIR" -name "global_step_*" -type d | sort -V | tail -3 | while read -r checkpoint; do
        echo -e "   📌 $(basename "$checkpoint")"
    done
fi

echo -e "${GREEN}🎯 Intuitor-DAPO alternating training completed! Check the output directory for results.${NC}"
