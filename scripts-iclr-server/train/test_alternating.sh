#!/usr/bin/env bash
set -euo pipefail

# 🧪 Alternating Training Test Script
# 快速测试交替训练功能

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🧪 ALTERNATING TRAINING TEST${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "${YELLOW}This script will run a quick test of the alternating training system:${NC}"
echo -e "  📊 Dataset limited to 1000 samples"
echo -e "  🔄 Algorithm switches every 1 step"
echo -e "  📈 Only 2 epochs"
echo -e "  🎯 Tests both Intuitor and GRPO"
echo -e "${BLUE}================================${NC}"
echo ""

# Ask for confirmation
read -p "$(echo -e ${YELLOW}Continue with test? [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Test cancelled${NC}"
    exit 1
fi

# Run the test
echo -e "${GREEN}🚀 Starting alternating training test...${NC}"

./scripts/train/run_alternating_unified.sh \
    --test-mode \
    --dataset-limit 1000 \
    --config alternating_test \
    --project-name "ArcherCodeR-Test" \
    --total-epochs 2 \
    --algorithms "intuitor,grpo" \
    --kl-mode "no-kl"

echo -e "${GREEN}✅ Test completed!${NC}"
