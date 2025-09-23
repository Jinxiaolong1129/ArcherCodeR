#!/usr/bin/env bash
set -euo pipefail

# 🚀 Alternating Training Examples
# 展示不同使用场景的示例脚本

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 ALTERNATING TRAINING EXAMPLES${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

echo -e "${YELLOW}Available examples:${NC}"
echo -e "  1️⃣  Quick Test (1-step switching, 1000 samples)"
echo -e "  2️⃣  Standard Training (50-step phases, no KL)"
echo -e "  3️⃣  KL Loss Training (25-step phases, KL=0.05)"
echo -e "  4️⃣  Custom Project (different project name)"
echo -e "  5️⃣  GRPO First (start with GRPO instead of Intuitor)"
echo ""

read -p "$(echo -e ${YELLOW}Select example [1-5]: ${NC})" choice

case $choice in
    1)
        echo -e "${GREEN}🧪 Running Quick Test...${NC}"
        ./scripts/train/run_alternating_unified.sh \
            --test-mode \
            --dataset-limit 1000 \
            --config alternating_test \
            --project-name "ArcherCodeR-QuickTest" \
            --total-epochs 2
        ;;
    2)
        echo -e "${GREEN}📊 Running Standard Training...${NC}"
        ./scripts/train/run_alternating_unified.sh \
            --steps-per-phase 50 \
            --kl-mode no-kl \
            --project-name "ArcherCodeR-Standard" \
            --total-epochs 4
        ;;
    3)
        echo -e "${GREEN}🔧 Running KL Loss Training...${NC}"
        ./scripts/train/run_alternating_unified.sh \
            --steps-per-phase 25 \
            --kl-mode kl005 \
            --project-name "ArcherCodeR-KL005" \
            --total-epochs 6
        ;;
    4)
        echo -e "${GREEN}🏷️  Running Custom Project...${NC}"
        ./scripts/train/run_alternating_unified.sh \
            --project-name "MyCustomProject" \
            --steps-per-phase 30 \
            --kl-mode no-kl \
            --total-epochs 4
        ;;
    5)
        echo -e "${GREEN}🎯 Running GRPO First...${NC}"
        ./scripts/train/run_alternating_unified.sh \
            --algorithms "grpo,intuitor" \
            --start-with grpo \
            --steps-per-phase 40 \
            --project-name "ArcherCodeR-GRPOFirst" \
            --total-epochs 4
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✅ Example completed!${NC}"
