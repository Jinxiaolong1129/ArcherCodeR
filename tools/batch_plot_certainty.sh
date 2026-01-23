#!/bin/bash
# Batch plot certainty distributions for all *_certainty.parquet files

set -e

BASE_DIR="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"
SCRIPT="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/tools/plot_certainty_distribution.py"
PYTHON="/data/xuandong_zhao/anaconda3/envs/archer/bin/python"

echo "========================================"
echo "Batch Plot Certainty Distributions"
echo "========================================"
echo "Base directory: $BASE_DIR"
echo ""

# Find all *_certainty.parquet files
FILES=$(find "$BASE_DIR" -name "*_certainty.parquet" -type f 2>/dev/null | sort)

if [ -z "$FILES" ]; then
    echo "No *_certainty.parquet files found!"
    exit 1
fi

TOTAL=$(echo "$FILES" | wc -l)
echo "Found $TOTAL certainty parquet files"
echo ""

# Process each file
COUNT=0
SUCCESS=0
FAILED=0

for FILE in $FILES; do
    COUNT=$((COUNT + 1))
    
    # Get relative path for display
    REL_PATH=${FILE#$BASE_DIR/}
    PLOT_DIR=$(dirname "$FILE")/plots
    
    echo "----------------------------------------"
    echo "[$COUNT/$TOTAL] Processing: $REL_PATH"
    
    # Check if plots already exist
    if [ -d "$PLOT_DIR" ] && [ "$(ls -A $PLOT_DIR 2>/dev/null)" ]; then
        echo "  Plots already exist in: $PLOT_DIR"
        echo "  Skipping... (delete plots/ to regenerate)"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi
    
    # Run plotting script
    START_TIME=$(date +%s)
    
    if $PYTHON "$SCRIPT" --input "$FILE" --bins 15; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  ✓ Completed in ${DURATION}s"
        echo "  Plots saved to: $PLOT_DIR"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ Failed to generate plots"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Total files:    $TOTAL"
echo "Successful:     $SUCCESS"
echo "Failed:         $FAILED"
echo "========================================"

