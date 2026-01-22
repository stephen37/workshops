#!/bin/bash
# =============================================================================
# Run all LoRA training variants
# Execute this on Runpod after setup
# =============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

CONFIGS_DIR="/workspace/lora_training/configs"
LOG_DIR="/workspace/lora_training/logs"
TOOLKIT_DIR="/workspace/ai-toolkit"

mkdir -p "$LOG_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  LoRA Training - All Variants${NC}"
echo -e "${GREEN}========================================${NC}"

# Count configs
CONFIG_COUNT=$(ls -1 "$CONFIGS_DIR"/*.yaml 2>/dev/null | wc -l)
echo -e "Found ${YELLOW}$CONFIG_COUNT${NC} configs to train"
echo ""

# List what we're about to train
echo "Training queue:"
for config in "$CONFIGS_DIR"/*.yaml; do
    name=$(basename "$config" .yaml)
    echo "  - $name"
done
echo ""

# Confirm
read -p "Start training all $CONFIG_COUNT variants? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Track progress
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Train each config
for config in "$CONFIGS_DIR"/*.yaml; do
    name=$(basename "$config" .yaml)
    log_file="$LOG_DIR/${name}.log"

    echo -e "\n${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}Training: $name${NC}"
    echo -e "${GREEN}Config: $config${NC}"
    echo -e "${GREEN}Log: $log_file${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"

    # Run training
    cd "$TOOLKIT_DIR"

    if python run.py "$config" 2>&1 | tee "$log_file"; then
        echo -e "${GREEN}✓ Completed: $name${NC}"
        ((COMPLETED++))
    else
        echo -e "${RED}✗ Failed: $name${NC}"
        ((FAILED++))
    fi
done

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Completed: ${GREEN}$COMPLETED${NC}"
echo -e "Failed:    ${RED}$FAILED${NC}"
echo -e "Duration:  ${HOURS}h ${MINUTES}m"
echo -e "Outputs:   /workspace/lora_outputs/"
echo ""
