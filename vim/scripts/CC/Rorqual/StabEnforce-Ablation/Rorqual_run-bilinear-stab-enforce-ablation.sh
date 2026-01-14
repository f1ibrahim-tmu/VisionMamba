#!/bin/bash
# Feature-StabEnforce: Run ablation study for stability enforcement with Bilinear discretization
# This script runs all 8 configurations with multiple seeds for statistical significance

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS=(0 1 2 3 4)  # 5 different seeds for Mean ± Std calculation

# All 8 configurations
CONFIGS=(
    "baseline"
    "sn"
    "ec"
    "sp"
    "sn-ec"
    "sn-sp"
    "ec-sp"
    "all"
)

echo "=========================================="
echo "Stability Enforcement Ablation Study"
echo "Discretization Method: Bilinear (Tustin)"
echo "=========================================="
echo "Configurations: ${CONFIGS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

# Function to run a configuration with all seeds
run_config_seeds() {
    local CONFIG=$1
    local SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-bilinear-stab-${CONFIG}.sh"
    
    if [ ! -f "$SCRIPT" ]; then
        echo "Error: Script not found: $SCRIPT"
        return 1
    fi
    
    echo "=========================================="
    echo "Configuration: $CONFIG (Bilinear)"
    echo "=========================================="
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Running $CONFIG (Bilinear) with seed: $SEED"
        echo "----------------------------------------"
        
        bash "$SCRIPT" $SEED
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed $CONFIG (Bilinear) seed $SEED"
        else
            echo "✗ Failed $CONFIG (Bilinear) seed $SEED"
        fi
        echo ""
    done
    
    echo "Completed all seeds for $CONFIG (Bilinear)"
    echo ""
}

# Run all configurations with multiple seeds
for CONFIG in "${CONFIGS[@]}"; do
    run_config_seeds "$CONFIG"
done

echo "=========================================="
echo "All ablation study runs completed!"
echo "Discretization Method: Bilinear (Tustin)"
echo "=========================================="
echo ""
echo "Next step: Run extract_results.py to compute Mean ± Std"
echo "Example: python ../extract_results.py --base-dir ./output/classification_logs"
