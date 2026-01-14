#!/bin/bash
# Feature-StabEnforce: Run ablation study for stability enforcement
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
echo "=========================================="
echo "Configurations: ${CONFIGS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

# Function to run a configuration with all seeds
run_config_seeds() {
    local CONFIG=$1
    local SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-zoh-stab-${CONFIG}.sh"
    
    if [ ! -f "$SCRIPT" ]; then
        echo "Error: Script not found: $SCRIPT"
        return 1
    fi
    
    echo "=========================================="
    echo "Configuration: $CONFIG"
    echo "=========================================="
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Running $CONFIG with seed: $SEED"
        echo "----------------------------------------"
        
        bash "$SCRIPT" $SEED
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed $CONFIG seed $SEED"
        else
            echo "✗ Failed $CONFIG seed $SEED"
        fi
        echo ""
    done
    
    echo "Completed all seeds for $CONFIG"
    echo ""
}

# Run all configurations with multiple seeds
for CONFIG in "${CONFIGS[@]}"; do
    run_config_seeds "$CONFIG"
done

echo "=========================================="
echo "All ablation study runs completed!"
echo "=========================================="
echo ""
echo "Next step: Run extract_results.py to compute Mean ± Std"
echo "Example: python ../extract_results.py --base-dir ./output/classification_logs"
