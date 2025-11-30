#!/bin/bash
# Run training with multiple random seeds for all discretization methods
# This script runs each method with 5 different seeds (0-4) for statistical significance

SEEDS=(0 1 2 3 4)  # 5 different seeds for Mean ± Std calculation

echo "=========================================="
echo "Running Multiple Seeds Experiment"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

# Function to run a method with all seeds
run_method_seeds() {
    local METHOD=$1
    local SCRIPT=$2
    
    echo "=========================================="
    echo "Method: $METHOD"
    echo "=========================================="
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Running $METHOD with seed: $SEED"
        echo "----------------------------------------"
        
        bash "$SCRIPT" $SEED
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed $METHOD seed $SEED"
        else
            echo "✗ Failed $METHOD seed $SEED"
        fi
        echo ""
    done
    
    echo "Completed all seeds for $METHOD"
    echo ""
}

# Run all methods with multiple seeds
echo "Running Zero Order Hold (ZOH) with multiple seeds..."
run_method_seeds "ZOH" "pt-cifar100-vim-zoh.sh"

echo "Running First Order Hold (FOH) with multiple seeds..."
run_method_seeds "FOH" "pt-cifar100-vim-foh.sh"

echo "Running Bilinear (Tustin) Transform with multiple seeds..."
run_method_seeds "Bilinear" "pt-cifar100-vim-bilinear.sh"

echo "Running Polynomial Interpolation with multiple seeds..."
run_method_seeds "Polynomial" "pt-cifar100-vim-poly.sh"

echo "Running Higher-Order Hold with multiple seeds..."
run_method_seeds "HighOrder" "pt-cifar100-vim-highorder.sh"

echo "Running Runge-Kutta 4th Order (RK4) with multiple seeds..."
run_method_seeds "RK4" "pt-cifar100-vim-rk4.sh"

echo "=========================================="
echo "All training runs with multiple seeds completed!"
echo "=========================================="
echo ""
echo "Next step: Run extract_results.py to compute Mean ± Std"

