#!/bin/bash
# Run a single discretization method with multiple random seeds
# Usage: ./run-single-method-seeds.sh <method> [seeds]
# Example: ./run-single-method-seeds.sh zoh
# Example: ./run-single-method-seeds.sh highorder 0,1,2,3,4

if [ $# -lt 1 ]; then
    echo "Usage: $0 <method> [seeds]"
    echo "Methods: zoh, foh, bilinear, poly, highorder, rk4"
    echo "Example: $0 zoh"
    echo "Example: $0 highorder 0,1,2,3,4"
    exit 1
fi

METHOD=$1
SEEDS_STR=${2:-"0,1,2,3,4"}

# Convert comma-separated seeds to array
IFS=',' read -ra SEEDS <<< "$SEEDS_STR"

# Map method name to script
case $METHOD in
    zoh)
        SCRIPT="./scripts/CVIS/cifar100/pt-cifar100-vim-zoh.sh"
        METHOD_NAME="Zero Order Hold (ZOH)"
        ;;
    foh)
        SCRIPT="./scripts/CVIS/cifar100/pt-cifar100-vim-foh.sh"
        METHOD_NAME="First Order Hold (FOH)"
        ;;
    bilinear)
        SCRIPT="./scripts/CVIS/cifar100/pt-cifar100-vim-bilinear.sh"
        METHOD_NAME="Bilinear (Tustin) Transform"
        ;;
    poly)
        SCRIPT="./scripts/CVIS/cifar100/pt-cifar100-vim-poly.sh"
        METHOD_NAME="Polynomial Interpolation"
        ;;
    highorder)
        SCRIPT="./scripts/CVIS/cifar100/pt-cifar100-vim-highorder.sh"
        METHOD_NAME="Higher-Order Hold"
        ;;
    rk4)
        SCRIPT="./scripts/CVIS/cifar100/pt-cifar100-vim-rk4.sh"
        METHOD_NAME="Runge-Kutta 4th Order (RK4)"
        ;;
    *)
        echo "Error: Unknown method '$METHOD'"
        echo "Available methods: zoh, foh, bilinear, poly, highorder, rk4"
        exit 1
        ;;
esac

echo "=========================================="
echo "Running $METHOD_NAME with Multiple Seeds"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------"
    echo "Running with seed: $SEED"
    echo "----------------------------------------"
    
    bash "$SCRIPT" $SEED
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed seed $SEED"
    else
        echo "✗ Failed seed $SEED"
        echo "Continuing with next seed..."
    fi
    echo ""
done

echo "=========================================="
echo "All seeds completed for $METHOD_NAME"
echo "=========================================="
echo ""
echo "To extract results, run:"
echo "  python ./scripts/CVIS/cifar100/extract_results.py --method $METHOD"

