#!/bin/bash
# Run a single discretization method with multiple random seeds
# Usage: ./Rorqual_run-single-method-seeds.sh <method> [seeds]
# Example: ./Rorqual_run-single-method-seeds.sh zoh
# Example: ./Rorqual_run-single-method-seeds.sh highorder 0,1,2,3,4

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
        SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-zoh.sh"
        METHOD_NAME="Zero Order Hold (ZOH)"
        ;;
    foh)
        SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-foh.sh"
        METHOD_NAME="First Order Hold (FOH)"
        ;;
    bilinear)
        SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-bilinear.sh"
        METHOD_NAME="Bilinear (Tustin) Transform"
        ;;
    poly)
        SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-poly.sh"
        METHOD_NAME="Polynomial Interpolation"
        ;;
    highorder)
        SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-highorder.sh"
        METHOD_NAME="Higher-Order Hold"
        ;;
    rk4)
        SCRIPT="$SCRIPT_DIR/Rorqual_pt-vim-rk4.sh"
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
echo "  python $SCRIPT_DIR/extract_results.py --method $METHOD"

