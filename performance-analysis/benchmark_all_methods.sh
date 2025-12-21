#!/bin/bash
# Benchmark all discretization methods
# Usage: ./performance-analysis/benchmark_all_methods.sh

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

METHODS=("zoh" "foh" "bilinear" "poly" "highorder" "rk4")
OUTPUT_BASE="./output"

echo "=========================================="
echo "Benchmarking All Discretization Methods"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

for METHOD in "${METHODS[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/classification_logs/vim_tiny_$METHOD"
    MODEL_NAME="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    
    # Try alternative model names if the standard one doesn't work
    # Some methods might have specific model variants
    if [ "$METHOD" != "zoh" ]; then
        # Try method-specific model name
        ALT_MODEL_NAME="vim_tiny_patch16_224_bimambav2_${METHOD}"
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Benchmarking: $METHOD (using $ALT_MODEL_NAME)"
            bash "$SCRIPT_DIR/run_benchmark_after_training.sh" "$ALT_MODEL_NAME" "$OUTPUT_DIR" || \
            bash "$SCRIPT_DIR/run_benchmark_after_training.sh" "$MODEL_NAME" "$OUTPUT_DIR"
            echo ""
        else
            echo "⚠ Output directory not found: $OUTPUT_DIR"
        fi
    else
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Benchmarking: $METHOD"
            bash "$SCRIPT_DIR/run_benchmark_after_training.sh" "$MODEL_NAME" "$OUTPUT_DIR"
            echo ""
        else
            echo "⚠ Output directory not found: $OUTPUT_DIR"
        fi
    fi
done

echo "=========================================="
echo "All benchmarks complete!"
echo "Check benchmark_results directories for detailed results"
echo "=========================================="

