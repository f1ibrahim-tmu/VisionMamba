#!/bin/bash
# Run benchmark after training completes
# Usage: ./run_benchmark_after_training.sh <model_name> <output_dir>
#
# Example:
#   ./performance-analysis/run_benchmark_after_training.sh \
#       vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#       ./output/classification_logs/vim_tiny_zoh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <output_dir>"
    echo "Example: $0 vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 ./output/vim_tiny_zoh"
    exit 1
fi

MODEL_NAME=$1
OUTPUT_DIR=$2
CHECKPOINT="$OUTPUT_DIR/best_checkpoint.pth"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Starting Latency & FLOPs Benchmark"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "⚠ Checkpoint not found at $CHECKPOINT"
    echo "Looking for latest checkpoint..."
    
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    # If find with -printf doesn't work (macOS), try alternative
    if [ -z "$LATEST_CHECKPOINT" ]; then
        LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint*.pth" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "✗ No checkpoint found in $OUTPUT_DIR"
        exit 1
    fi
    
    CHECKPOINT=$LATEST_CHECKPOINT
    echo "✓ Using checkpoint: $CHECKPOINT"
fi

# Run benchmark with multiple batch sizes
echo ""
echo "Running benchmarks with different batch sizes..."
echo ""

for BATCH_SIZE in 1 4 8 16 32; do
    echo "----------------------------------------"
    echo "Benchmark: Batch Size = $BATCH_SIZE"
    echo "----------------------------------------"
    
    cd "$PROJECT_ROOT"
    python performance-analysis/benchmark_latency_flops.py \
        --model "$MODEL_NAME" \
        --checkpoint "$CHECKPOINT" \
        --batch-size $BATCH_SIZE \
        --input-size 224 \
        --num-samples 1000 \
        --warmup-iters 10 \
        --num-repeats 5 \
        --output-dir "$OUTPUT_DIR/benchmark_results" \
        --device cuda
    
    echo ""
done

echo "=========================================="
echo "Benchmark Complete!"
echo "Results saved to: $OUTPUT_DIR/benchmark_results/"
echo "=========================================="

