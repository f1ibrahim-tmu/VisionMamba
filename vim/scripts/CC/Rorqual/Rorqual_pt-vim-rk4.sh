#!/bin/bash

# conda activate conda_visionmamba
# Get the project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Usage: $0 <local_data> [seed]
# Example (Rorqual): $0 /path/to/Rorqual/cifar100 0
LOCAL_DATA="$1"
if [ -z "$LOCAL_DATA" ]; then
    echo "No dataset path found."
    echo "Usage: $0 <local_data> [seed]"
    exit 1
fi
SEED=${2:-0}

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT"

# Generate unique port based on SLURM job ID (if available) or use process ID
# Port range: 29500-29999 (500 ports available)
if [ -n "$SLURM_JOB_ID" ]; then
    MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 500))
else
    # Fallback: use process ID if not in SLURM environment
    MASTER_PORT=$((29500 + $$ % 500))
fi
export MASTER_PORT
echo "Using MASTER_PORT=$MASTER_PORT for job ${SLURM_JOB_ID:-$$}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=$MASTER_PORT \
    ./main.py \
    --model vim_tiny_patch16_224_bimambav2_rk4 \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 4 \
    --input-size 32 \
    --data-set CIFAR \
    --data-path "$LOCAL_DATA" \
    --seed $SEED \
    --output_dir /home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_rk4_seed${SEED} \
    --resume /home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_rk4_seed${SEED}/checkpoint.pth