#!/bin/bash

# conda activate conda_visionmamba
# Get the project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Accept seed as first argument, default to 0
SEED=${1:-0}

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT"

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./main.py \
    --model vim_tiny_patch16_224_bimambav2_foh \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 0 \
    --input-size 32 \
    --data-set CIFAR \
    --data-path /data/fady/datasets/cifar100 \
    --seed $SEED \
    --output_dir ./output/classification_logs/cifar100/vim_tiny_foh_seed${SEED} \
    --resume ./output/classification_logs/cifar100/vim_tiny_foh_seed${SEED}/checkpoint.pth 