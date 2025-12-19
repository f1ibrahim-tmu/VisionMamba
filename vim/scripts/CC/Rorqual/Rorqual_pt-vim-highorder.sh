#!/bin/bash

# conda activate conda_visionmamba
# Get the project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Accept seed as first argument, default to 0
SEED=${1:-0}

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --rdzv-endpoint=localhost:0 \
    ./main.py \
    --model vim_tiny_patch16_224_bimambav2_highorder \
    --batch-size 512 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 2 \
    --input-size 32 \
    --data-set CIFAR \
    --data-path /home/f7ibrahi/links/scratch/dataset/cifar100 \
    --seed $SEED \
    --output_dir /home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_highorder_seed${SEED} \
    --resume /home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_highorder_seed${SEED}/checkpoint.pth