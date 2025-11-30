#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

# Accept seed as first argument, default to 0
SEED=${1:-0}

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_bilinear \
    --batch-size 512 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 0 \
    --input-size 32 \
    --data-set CIFAR \
    --data-path /home/f7ibrahi/links/scratch/dataset/cifar100 \
    --seed $SEED \
    --output_dir ./output/classification_logs/vim_tiny_bilinear_seed${SEED} \
    --resume ./output/classification_logs/vim_tiny_bilinear_seed${SEED}/checkpoint.pth