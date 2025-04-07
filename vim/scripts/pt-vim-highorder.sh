#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_highorder \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 0 \
    --data-path /data/fady/datasets/imagenet-1k \
    --output_dir ./output/vim_base_highorder \
    --no_amp 