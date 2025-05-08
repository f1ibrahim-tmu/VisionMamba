#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

torchrun \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 0 \
    --data-path /data/fady/datasets/imagenet-1k \
    --output_dir ./output/vim_base_zoh
