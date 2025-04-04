#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model vim_base_patch16_224_bimambav2_bilinear \
    --batch-size 128 \
    --drop-path 0.1 \
    --weight-decay 0.1 \
    --num_workers 25 \
    --data-path "/data/fady/datasets/imagenet-1k/train" \
    --output_dir ./output/vim_base_bilinear \
    --no_amp 