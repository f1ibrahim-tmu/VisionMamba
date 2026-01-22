#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
    --master_port=0 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_bilinear \
    --batch-size 256 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.002 \
    --num_workers 0 \
    --data-path /data/fady/datasets/imagenet-1k \
    --output_dir ./output/vim_tiny_bilinear \
    --resume ./output/vim_tiny_bilinear/checkpoint.pth