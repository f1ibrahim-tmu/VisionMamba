#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_highorder \
    --batch-size 512 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 0 \
    --input-size 32 \
    --data-set CIFAR \
    --data-path /home/f7ibrahi/links/scratch/dataset/cifar100/cifar-100-python \
    --output_dir ./output/vim_tiny_highorder \
    --resume ./output/vim_tiny_highorder/checkpoint.pth