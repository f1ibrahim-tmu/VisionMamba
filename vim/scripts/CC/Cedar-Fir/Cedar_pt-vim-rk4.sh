#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 0 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_rk4 \
    --batch-size 256 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 4 \
    --data-path /home/f7ibrahi/projects/def-wangcs/dataset/ImageNet/ILSVRC2012 \
    --output_dir ./output/classification_logs/vim_tiny_rk4 \
    --resume ./output/classification_logs/vim_tiny_rk4/checkpoint.pth