#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_foh \
    --batch-size 64 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 4 \
    --data-path /home/f7ibrahi/projects/def-wangcs/dataset/ImageNet/ILSVRC2012 \
    --output_dir ./output/vim_tiny_patch16_224_bimambav2_foh \
    --no_amp