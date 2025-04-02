#!/bin/bash
conda activate conda_visionmamba
cd ./projects/VisionMamba/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2 \
    --batch-size 128 \
    --drop-path 0.1 \
    --weight-decay 0.1 \
    --num_workers 25 \
    --data-path "/Volumes/X10 Pro/datasets/imagenet-1k/train" \
    --output_dir ./output/vim_base_zoh \
    --no_amp 