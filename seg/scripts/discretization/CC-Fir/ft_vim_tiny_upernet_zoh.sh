#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba segmentation on ADE20K

SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_zoh.py
PRETRAIN_CKPT=home/f7ibrahi/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/vim_tiny_zoh/best_checkpoint.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --nproc_per_node=4 --nnodes=${WORLD_SIZE:-1} --node_rank=${RANK:-0} --master_addr=${MASTER_ADDR:-localhost} --master_port=10297 \
    --use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-zoh --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=zoh \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05
