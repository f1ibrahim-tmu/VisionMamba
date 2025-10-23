#!/bin/bash
# Runge-Kutta 4th Order (RK4) discretization for Vision Mamba segmentation on ADE20K

SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_rk4.py
PRETRAIN_CKPT=home/f7ibrahi/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/vim_tiny_rk4/best_checkpoint.pth
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --nnodes=${WORLD_SIZE:-1} --node_rank=${RANK:-0} --master_addr=${MASTER_ADDR:-localhost} --master_port=10297 \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    ./seg/train.py --launcher slurm \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-rk4 --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=rk4 \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05