#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba segmentation on ADE20K

SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_zoh.py
PRETRAIN_CKPT=/data/fady/projects/VisionMamba/output/vim_tiny_zoh/best_checkpoint.pth

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./seg/train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-zoh --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=zoh \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05 \
    --output_dir output/segmentation_logs/vim_tiny_vimseg_upernet_zoh \
    --resume output/segmentation_logs/vim_tiny_vimseg_upernet_zoh/checkpoint.pth
