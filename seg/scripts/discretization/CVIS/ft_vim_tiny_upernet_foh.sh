#!/bin/bash
# First Order Hold (FOH) discretization for Vision Mamba segmentation on ADE20K

SEG_CONFIG=seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_foh.py
PRETRAIN_CKPT=/data/fady/projects/VisionMamba/output/vim_tiny_foh/best_checkpoint.pth

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    --master_port=0 \
    ./seg/train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-foh--deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             train_dataloader.batch_size=32 \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=foh \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05 \
    --work-dir output/segmentation_logs/vim_tiny_vimseg_upernet_foh

