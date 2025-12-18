#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba segmentation on ADE20K

SEG_CONFIG=./seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_zoh.py
PRETRAIN_CKPT=/home/f7ibrahi/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_zoh/best_checkpoint.pth

# Conditionally set resume checkpoint if it exists
CHECKPOINT_PATH=output/segmentation_logs/vim_tiny_vimseg_upernet_zoh/checkpoint.pth
RESUME_ARG=""
if [ -f "${CHECKPOINT_PATH}" ]; then
    RESUME_ARG="--resume ${CHECKPOINT_PATH}"
    echo "Found checkpoint at ${CHECKPOINT_PATH}, will resume training from it."
else
    echo "No checkpoint found at ${CHECKPOINT_PATH}, starting training from scratch."
fi

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --nnodes=${WORLD_SIZE:-1} --node_rank=${RANK:-0} --master_addr=${MASTER_ADDR:-localhost} --master_port=10297 \

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    seg/train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=zoh \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05 \
    --work-dir output/segmentation_logs/vim_tiny_vimseg_upernet_zoh \
    ${RESUME_ARG}
