#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba segmentation on ADE20K

SEG_CONFIG=seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_zoh.py
PRETRAIN_CKPT=/data/fady/projects/VisionMamba/output/vim_tiny_zoh/best_checkpoint.pth

# Conditionally set resume checkpoint if it exists
CHECKPOINT_PATH=output/segmentation_logs/vim_tiny_vimseg_upernet_zoh/checkpoint.pth
RESUME_ARG=""
if [ -f "${CHECKPOINT_PATH}" ]; then
    RESUME_ARG="--resume ${CHECKPOINT_PATH}"
    echo "Found checkpoint at ${CHECKPOINT_PATH}, will resume training from it."
else
    echo "No checkpoint found at ${CHECKPOINT_PATH}, starting training from scratch."
fi

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    --master_port=0 \
    ./seg/train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-zoh --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             train_dataloader.batch_size=32 \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=zoh \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05 \
    --work-dir output/segmentation_logs/vim_tiny_vimseg_upernet_zoh \
    ${RESUME_ARG}
