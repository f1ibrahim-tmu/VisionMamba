#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba detection on MS-COCO

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py

# Conditionally set checkpoint if it exists
CHECKPOINT_PATH=./output/detection_logs/vim_tiny_vimdet_zoh/checkpoint.pth
CHECKPOINT_ARG=""
if [ -f "${CHECKPOINT_PATH}" ]; then
    CHECKPOINT_ARG="train.init_checkpoint=${CHECKPOINT_PATH}"
    echo "Found checkpoint at ${CHECKPOINT_PATH}, will resume training from it."
else
    echo "No checkpoint found at ${CHECKPOINT_PATH}, starting training from scratch."
fi

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=output/detection_logs/vim_tiny_vimdet_foh \
    dataloader.train.num_workers=128 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=zoh \
    ${CHECKPOINT_ARG}