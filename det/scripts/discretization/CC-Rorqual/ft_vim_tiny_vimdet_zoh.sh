#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba detection on MS-COCO

# Change to det directory
# cd /lustre09/project/6062393/f7ibrahi/projects/VisionMamba/det

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh
# Script now auto-detects whether running from root or det/ directory
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

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=output/detection_logs/vim_tiny_vimdet_foh \
    dataloader.train.num_workers=16 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=zoh \
    ${CHECKPOINT_ARG}

    # --num-gpus 4 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60903" \
