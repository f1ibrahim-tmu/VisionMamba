#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba detection on MS-COCO
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/output}"
# Init backward Mamba params from forward when loading unidirectional ckpt (default: true). Set INIT_BACKWARD_FROM_FORWARD=false to disable.
INIT_BACKWARD_FROM_FORWARD=${INIT_BACKWARD_FROM_FORWARD:-true}

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py
PRETRAIN_CKPT="${OUTPUT_ROOT}/classification_logs/vim_tiny_zoh/best_checkpoint.pth"
OUTPUT_DIR="${OUTPUT_ROOT}/detection_logs/vim_tiny_vimdet_zoh"

# Conditionally set checkpoint if it exists
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint.pth"
CHECKPOINT_ARG=""
if [ -f "${CHECKPOINT_PATH}" ]; then
    CHECKPOINT_ARG="train.init_checkpoint=${CHECKPOINT_PATH}"
    echo "Found checkpoint at ${CHECKPOINT_PATH}, will resume training from it."
else
    echo "No checkpoint found at ${CHECKPOINT_PATH}, starting training from scratch."
fi

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    --master_port=0 \
    ./det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=${OUTPUT_DIR} \
    dataloader.train.total_batch_size=32 \
    dataloader.train.num_workers=16 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=zoh \
    model.backbone.net.init_backward_from_forward=${INIT_BACKWARD_FROM_FORWARD} \
    model.backbone.net.pretrained=${PRETRAIN_CKPT} \
    optimizer.lr=1e-5 \
    optimizer.weight_decay=0.01 \
    ${CHECKPOINT_ARG}