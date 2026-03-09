#!/bin/bash
# First Order Hold (FOH) discretization for Vision Mamba detection on MS-COCO
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/output}"
# Init backward Mamba params from forward when loading unidirectional ckpt (default: true). Set INIT_BACKWARD_FROM_FORWARD=false to disable.
INIT_BACKWARD_FROM_FORWARD=${INIT_BACKWARD_FROM_FORWARD:-true}

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_foh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py
PRETRAIN_CKPT="${OUTPUT_ROOT}/classification_logs/vim_tiny_foh/best_checkpoint.pth"
OUTPUT_DIR="${OUTPUT_ROOT}/detection_logs/vim_tiny_vimdet_foh"

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    --master_port=0 \
    ./det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=${OUTPUT_DIR} \
    dataloader.train.total_batch_size=32 \
    dataloader.train.num_workers=128 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=foh \
    model.backbone.net.init_backward_from_forward=${INIT_BACKWARD_FROM_FORWARD} \
    model.backbone.net.pretrained=${PRETRAIN_CKPT} \
    optimizer.lr=1e-5 \
    optimizer.weight_decay=0.01