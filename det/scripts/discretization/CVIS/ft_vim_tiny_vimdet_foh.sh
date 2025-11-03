#!/bin/bash
# First Order Hold (FOH) discretization for Vision Mamba detection on MS-COCO

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_foh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    --output_dir ./output/detection_logs/vim_tiny_vimdet_foh \
    dataloader.train.num_workers=128 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=foh \
    train.init_checkpoint=./output/detection_logs/vim_tiny_vimdet_foh/checkpoint.pth