#!/bin/bash
# Polynomial Interpolation discretization for Vision Mamba detection on MS-COCO

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_poly
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port 0 \
    det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=output/detection_logs/vim_tiny_vimdet_poly \
    train.init_checkpoint="" \
    dataloader.train.num_workers=16 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=poly
    # --num-gpus 4 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60903" \
