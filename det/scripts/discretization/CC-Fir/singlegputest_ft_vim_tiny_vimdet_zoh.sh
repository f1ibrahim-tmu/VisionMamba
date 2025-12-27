#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba detection on MS-COCO

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=output/detection_logs/vim_tiny_vimdet_zoh \
    train.init_checkpoint="" \
    dataloader.train.num_workers=4 \
    dataloader.test.num_workers=2 \
    model.backbone.net.discretization_method=zoh
    # --num-gpus 4 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60903" \
