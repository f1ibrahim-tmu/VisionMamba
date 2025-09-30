#!/bin/bash
# First Order Hold (FOH) discretization for Vision Mamba detection on MS-COCO

# Set COCO dataset path
export DETECTRON2_DATASETS=/home/f7ibrahi/links/scratch/dataset

# Activate environment (adjust path as needed)
# source /path/to/your/conda/bin/activate det2
# cd /path/to/VisionMamba/det

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_foh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py

python3 tools/lazyconfig_train_net.py \
 --num-gpus 4 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60902" \
 --config-file ${DET_CONFIG} \
 train.output_dir=work_dirs/${DET_CONFIG_NAME}-4gpu \
 dataloader.train.num_workers=128 \
 dataloader.test.num_workers=8 \
 model.backbone.discretization_method=foh
