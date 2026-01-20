#!/bin/bash
# Higher-Order Hold discretization for Vision Mamba detection on MS-COCO
#
# Usage: ./ft_vim_tiny_vimdet_highorder.sh [DATASET_PATH]
#   DATASET_PATH: Path to coco directory (e.g., /path/to/coco)
#                 If not provided, defaults to ./datasets
#   Note: Detectron2 expects COCO at ${DETECTRON2_DATASETS}/coco/

# Get dataset path from command line argument or use default
# Expected structure: ${DETECTRON2_DATASETS}/coco/ (so we use parent of coco directory)
if [ -z "$1" ]; then
    DETECTRON2_DATASETS="./datasets"
    echo "No dataset path provided, using default: ${DETECTRON2_DATASETS}"
else
    DETECTRON2_DATASETS="$(dirname "${1}")"
    echo "Using datasets root: ${DETECTRON2_DATASETS} (COCO at ${DETECTRON2_DATASETS}/coco/)"
fi

export DETECTRON2_DATASETS

# Generate unique port based on SLURM job ID (if available) or use process ID
# Port range: 29500-29999 (500 ports available)
if [ -n "$SLURM_JOB_ID" ]; then
    MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 500))
else
    # Fallback: use process ID if not in SLURM environment
    MASTER_PORT=$((29500 + $$ % 500))
fi
export MASTER_PORT

echo "Using MASTER_PORT=$MASTER_PORT for job ${SLURM_JOB_ID:-$$}"

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_highorder
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port $MASTER_PORT \
    det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=output/detection_logs/vim_tiny_vimdet_highorder \
    train.init_checkpoint="" \
    dataloader.train.num_workers=16 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=highorder
    # --num-gpus 4 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60903" \
