#!/bin/bash
# Polynomial Interpolation discretization for Vision Mamba detection on MS-COCO

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

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_poly
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port $MASTER_PORT \
    det/tools/lazyconfig_train_net.py \
    --config-file ${DET_CONFIG} \
    train.output_dir=output/detection_logs/vim_tiny_vimdet_poly \
    train.init_checkpoint="" \
    dataloader.train.num_workers=16 \
    dataloader.test.num_workers=8 \
    model.backbone.net.discretization_method=poly
    # --num-gpus 4 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60903" \
