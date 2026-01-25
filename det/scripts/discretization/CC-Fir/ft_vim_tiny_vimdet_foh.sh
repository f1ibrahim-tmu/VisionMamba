#!/bin/bash
# First Order Hold (FOH) discretization for Vision Mamba detection on MS-COCO
#
# Usage: ./ft_vim_tiny_vimdet_foh.sh [DATASET_PATH]
#   DATASET_PATH: Path to coco directory (e.g., /path/to/coco)
#                 If not provided, defaults to ./datasets
#   Note: Detectron2 expects COCO at ${DETECTRON2_DATASETS}/coco/

# 1. Dataset Path Logic
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

# 2. Training Variables
DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_foh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py
PRETRAIN_CKPT=/home/f7ibrahi/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_foh/best_checkpoint.pth
OUTPUT_DIR=output/detection_logs/vim_tiny_fir_vimdet_foh
# Calculate workers per GPU based on the total cores allocated by SLURM
# Fir: 48 cores / 4 GPUs = 12 workers per task
# Rorqual: 64 cores / 4 GPUs = 16 workers per task
WORKERS_PER_GPU=$((SLURM_CPUS_PER_TASK / 4))

# 3. Resume Logic
# The checkpointer looks for a 'last_checkpoint' file in the output directory
RESUME_FLAG=""
LAST_CHECKPOINT_FILE="${OUTPUT_DIR}/last_checkpoint"
if [ -f "${LAST_CHECKPOINT_FILE}" ]; then
    RESUME_FLAG="--resume"
    echo "Found last_checkpoint file at ${LAST_CHECKPOINT_FILE}, will resume training."
    echo "Checkpoint path: $(cat ${LAST_CHECKPOINT_FILE})"
else
    echo "No checkpoint found, starting training from scratch."
    echo "Note: train.init_checkpoint is set to empty string, so pretrained backbone weights will be loaded from ${PRETRAIN_CKPT}"
fi

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

# 4. Training Command
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=$MASTER_PORT \
#     det/tools/lazyconfig_train_net.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 python det/tools/lazyconfig_train_net.py \
    --num-gpus 4 \
    --config-file ${DET_CONFIG} \
    ${RESUME_FLAG} \
    train.output_dir=${OUTPUT_DIR} \
    train.init_checkpoint="" \
    dataloader.train.total_batch_size=64 \
    dataloader.train.num_workers=${WORKERS_PER_GPU} \
    dataloader.test.num_workers=$((WORKERS_PER_GPU / 2)) \
    model.backbone.net.discretization_method=foh \
    model.backbone.net.pretrained=${PRETRAIN_CKPT}
    # --use-wandb \
    # --wandb-project visionmamba \
    # --wandb-run-name vim_tiny_vimdet_foh_cc-fir \
    # --wandb-tags detection foh cc-fir \