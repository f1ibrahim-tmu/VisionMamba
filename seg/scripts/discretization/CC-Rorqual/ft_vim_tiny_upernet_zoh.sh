#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba segmentation on ADE20K
#
# Usage: ./ft_vim_tiny_upernet_zoh.sh [DATASET_PATH]
#   DATASET_PATH: Path to ade20k directory (e.g., /path/to/ade20k)
#                 If not provided, defaults to /home/f7ibrahi/scratch/dataset/ade20k/ADEChallengeData2016

# Get dataset path from command line argument or use default
# Expected structure: ${DATASET_PATH}/ADEChallengeData2016/
if [ -z "$1" ]; then
    ADE20K_DATASET_PATH="/home/f7ibrahi/scratch/dataset/ade20k/ADEChallengeData2016"
    echo "No dataset path provided, using default: ${ADE20K_DATASET_PATH}"
else
    ADE20K_DATASET_PATH="${1}/ADEChallengeData2016"
    echo "Using dataset path: ${ADE20K_DATASET_PATH}"
fi

SEG_CONFIG=seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_zoh.py
PRETRAIN_CKPT=/home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/vim_tiny_zoh/best_checkpoint.pth

# Conditionally set resume checkpoint if it exists
CHECKPOINT_PATH=./output/segmentation_logs/vim_tiny_vimseg_upernet_zoh/checkpoint.pth
RESUME_ARG=""
if [ -f "${CHECKPOINT_PATH}" ]; then
    RESUME_ARG="--resume-from ${CHECKPOINT_PATH}"
    echo "Found checkpoint at ${CHECKPOINT_PATH}, will resume training from it."
else
    echo "No checkpoint found at ${CHECKPOINT_PATH}, starting training from scratch."
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

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=${WORLD_SIZE:-1} --node_rank=${RANK:-0} --master_addr=${MASTER_ADDR:-localhost} --master_port=10297 \

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port $MASTER_PORT \
    seg/train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir ./output/segmentation_logs/vim_tiny_vimseg_upernet_zoh --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=zoh \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05 \
             train_dataloader.dataset.data_root="${ADE20K_DATASET_PATH}" \
             val_dataloader.dataset.data_root="${ADE20K_DATASET_PATH}" \
             test_dataloader.dataset.data_root="${ADE20K_DATASET_PATH}" \
    ${RESUME_ARG}