#!/bin/bash
# Higher-Order Hold discretization for Vision Mamba segmentation on ADE20K
#
# Usage: ./ft_vim_tiny_upernet_highorder.sh [DATASET_PATH]
#   DATASET_PATH: Path to ade20k directory (e.g., /path/to/ade20k)
#                 If not provided, defaults to /home/f7ibrahi/scratch/dataset/ade20k/ADEChallengeData2016

# 1. Dataset Path Logic
# Get dataset path from command line argument or use default
# Expected structure: ${DATASET_PATH}/ADEChallengeData2016/
if [ -z "$1" ]; then
    ADE20K_DATASET_PATH="/home/f7ibrahi/scratch/dataset/ade20k/ADEChallengeData2016"
    echo "No dataset path provided, using default: ${ADE20K_DATASET_PATH}"
else
    ADE20K_DATASET_PATH="${1}/ADEChallengeData2016"
    echo "Using dataset path: ${ADE20K_DATASET_PATH}"
fi

# 2. Training Variables
# Required for deterministic mode with CuBLAS
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SEG_CONFIG=seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_highorder.py
PRETRAIN_CKPT=/home/f7ibrahi/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_highorder/best_checkpoint.pth

# 3. Resume Logic
# MMEngine saves checkpoints as latest.pth, iter_*.pth, or custom names
WORK_DIR=./output/segmentation_logs/vim_tiny_fir_vimseg_upernet_highorder
RESUME_ARG=""
CHECKPOINT_PATH=""

# Check for latest.pth first (MMEngine default)
if [ -f "${WORK_DIR}/latest.pth" ]; then
    CHECKPOINT_PATH="${WORK_DIR}/latest.pth"
elif [ -f "${WORK_DIR}/checkpoint.pth" ]; then
    CHECKPOINT_PATH="${WORK_DIR}/checkpoint.pth"
else
    # Find the most recent .pth file in work_dir
    if [ -d "${WORK_DIR}" ]; then
        LATEST_CKPT=$(find "${WORK_DIR}" -maxdepth 1 -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "${LATEST_CKPT}" ] && [ -f "${LATEST_CKPT}" ]; then
            CHECKPOINT_PATH="${LATEST_CKPT}"
        fi
    fi
fi

if [ -n "${CHECKPOINT_PATH}" ] && [ -f "${CHECKPOINT_PATH}" ]; then
    RESUME_ARG="--resume-from ${CHECKPOINT_PATH}"
    echo "Found checkpoint at ${CHECKPOINT_PATH}, will resume training from it."
else
    echo "No checkpoint found in ${WORK_DIR}, starting training from scratch."
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

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=$MASTER_PORT \
    seg/train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 \
    --work-dir ${WORK_DIR} \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             train_dataloader.batch_size=48 \
             model.backbone.if_bimamba=True \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=highorder \
             optimizer.lr=1e-5 \
             optimizer.weight_decay=0.01 \
             train_dataloader.dataset.data_root="${ADE20K_DATASET_PATH}" \
             val_dataloader.dataset.data_root="${ADE20K_DATASET_PATH}" \
             test_dataloader.dataset.data_root="${ADE20K_DATASET_PATH}" \
             train_cfg.max_iters=200000 \
    ${RESUME_ARG}
