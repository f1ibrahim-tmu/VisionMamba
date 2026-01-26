#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba segmentation on ADE20K

SEG_CONFIG=seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_zoh.py
PRETRAIN_CKPT=/data/fady/projects/VisionMamba/output/vim_tiny_zoh/best_checkpoint.pth

# Check if we should resume training
# MMEngine saves checkpoints as latest.pth, iter_*.pth, or custom names
WORK_DIR=output/segmentation_logs/vim_tiny_vimseg_upernet_zoh
RESUME_ARG=""
CHECKPOINT_PATH=""

# Check for latest.pth first (MMEngine default)
if [ -f "${WORK_DIR}/latest.pth" ]; then
    CHECKPOINT_PATH="${WORK_DIR}/latest.pth"
elif [ -f "${WORK_DIR}/checkpoint.pth" ]; then
    CHECKPOINT_PATH="${WORK_DIR}/checkpoint.pth"
else
    # Find the most recent .pth file in work_dir (GNU find -printf works on Linux HPC clusters)
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

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
    --master_port=0 \
    ./seg/train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-zoh --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             train_dataloader.batch_size=32 \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=zoh \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05 \
    --work-dir output/segmentation_logs/vim_tiny_vimseg_upernet_zoh \
    ${RESUME_ARG}
