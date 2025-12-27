#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

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

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port $MASTER_PORT \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_highorder \
    --batch-size 256 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 2 \
    --data-path /home/f7ibrahi/projects/def-wangcs/dataset/ImageNet/ILSVRC2012 \
    --output_dir ./output/classification_logs/vim_tiny_highorder \
    --resume ./output/classification_logs/vim_tiny_highorder/checkpoint.pth