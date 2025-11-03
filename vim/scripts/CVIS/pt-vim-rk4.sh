#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

# Fix for Bus error (SIGBUS) in distributed training with RK4
# RK4 discretization uses memory-intensive operations that can cause alignment issues
# Set shared memory size and other mitigations
export TORCH_SHARED_MEMORY_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Increase shared memory size for Docker/containers (if applicable)
# Uncomment if running in Docker:
# export TORCH_SHARED_MEMORY_ALLOCATOR=file

# Use a slightly smaller batch size for RK4 to reduce memory pressure
# RK4 is more memory-intensive due to its 4-stage computations
BATCH_SIZE=96  # Reduced from 128 to avoid memory issues

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_rk4 \
    --batch-size ${BATCH_SIZE} \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 0 \
    --data-path /data/fady/datasets/imagenet-1k \
    --output_dir ./output/vim_tiny_rk4 \
    --resume ./output/vim_tiny_rk4/checkpoint.pth