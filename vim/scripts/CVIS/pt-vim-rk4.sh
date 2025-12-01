#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

# WARNING: RK4 discretization has known SIGBUS issues with distributed training on some systems
# SIGBUS errors during distributed barrier suggest system-level memory alignment issues
# If distributed training fails, try single-GPU training (uncomment the single-GPU section below)

# Disable shared memory for PyTorch to avoid alignment issues
export TORCH_SHARED_MEMORY_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:16
export CUDA_LAUNCH_BLOCKING=0

# Memory alignment and allocation settings
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_MAX_=65536

# Disable memory mapping for better alignment
export LD_PRELOAD=""

# Reduce OMP threads to avoid memory contention
# RK4 is memory-intensive, so we use fewer CPU threads
# SIGBUS during distributed barrier suggests system-level issues
# If this fails, try single-GPU training (see commented section below)

# Distributed training with 2 GPUs (default)
# If this fails with SIGBUS, use the single-GPU fallback below
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_rk4 \
    --batch-size 8 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 0 \
    --data-path /data/fady/datasets/imagenet-1k \
    --output_dir ./output/vim_tiny_rk4 \
    --resume ./output/vim_tiny_rk4/checkpoint.pth

# SINGLE-GPU FALLBACK: If distributed training fails with SIGBUS, uncomment this and comment out the distributed section above
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python ./vim/main.py \
#     --model vim_tiny_patch16_224_bimambav2_rk4 \
#     --batch-size 16 \
#     --drop-path 0.0 \
#     --weight-decay 0.05 \
#     --lr 0.001 \
#     --num_workers 0 \
#     --data-path /data/fady/datasets/imagenet-1k \
#     --output_dir ./output/vim_tiny_rk4 \
#     --resume ./output/vim_tiny_rk4/checkpoint.pth