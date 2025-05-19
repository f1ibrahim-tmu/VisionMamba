#!/bin/bash

# conda activate conda_visionmamba
# cd ./projects/VisionMamba/vim;

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --master_port=0 \
    ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_foh \
    --batch-size 64 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 0 \
    --data-path /data/fady/datasets/imagenet-1k \
    --output_dir ./output/vim_tiny_foh 

# GPU stats every 8 hours to file
watch -n 28800 -t 'nvidia-smi >> ./output/vim_tiny_foh/vim_tiny_foh_gpu.log'

# PyTorch memory every 8 hours
# watch -n 28800 -t 'python -c "import torch; print(torch.cuda.memory_summary())" >> ./output/vim_tiny_foh_torch_mem.log'
