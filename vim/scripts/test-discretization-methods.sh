#!/bin/bash
# This script runs quick testing for all discretization methods with fewer resources

# Common parameters for testing
COMMON_PARAMS="--batch-size 32 --epochs 5 --weight-decay 0.1 --num_workers 0 --data-path /data/fady/datasets/imagenet-1k --no_amp"

cd ./projects/VisionMamba/vim;

echo "Testing Zero Order Hold (ZOH)..."
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    $COMMON_PARAMS \
    --output_dir ./output/test_vim_tiny_zoh

echo "Testing First Order Hold (FOH)..."
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 ./vim/main.pyy \
    --model vim_tiny_patch16_224_bimambav2_foh \
    $COMMON_PARAMS \
    --output_dir ./output/test_vim_tiny_foh

echo "Testing Bilinear (Tustin) Transform..."
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_bilinear \
    $COMMON_PARAMS \
    --output_dir ./output/test_vim_tiny_bilinear

echo "Testing Polynomial Interpolation..."
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_poly \
    $COMMON_PARAMS \
    --output_dir ./output/test_vim_tiny_poly

echo "Testing Higher-Order Hold..."
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 ./vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_highorder \
    $COMMON_PARAMS \
    --output_dir ./output/test_vim_tiny_highorder

echo "All testing runs completed!" 