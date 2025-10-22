#!/bin/bash
# This script runs training for all discretization methods on MS-COCO object detection

echo "=========================================="
echo "Running Vision Mamba Discretization Experiments"
echo "Task: Object Detection on MS-COCO"
echo "=========================================="

echo "Running Zero Order Hold (ZOH) training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_zoh.sh

echo "Running First Order Hold (FOH) training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_foh.sh

echo "Running Bilinear (Tustin) Transform training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_bilinear.sh

echo "Running Polynomial Interpolation training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_poly.sh

echo "Running Higher-Order Hold training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_highorder.sh

echo "Running RK4 training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_rk4.sh

echo "=========================================="
echo "All detection discretization experiments completed!"
echo "Results saved in work_dirs/cascade_mask_rcnn_vimdet_t_100ep_adj1_{method}-4gpu/"
echo "=========================================="
