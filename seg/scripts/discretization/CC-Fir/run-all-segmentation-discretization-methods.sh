#!/bin/bash
# This script runs training for all discretization methods on ADE20K semantic segmentation

echo "=========================================="
echo "Running Vision Mamba Discretization Experiments"
echo "Task: Semantic Segmentation on ADE20K"
echo "=========================================="

echo "Running Zero Order Hold (ZOH) training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_upernet_zoh.sh

echo "Running First Order Hold (FOH) training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_upernet_foh.sh

echo "Running Bilinear (Tustin) Transform training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_upernet_bilinear.sh

echo "Running Polynomial Interpolation training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_upernet_poly.sh

echo "Running Higher-Order Hold training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_upernet_highorder.sh

echo "Running RK4 training..."
bash ./scripts/discretization/CC-Fir/ft_vim_tiny_upernet_rk4.sh

echo "=========================================="
echo "All segmentation discretization experiments completed!"
echo "Results saved in work_dirs/vimseg-t-{method}/"
echo "=========================================="
