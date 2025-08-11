#!/bin/bash
# This script runs training for all discretization methods sequentially

echo "Running Zero Order Hold (ZOH) training..."
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-zoh.sh

echo "Running First Order Hold (FOH) training..."
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-foh.sh

echo "Running Bilinear (Tustin) Transform training..."
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-bilinear.sh

echo "Running Polynomial Interpolation training..."
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-poly.sh

echo "Running Higher-Order Hold training..."
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-highorder.sh

echo "Running RK4 training..."
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-rk4.sh

echo "All training runs completed!" 