#!/bin/bash
# This script runs training for all discretization methods sequentially on Compute Canada

echo "Running Zero Order Hold (ZOH) training..."
bash ./scripts/CC/CC_pt-vim-zoh.sh

echo "Running First Order Hold (FOH) training..."
bash ./scripts/CC/CC_pt-vim-foh.sh

echo "Running Bilinear (Tustin) Transform training..."
bash ./scripts/CC/CC_pt-vim-bilinear.sh

echo "Running Polynomial Interpolation training..."
bash ./scripts/CC/CC_pt-vim-poly.sh

echo "Running Higher-Order Hold training..."
bash ./scripts/CC/CC_pt-vim-highorder.sh

echo "Running Runge-Kutta 4th Order (RK4) training..."
bash ./scripts/CC/CC_pt-vim-rk4.sh

echo "All training runs completed!" 