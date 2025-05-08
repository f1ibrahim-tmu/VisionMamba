#!/bin/bash
# This script runs training for all discretization methods sequentially

echo "Running Zero Order Hold (ZOH) training..."
bash ./scripts/CVIS/pt-vim-zoh.sh

echo "Running First Order Hold (FOH) training..."
bash ./scripts/CVIS/pt-vim-foh.sh

echo "Running Bilinear (Tustin) Transform training..."
bash ./scripts/CVIS/pt-vim-bilinear.sh

echo "Running Polynomial Interpolation training..."
bash ./scripts/CVIS/pt-vim-poly.sh

echo "Running Higher-Order Hold training..."
bash ./scripts/CVIS/pt-vim-highorder.sh

echo "All training runs completed!" 