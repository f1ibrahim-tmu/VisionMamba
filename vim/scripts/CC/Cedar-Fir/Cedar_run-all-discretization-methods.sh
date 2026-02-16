#!/bin/bash
# This script runs training for all discretization methods sequentially on Compute Canada
# Usage: $0 [local_data]
# Example (Fir): $0 /path/to/Fir/imagenet

LOCAL_DATA="$1"

echo "Running Zero Order Hold (ZOH) training..."
bash ./scripts/CC/Cedar-Fir/Cedar_pt-vim-zoh.sh ${LOCAL_DATA:+"$LOCAL_DATA"}

echo "Running First Order Hold (FOH) training..."
bash ./scripts/CC/Cedar-Fir/Cedar_pt-vim-foh.sh ${LOCAL_DATA:+"$LOCAL_DATA"}

echo "Running Bilinear (Tustin) Transform training..."
bash ./scripts/CC/Cedar-Fir/Cedar_pt-vim-bilinear.sh ${LOCAL_DATA:+"$LOCAL_DATA"}

echo "Running Polynomial Interpolation training..."
bash ./scripts/CC/Cedar-Fir/Cedar_pt-vim-poly.sh ${LOCAL_DATA:+"$LOCAL_DATA"}

echo "Running Higher-Order Hold training..."
bash ./scripts/CC/Cedar-Fir/Cedar_pt-vim-highorder.sh ${LOCAL_DATA:+"$LOCAL_DATA"}

echo "Running Runge-Kutta 4th Order (RK4) training..."
bash ./scripts/CC/Cedar-Fir/Cedar_pt-vim-rk4.sh ${LOCAL_DATA:+"$LOCAL_DATA"}

echo "All training runs completed!" 