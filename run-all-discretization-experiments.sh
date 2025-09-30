#!/bin/bash
# Master script to run all discretization experiments across all tasks
# This script runs discretization experiments for:
# 1. Image Classification (existing)
# 2. Semantic Segmentation on ADE20K
# 3. Object Detection on MS-COCO

echo "=========================================="
echo "Vision Mamba Discretization Experiments"
echo "Running all tasks with all discretization methods"
echo "=========================================="

# Set base directory
BASE_DIR=$(pwd)

# 1. Run Image Classification experiments (if not already done)
echo "=========================================="
echo "1. Image Classification Experiments"
echo "=========================================="
if [ -d "vim/scripts/CVIS/cifar100" ]; then
    echo "Running image classification discretization experiments..."
    cd vim
    bash ./scripts/CVIS/cifar100/run-all-discretization-methods.sh
    cd $BASE_DIR
else
    echo "Image classification scripts not found, skipping..."
fi

# 2. Run Semantic Segmentation experiments
echo "=========================================="
echo "2. Semantic Segmentation Experiments (ADE20K)"
echo "=========================================="
if [ -d "seg" ]; then
    echo "Running semantic segmentation discretization experiments..."
    cd seg
    bash ./scripts/discretization/run-all-segmentation-discretization-methods.sh
    cd $BASE_DIR
else
    echo "Segmentation directory not found, skipping..."
fi

# 3. Run Object Detection experiments
echo "=========================================="
echo "3. Object Detection Experiments (MS-COCO)"
echo "=========================================="
if [ -d "det" ]; then
    echo "Running object detection discretization experiments..."
    cd det
    bash ./scripts/discretization/run-all-detection-discretization-methods.sh
    cd $BASE_DIR
else
    echo "Detection directory not found, skipping..."
fi

echo "=========================================="
echo "All discretization experiments completed!"
echo "=========================================="
echo "Results summary:"
echo "- Image Classification: vim/output/"
echo "- Semantic Segmentation: seg/work_dirs/"
echo "- Object Detection: det/work_dirs/"
echo "=========================================="
