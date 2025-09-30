# Vision Mamba Discretization Experiments

This repository contains comprehensive experiments comparing different discretization methods for Vision Mamba across multiple computer vision tasks: **Image Classification**, **Semantic Segmentation**, and **Object Detection**.

## Overview

The original Vision Mamba implementation uses Zero Order Hold (ZOH) discretization. This project implements and compares 6 different discretization methods:

1. **Zero Order Hold (ZOH)** - Original implementation
2. **First Order Hold (FOH)** - Linear interpolation
3. **Bilinear (Tustin) Transform** - Trapezoidal approximation
4. **Polynomial Interpolation** - Higher-order polynomial approximation
5. **Higher-Order Hold** - Taylor series expansion
6. **Runge-Kutta 4th Order (RK4)** - 4th order numerical integration

## Tasks and Datasets

### 1. Image Classification

- **Dataset**: CIFAR-100, ImageNet-1K
- **Model**: Vision Mamba Tiny
- **Location**: `vim/scripts/CVIS/`

### 2. Semantic Segmentation

- **Dataset**: ADE20K
- **Model**: Vision Mamba Tiny + UperNet
- **Location**: `seg/scripts/discretization/`

### 3. Object Detection

- **Dataset**: MS-COCO
- **Model**: Vision Mamba Tiny + Cascade Mask R-CNN
- **Location**: `det/scripts/discretization/`

## Quick Start

### Run All Experiments

To run all discretization experiments across all tasks:

```bash
# Make the master script executable
chmod +x run-all-discretization-experiments.sh

# Run all experiments
./run-all-discretization-experiments.sh
```

### Run Individual Task Experiments

#### Semantic Segmentation (ADE20K)

```bash
cd seg
chmod +x scripts/discretization/run-all-segmentation-discretization-methods.sh
./scripts/discretization/run-all-segmentation-discretization-methods.sh
```

#### Object Detection (MS-COCO)

```bash
cd det
chmod +x scripts/discretization/run-all-detection-discretization-methods.sh
./scripts/discretization/run-all-detection-discretization-methods.sh
```

### Run Individual Method Experiments

#### Segmentation - Zero Order Hold

```bash
cd seg
bash scripts/discretization/ft_vim_tiny_upernet_zoh.sh
```

#### Detection - First Order Hold

```bash
cd det
bash scripts/discretization/ft_vim_tiny_vimdet_foh.sh
```

## Configuration Files

### Segmentation Configs

- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_zoh.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_foh.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_bilinear.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_poly.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_highorder.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_rk4.py`

### Detection Configs

- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_foh.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_bilinear.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_poly.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_highorder.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_rk4.py`

## Evaluation and Comparison

### Compare All Methods

```bash
# Compare all tasks
python compare-all-discretization-methods.py

# Compare segmentation only
cd seg
python scripts/discretization/compare-segmentation-discretization-methods.py

# Compare detection only
cd det
python scripts/discretization/compare-detection-discretization-methods.py
```

### Expected Outputs

The comparison scripts will generate:

- CSV files with detailed results
- PNG plots showing performance comparisons
- JSON files with raw data
- Summary statistics and best method identification

## Discretization Methods Details

### 1. Zero Order Hold (ZOH)

- **Formula**: A_d = exp(A*δ), B_d = δ*B
- **Characteristics**: Simple, original implementation
- **Trade-offs**: Fast but may introduce inaccuracies

### 2. First Order Hold (FOH)

- **Formula**: A_d = exp(A*δ), B_d = (A^-1)*(A_d - I)\*B
- **Characteristics**: Linear interpolation between samples
- **Trade-offs**: More accurate than ZOH for smooth signals

### 3. Bilinear (Tustin) Transform

- **Formula**: A_d = (I + A*δ/2)^(-1) * (I - A*δ/2), B_d = (I + A*δ/2)^(-1) * δ*B
- **Characteristics**: Preserves stability properties
- **Trade-offs**: Good balance of accuracy and stability

### 4. Polynomial Interpolation

- **Formula**: B_d = δ*B + δ²*A*B/2 + δ³*A²\*B/6
- **Characteristics**: 3rd order polynomial approximation
- **Trade-offs**: Higher accuracy for complex signals

### 5. Higher-Order Hold

- **Formula**: Taylor series expansion with higher-order terms
- **Characteristics**: Improved accuracy over ZOH and FOH
- **Trade-offs**: More computationally expensive

### 6. Runge-Kutta 4th Order (RK4)

- **Formula**:
  - k1 = δ \* f(t_n, y_n)
  - k2 = δ \* f(t_n + δ/2, y_n + k1/2)
  - k3 = δ \* f(t_n + δ/2, y_n + k2/2)
  - k4 = δ \* f(t_n + δ, y_n + k3)
  - y\_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4)/6
- **Characteristics**: Highest accuracy, 4th order method
- **Trade-offs**: Most computationally expensive

## Hardware Requirements

### Minimum Requirements

- 4x GPUs (recommended: A100, V100, or RTX 3090/4090)
- 32GB RAM
- 500GB storage for datasets and checkpoints

### Recommended Setup

- 8x GPUs for faster training
- 64GB RAM
- 1TB NVMe SSD storage

## Environment Setup

### Segmentation Environment

```bash
# Install MMSegmentation and dependencies
cd seg
pip install -r seg-requirements.txt
```

### Detection Environment

```bash
# Install Detectron2 and dependencies
cd det
pip install -r det-requirements.txt
```

### Classification Environment

```bash
# Install Vision Mamba dependencies
cd vim
pip install -r vim_requirements.txt
```

## Dataset Preparation

### ADE20K (Segmentation)

```bash
# Download and prepare ADE20K dataset
cd seg
python datasets/prepare_ade20k_sem_seg.py
```

### MS-COCO (Detection)

```bash
# Download and prepare MS-COCO dataset
cd det
# Follow Detectron2 dataset setup instructions
```

### CIFAR-100/ImageNet (Classification)

```bash
# Download datasets to appropriate directories
# Update data paths in config files
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size in config files
   - Use gradient accumulation
   - Enable mixed precision training

2. **Dataset Not Found**

   - Check dataset paths in config files
   - Ensure datasets are properly downloaded and extracted

3. **Checkpoint Loading Errors**

   - Verify pretrained checkpoint paths
   - Check model architecture compatibility

4. **Distributed Training Issues**
   - Verify master port availability
   - Check network connectivity between nodes
   - Ensure proper environment variable setup

### Performance Tips

1. **Faster Training**

   - Use more GPUs
   - Enable mixed precision (AMP)
   - Optimize data loading (increase num_workers)

2. **Memory Optimization**
   - Use gradient checkpointing
   - Reduce model size (use smaller variants)
   - Enable memory-efficient attention

## Results Interpretation

### Key Metrics

#### Classification

- **Top-1 Accuracy**: Overall classification accuracy
- **Top-5 Accuracy**: Top-5 classification accuracy

#### Segmentation

- **mIoU**: Mean Intersection over Union
- **mAcc**: Mean Accuracy
- **aAcc**: Average Accuracy

#### Detection

- **Bbox AP**: Bounding box Average Precision
- **Segm AP**: Segmentation Average Precision
- **AP50/AP75**: AP at IoU thresholds 0.5/0.75

### Expected Performance Patterns

1. **ZOH**: Baseline performance, fastest training
2. **FOH**: Moderate improvement, slightly slower
3. **Bilinear**: Good balance of accuracy and speed
4. **Polynomial**: Better accuracy, higher computational cost
5. **Higher-Order**: Improved accuracy, significant computational overhead
6. **RK4**: Best accuracy, highest computational cost

## Contributing

To add new discretization methods:

1. Implement the method in `mamba_ssm/ops/selective_scan_interface.py`
2. Add model variants in `vim/models_mamba.py`
3. Create corresponding config files
4. Add training scripts
5. Update comparison scripts

## Citation

If you use this work, please cite the original Vision Mamba paper and this discretization study:

```bibtex
@article{vision_mamba_2024,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project follows the same license as the original Vision Mamba implementation.

## Contact

For questions or issues related to the discretization experiments, please open an issue in the repository.
