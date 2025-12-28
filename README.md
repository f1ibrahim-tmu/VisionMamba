<div align="center">
<h1>State Space Models (S7) - VisionMambaUpgrades </h1>
<h3>Beyond ZOH: Advanced Discretization Strategies for State Space Models</h3>

[Fady Ibrahim](https://github.com/f1ibrahim-tmu)<sup>1</sup> \*

<sup>1</sup> Department of Computer Science, Toronto Metropolitan University
<sup>2</sup> Department of Aerospace Engineering, Toronto Metropolitan University

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

<!-- Conference? ([conference paper]()), ArXiv Preprint ([]()) -->

</div>

#

### News

<!-- * **` May. 2nd, 2024`:** Vision Mamba (Vim) is accepted by ICML2024. 🎉 Conference page can be found [here](https://icml.cc/virtual/2024/paper_metadata_from_author/33768).

* **` Feb. 10th, 2024`:** We update Vim-tiny/small weights and training scripts. By placing the class token at middle, Vim achieves improved results. Further details can be found in code and our updated [arXiv](https://arxiv.org/abs/2401.09417).

* **` Jun. 18th, 2024`:** We released our paper on Arxiv. Code/Models are coming soon. Please stay tuned! ☕️ -->

## Abstract

State Space Models, like the Vision Mamba architecture, apply discrete time principles to capture global dependencies in visual sequences with linear scalability. Current state space models rely on Zero Order Hold, simply treating inputs as constant between sampling steps. This reduces fidelity in dynamic visual settings, ultimately limiting the accuracy of modern state space models.
This paper conducts a controlled comparison of six discretization methods included in the Vision Mamba framework: Zero Order Hold (ZOH), First Order Hold (FOH), Bilinear/Tustin Transform (BIL), Polynomial Interpolation (POL), Higher Order Hold (HOH), and Runge–Kutta 4 (RK4). Each method is evaluated across core vision benchmarks to quantify its effect on accuracy and predictive performance in image classification, semantic segmentation, and object detection. Results show that Polynomial and Higher Order methods provide the highest accuracy gains, though with increased training cost. Bilinear delivers consistent improvements with minimal overhead, offering the strongest balance between precision and efficiency. We clarify the impact of discretization on these models and provide evident reasons for adopting Bilinear Transform as a default baseline for state-of-the-art SSM architectures, including all Mamba-based variants.

<!-- <div align="center">
<img />
</div>

## Overview
<div align="center">
<img />
</div> -->

# Repo Overview

A comprehensive implementation of improved State Space Models (SSMs) for vision tasks, featuring enhanced Mamba architectures optimized for visual representation learning. This repository provides efficient, scalable models for image classification, object detection, and semantic segmentation.

This codebase implements advanced vision models built upon State Space Models, offering significant improvements over traditional Transformer-based architectures. The models leverage bidirectional state space processing, multiple discretization methods, and optimized architectures to achieve superior performance with improved computational efficiency.

### Key Features

- **Bidirectional State Space Processing**: Enhanced bidirectional Mamba blocks that capture global context efficiently
- **Multiple Discretization Methods**: Support for various discretization techniques including ZOH, FOH, Bilinear, Polynomial, High-Order, and RK4 methods
- **Flexible Architecture**: Configurable model variants (tiny, small, base) with customizable depth and embedding dimensions
- **Advanced Position Encoding**: Support for absolute position embeddings and Rotary Position Embedding (RoPE)
- **Efficient Memory Usage**: Optimized implementations that significantly reduce GPU memory requirements compared to Transformer-based models
- **Multi-Task Support**: Unified framework for classification, detection, and segmentation tasks
- **High-Resolution Capability**: Efficient processing of high-resolution images with improved scalability

## Architecture Improvements

### State Space Model Enhancements

The implementation includes several key improvements to Mamba for vision applications:

- **Bidirectional Processing**: Enables context-aware feature extraction from both forward and backward directions
- **Advanced Discretization**: Multiple discretization methods for improved numerical stability and accuracy
- **Position-Aware Design**: Position embeddings and RoPE integration for handling position-sensitive visual data
- **Flexible Class Token Placement**: Support for head, middle, and tail class token positioning strategies
- **Optimized Pooling**: Multiple pooling strategies for final feature extraction

### Model Variants

- **Tiny**: Lightweight model with ~7M parameters, suitable for resource-constrained environments
- **Small**: Balanced model with ~26M parameters, offering good performance-efficiency trade-off
- **Base**: Larger model with ~98M parameters, optimized for maximum performance

## Installation

### Environment Setup

#### NVIDIA GPUs

```bash
# Create conda environment
conda create -n visionmamba python=3.10.13
conda activate visionmamba

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### Dependencies

Install required packages:

<!-- ```bash
# Install base requirements
pip install -r vim/vim_requirements.txt

# Install causal_conv1d
pip install -e causal_conv1d

# Install mamba-ssm
pip install -e mamba-1p1p1
``` -->

## Quick Start

### Training

#### Pretraining

<!-- Train a model from scratch:

```bash
bash vim/scripts/pt-vim-t.sh
```

#### Fine-tuning

Fine-tune with finer granularity:

```bash
bash vim/scripts/ft-vim-t.sh
``` -->

### Evaluation

<!-- Evaluate a trained model on ImageNet-1K:

```bash
python vim/main.py \
    --eval \
    --resume /path/to/checkpoint \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --data-path /path/to/imagenet
``` -->

## Model Configuration

### Key Parameters

- `discretization_method`: Choose from `"zoh"`, `"foh"`, `"poly"`, `"bilinear"`, `"highorder"`, or `"rk4"`
- `if_bidirectional`: Enable bidirectional state space processing
- `if_rope`: Enable Rotary Position Embedding
- `use_middle_cls_token`: Place class token in the middle of the sequence
- `final_pool_type`: Pooling strategy for final feature extraction
- `if_abs_pos_embed`: Use absolute position embeddings

### Example Configuration

```python
model = VisionMamba(
    img_size=224,
    patch_size=16,
    depth=24,
    embed_dim=192,
    d_state=16,
    discretization_method="zoh",
    if_bidirectional=True,
    if_rope=True,
    use_middle_cls_token=True,
    final_pool_type='mean'
)
```

## Performance Characteristics

The models demonstrate:

- **Computational Efficiency**: Significantly faster inference compared to Transformer-based models
- **Memory Efficiency**: Reduced GPU memory footprint, enabling processing of higher resolution images
- **Scalability**: Linear scaling with sequence length, making it suitable for high-resolution vision tasks
- **Accuracy**: Competitive or superior performance on standard vision benchmarks

## Supported Tasks

### Image Classification

Full support for ImageNet-1K classification with configurable model sizes and training strategies.

### Object Detection

Integration with detection frameworks for object detection tasks on COCO and other datasets.

### Semantic Segmentation

Support for semantic segmentation tasks with configurable backbones and decoders.

## Implementation Details

### CUDA Implementation: Custom Fusion Kernels vs Python Reference

The implementation provides two execution paths for all discretization methods, allowing users to choose between optimized CUDA kernels and PyTorch's optimized BLAS operations.

#### Custom CUDA Fusion Kernels

Custom CUDA kernels are implemented for all discretization methods, providing:

- **Fused Operations**: Multiple operations combined into single kernel launches to reduce overhead
- **Optimized Memory Access**: Coalesced memory access patterns for better GPU utilization
- **Scalar Approximations**: Efficient scalar operations for diagonal state matrices (common in SSMs)
- **Hardware-Specific Optimizations**: Tailored for modern GPU architectures

The CUDA kernels are located in `mamba-1p1p1/csrc/selective_scan/` and include:

- Forward pass kernels with support for all discretization methods
- Discretization-specific computation in `discretization_kernels.cuh`
- Automatic fallback to Python reference if kernel fails

#### Python Reference Implementation

The Python reference implementation leverages PyTorch's highly optimized operations:

- **cuBLAS Integration**: Uses NVIDIA's cuBLAS library for matrix operations
- **cuSOLVER Integration**: Leverages cuSOLVER for matrix inversions (critical for Bilinear method)
- **Optimized Einsum**: PyTorch's optimized einsum operations for tensor contractions
- **Better Memory Patterns**: PyTorch's memory management may provide better access patterns for some workloads

#### Controlling Execution Path

You can control which implementation is used:

**Environment Variable:**

```bash
# Force CUDA kernel
export VIM_USE_CUDA_KERNEL=1

# Force Python reference (cuBLAS)
export VIM_USE_CUDA_KERNEL=0

# Auto-select (try CUDA first, fallback to Python)
unset VIM_USE_CUDA_KERNEL
```

**Model Initialization:**

```python
model = VisionMamba(
    ...,
    use_cuda_kernel=True,   # Force CUDA
    # use_cuda_kernel=False, # Force Python reference
    # use_cuda_kernel=None,  # Auto-select (default)
)
```

### Discretization Methods: Detailed Implementation

All discretization methods use Taylor series expansions to avoid numerical instabilities and expensive matrix operations. The implementations are optimized for both CUDA kernels and Python reference paths.

#### ZOH (Zero-Order Hold)

The simplest and most commonly used discretization method:

**Mathematical Formulation:**

- `A_d = exp(A·Δ)`
- `B_d = Δ·B`

**Implementation:**

- CUDA: Uses `exp2f` for efficient exponential computation
- Python: Uses `torch.exp()` with optimized einsum operations
- **Taylor Series**: Not required (direct exponential)

**Use Case**: Baseline method, fastest for simple state transitions.

#### FOH (First-Order Hold)

Provides improved accuracy for smoother transitions:

**Mathematical Formulation:**

- `A_d = exp(A·Δ)`
- `B_d = A⁻² · (exp(A·Δ) - I - A·Δ) · B`

**Taylor Series Expansion:**
To avoid division by A², we use the Taylor series:

```
(exp(A·Δ) - 1 - A·Δ) / A² = Δ²/2! + A·Δ³/3! + A²·Δ⁴/4! + A³·Δ⁵/5! + ...
```

**Implementation:**

- **Coefficients**: `B_d = (Δ²/2 + A·Δ³/6 + A²·Δ⁴/24 + A³·Δ⁵/120) · B`
- CUDA: Computes powers of delta and A, then combines terms
- Python: Uses einsum for efficient tensor operations
- **Numerical Stability**: Taylor expansion avoids division by small A values

**Use Case**: Better accuracy for smooth input signals, slightly slower than ZOH.

#### Bilinear (Tustin Transform)

Stability-preserving bilinear transformation:

**Mathematical Formulation:**

- `Ā = (I - ΔA/2)⁻¹ · (I + ΔA/2)`
- `B̄ = (I - ΔA/2)⁻¹ · ΔB`

**Implementation:**

- **Matrix Inversion**: Requires computing `(I - ΔA/2)⁻¹` for each timestep
- CUDA: Uses scalar approximation `(1 - A·Δ/2)/(1 + A·Δ/2)` for diagonal A (exact)
- Python: Uses `torch.inverse()` with cuBLAS/cuSOLVER (handles non-diagonal A)
- **Stability**: Maps left half-plane to unit circle, preserving stability

**Use Case**: Best stability properties, often faster with Python reference due to optimized cuBLAS.

#### Polynomial Interpolation

Non-causal method using polynomial interpolation:

**Mathematical Formulation:**

- `B̄ = A⁻¹(exp(A·Δ) - I)B + ½A⁻²(exp(A·Δ) - I - A·Δ)B`

**Taylor Series Expansion:**
Combines ZOH and FOH terms:

```
A⁻¹(exp(A·Δ) - I) = Δ + A·Δ²/2 + A²·Δ³/6 + A³·Δ⁴/24
½A⁻²(exp(A·Δ) - I - A·Δ) = Δ²/4 + A·Δ³/12 + A²·Δ⁴/48
```

**Combined Coefficients:**

- `B̄ = (Δ + (A/2 + 1/4)Δ² + (A²/6 + A/12)Δ³ + (A³/24 + A²/48)Δ⁴) · B`

**Implementation:**

- CUDA: Computes combined Taylor terms efficiently
- Python: Uses einsum for tensor operations
- **Non-Causal**: Requires bidirectional scan (handled in Python reference)

**Use Case**: Smooth interpolation between points, useful for high-quality reconstruction.

#### High-Order (Quadratic Hold, n=2)

Generalized higher-order method combining multiple terms:

**Mathematical Formulation:**

- Combines ZOH (n=0), FOH (n=1), and Quadratic (n=2) terms
- `B̄ = Σ(i=0 to n) A^(-(i+1)) · [exp(A·Δ) - Σ(k=0 to i)(A·Δ)^k/k!] / i! · B`

**Taylor Series Expansion:**

```
n=0 (ZOH): Δ + A·Δ²/2 + A²·Δ³/6 + A³·Δ⁴/24
n=1 (FOH): Δ²/2 + A·Δ³/6 + A²·Δ⁴/24
n=2: Δ³/12 + A·Δ⁴/48
```

**Combined Coefficients:**

- `B̄ = (Δ + (A/2 + 1/2)Δ² + (A²/6 + A/6 + 1/12)Δ³ + (A³/24 + A²/24 + A/48)Δ⁴) · B`

**Implementation:**

- CUDA: Efficiently computes all combined terms
- Python: Uses optimized tensor operations
- **Accuracy**: Higher-order terms provide better approximation

**Use Case**: Maximum accuracy for complex state transitions.

#### RK4 (Runge-Kutta 4th Order)

Fourth-order numerical integration method:

**Mathematical Formulation:**

- `A_d = exp(A·Δ)`
- `B_d = (A⁻¹) · (A_d - I) · B` with RK4 coefficients

**RK4 Coefficients:**
Uses four evaluation points (k1, k2, k3, k4):

- `k1 = Δ · B`
- `k2 = Δ² · A·B / 2`
- `k3 = Δ³ · A²·B / 6`
- `k4 = Δ⁴ · A³·B / 24`
- `B_d = k1 + k2 + k3 + k4`

**Implementation:**

- CUDA: Computes A powers and combines RK4 terms
- Python: Uses einsum for efficient tensor operations
- **Accuracy**: Highest accuracy among all methods

**Use Case**: Maximum numerical accuracy for critical applications.

### Bidirectional Processing

The bidirectional Mamba blocks process sequences in both directions, enabling better context understanding while maintaining computational efficiency. This is particularly important for vision tasks where spatial relationships exist in both horizontal and vertical directions.

### Position Encoding

- **Absolute Position Embeddings**: Learnable position embeddings added to patch tokens, providing explicit spatial information
- **Rotary Position Embedding (RoPE)**: Relative position encoding that generalizes to different sequence lengths, enabling better handling of variable-resolution inputs

## Modern Library Upgrades

The codebase has been upgraded to use the latest versions of the OpenMMLab ecosystem, providing improved performance, better PyTorch 2.x compatibility, and enhanced features.

### MMCV 2.x Migration

Upgraded to MMCV 2.x for:

- **PyTorch 2.1+ Compatibility**: Full support for PyTorch 2.1.0 and later
- **CUDA 12.2 Support**: Optimized for modern CUDA versions and H100 GPUs
- **Improved Performance**: Better memory management and optimized operations
- **MMEngine Integration**: Modular architecture with MMEngine as the foundation

**Key Changes:**

- `mmcv.runner` → `mmengine.runner`
- `mmcv.Config` → `mmengine.Config`
- `mmcv.parallel` → `mmengine.model`
- Updated config format with `train_cfg`/`val_cfg`/`test_cfg` instead of `runner`
- `optimizer_config` → `optim_wrapper` with `AmpOptimWrapper` for FP16

### MMEngine Integration

MMEngine ≥ 0.10.0 provides:

- **Unified Runner API**: Single `Runner` class with configurable training/validation/test loops
- **Modern Hook System**: Improved hook registration and execution
- **Better Distributed Training**: Enhanced multi-GPU and multi-node support
- **Optimized Data Loading**: Improved dataloader configuration and performance

**Configuration Updates:**

```python
# Old format (MMCV 1.x)
runner = dict(type='IterBasedRunner', max_iters=60000)
data = dict(samples_per_gpu=8, workers_per_gpu=16)

# New format (MMEngine ≥ 0.7)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)
train_dataloader = dict(batch_size=8, num_workers=16, sampler=dict(type='InfiniteSampler'))
```

### MMSegmentation 1.0.0+ Support

Upgraded to MMSegmentation ≥ 1.0.0 with:

- **Updated Registry System**: New registry location in `mmseg.registry`
- **Modern API**: Updated evaluation hooks and testing functions
- **Better Integration**: Improved compatibility with MMEngine
- **Enhanced Features**: New segmentation models and techniques

**Key Updates:**

- `mmseg.models.builder` → `mmseg.registry.MODELS`
- `mmseg.core` → `mmseg.engine` for evaluation hooks
- `mmseg.apis.multi_gpu_test` → `Runner.test()` method

### MMDetection 3.0.0+ Compatibility

Upgraded to MMDetection ≥ 3.0.0 for:

- **MMCV 2.x Compatibility**: Required for MMCV 2.x support
- **Modern Detection Models**: Latest detection architectures
- **Improved Performance**: Optimized detection pipelines
- **Better Integration**: Seamless integration with MMEngine

### Migration Benefits

- **Better Performance**: Optimized operations and memory management
- **PyTorch 2.x Features**: Full support for compilation, torch.compile, and new optimizations
- **Modern GPU Support**: Optimized for H100, A100, and other modern architectures
- **Improved Maintainability**: Cleaner API and better code organization
- **Enhanced Features**: Access to latest models and techniques from OpenMMLab

## Development

This is an active codebase that will be regularly updated with:

- New model architectures and improvements
- Additional discretization methods
- Performance optimizations
- Extended task support
- Better training strategies

## Directory Structure

```
VisionMamba/
├── vim/                    # Main vision model implementation
│   ├── models_mamba.py    # Model architectures
│   ├── main.py            # Training and evaluation scripts
│   └── scripts/           # Training scripts
├── mamba-1p1p1/          # State space model core implementation
├── causal-conv1d/        # Causal convolution utilities
├── det/                   # Object detection support
├── seg/                   # Semantic segmentation support
└── documentation/         # Additional documentation
```

## Notes

- The implementation supports both training from scratch and fine-tuning from pretrained weights
- GPU configurations are supported (NVIDIA CUDA)
- The codebase is designed for extensibility and easy experimentation with new architectures
- Performance characteristics may vary based on hardware configuration and discretization method selection

## Future Improvements

Planned enhancements include:

- Additional model variants and scaling strategies
- Further optimizations
- Extended multi-task learning support
- Improved training efficiency
- Additional vision tasks integration
