# CUDA Kernel vs Python Reference Selection Guide

## Overview

You can now choose between using the custom CUDA kernel implementation or the Python reference implementation (PyTorch operations) for **all** discretization methods, including ZOH.

## Why This Matters

### Performance Differences

- **CUDA Kernel Path**: Custom optimized CUDA kernels (faster for some operations, but may have overhead)
- **Python Reference Path**: PyTorch's optimized BLAS/LAPACK operations (cuBLAS/cuSOLVER) - can be faster for matrix operations

### Historical Context

- **Before**: ZOH used CUDA kernel, others used Python reference → BILINEAR was faster due to optimized PyTorch matrix ops
- **Now**: All methods can use either path → You can compare and choose the best for your use case

## Usage

### Method 1: Environment Variable (Recommended for Scripts)

Set `VIM_USE_CUDA_KERNEL` environment variable:

```bash
# Force CUDA kernel
export VIM_USE_CUDA_KERNEL=1
python your_training_script.py

# Force Python reference
export VIM_USE_CUDA_KERNEL=0
python your_training_script.py

# Auto-select (try CUDA first, fallback to Python)
unset VIM_USE_CUDA_KERNEL
python your_training_script.py
```

### Method 2: Model Initialization Parameter

```python
from mamba_ssm.modules.mamba_simple import Mamba

# Force CUDA kernel
model = Mamba(
    d_model=512,
    discretization_method="zoh",
    use_cuda_kernel=True  # Force CUDA
)

# Force Python reference
model = Mamba(
    d_model=512,
    discretization_method="zoh",
    use_cuda_kernel=False  # Force Python
)

# Auto-select (default)
model = Mamba(
    d_model=512,
    discretization_method="zoh",
    use_cuda_kernel=None  # Auto (try CUDA first)
)
```

### Method 3: Direct Function Call

```python
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# Force CUDA kernel
out = selective_scan_fn(
    u, delta, A, B, C,
    discretization_method="zoh",
    use_cuda_kernel=True
)

# Force Python reference
out = selective_scan_fn(
    u, delta, A, B, C,
    discretization_method="zoh",
    use_cuda_kernel=False
)
```

## Updating Your Training Scripts

### Example: Update a Training Script

```bash
#!/bin/bash

# Option 1: Force Python reference (PyTorch operations)
export VIM_USE_CUDA_KERNEL=0

# Option 2: Force CUDA kernel
# export VIM_USE_CUDA_KERNEL=1

# Option 3: Auto-select (default behavior)
# unset VIM_USE_CUDA_KERNEL

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    ./main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 512 \
    --discretization_method zoh \
    # ... other args
```

## Performance Comparison

### Expected Results

| Method    | CUDA Kernel    | Python Reference | Notes                                 |
| --------- | -------------- | ---------------- | ------------------------------------- |
| ZOH       | Baseline       | May be faster    | Python uses optimized `torch.exp()`   |
| FOH       | Similar to ZOH | May be faster    | Python uses optimized matrix ops      |
| BILINEAR  | Scalar approx  | **Faster**       | Python uses cuBLAS for matrix inverse |
| POLY      | Similar to ZOH | May be faster    | Python uses optimized einsum          |
| HIGHORDER | Similar to ZOH | May be faster    | Python uses optimized ops             |
| RK4       | Similar to ZOH | May be faster    | Python uses optimized ops             |

### Why Python Reference Can Be Faster

1. **Optimized BLAS/LAPACK**: PyTorch uses highly optimized cuBLAS/cuSOLVER
2. **Better Memory Access**: PyTorch operations may have better memory patterns
3. **Kernel Launch Overhead**: Custom CUDA kernels have launch/sync overhead
4. **Matrix Operations**: For BILINEAR, PyTorch's `torch.inverse()` is highly optimized

## Recommendations

### For Fair Comparison

Use the **same implementation path** for all methods:

```bash
# Compare all methods with Python reference
export VIM_USE_CUDA_KERNEL=0
./run-all-discretization-experiments.sh

# Or compare all methods with CUDA kernel
export VIM_USE_CUDA_KERNEL=1
./run-all-discretization-experiments.sh
```

### For Maximum Performance

Test both paths and choose the faster one:

```bash
# Test ZOH with Python reference
export VIM_USE_CUDA_KERNEL=0
python train.py --discretization_method zoh

# Test ZOH with CUDA kernel
export VIM_USE_CUDA_KERNEL=1
python train.py --discretization_method zoh
```

### For Production

- **BILINEAR**: Likely faster with Python reference (`VIM_USE_CUDA_KERNEL=0`)
- **ZOH, FOH, POLY, HIGHORDER, RK4**: Test both, may vary by hardware/workload

## Troubleshooting

### CUDA Kernel Fails

If you force CUDA kernel (`use_cuda_kernel=True`) but it fails:

```python
# Will raise RuntimeError
model = Mamba(..., use_cuda_kernel=True)
```

**Solution**: Use auto-select or Python reference:

```python
model = Mamba(..., use_cuda_kernel=None)  # Auto-select
# or
model = Mamba(..., use_cuda_kernel=False)  # Force Python
```

### Check Which Path Is Used

```python
import os
os.environ['VIM_USE_CUDA_KERNEL'] = '1'  # or '0' or unset

# The model will use the specified path
model = Mamba(...)
```

## Example: Updated Training Script

```bash
#!/bin/bash
# Rorqual_pt-vim-zoh.sh

# Force Python reference for ZOH (to compare with BILINEAR)
export VIM_USE_CUDA_KERNEL=0

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    ./main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 512 \
    --discretization_method zoh \
    --seed ${SEED:-0} \
    # ... other args
```

## Summary

- **FOH has low throughput** because it also uses Python reference (same as BILINEAR before our changes)
- **ZOH can now use Python reference** by setting `VIM_USE_CUDA_KERNEL=0`
- **All methods can use either path** - choose based on performance testing
- **For fair comparison**: Use the same path for all methods
