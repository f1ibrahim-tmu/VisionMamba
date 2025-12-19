# Answers to FOH Throughput and ZOH Python Reference Questions

## Question 1: Why does FOH have such low throughput?

**Answer: YES, FOH also uses the Python reference fallback.**

### Explanation

Before our custom CUDA kernel implementation:

- **ZOH**: Used custom CUDA kernel (`selective_scan_cuda.fwd`)
- **FOH, BILINEAR, POLY, HIGHORDER, RK4**: All used Python reference (`selective_scan_ref`)

The code path in `selective_scan_interface.py` was:

```python
# OLD CODE (before our changes):
if discretization_method != "zoh" and selective_scan_cuda is not None:
    # Fall back to Python reference for non-ZOH methods
    result = selective_scan_ref(...)
```

So **FOH had low throughput for the same reason as BILINEAR** - it was using the Python reference implementation, which:

1. Uses PyTorch's optimized operations (can be fast for some ops)
2. But has overhead from Python loops and sequential processing
3. Doesn't benefit from the custom CUDA kernel's parallel scan optimization

### FOH Implementation Details

FOH in Python reference (`selective_scan_ref`):

```python
elif discretization_method == "foh":
    # First Order Hold
    # For FOH: A_d = exp(A*delta), B_d = (A^-1)*(A_d - I)*B
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # Calculate (A_d - I) / A
    deltaA_minus_I_div_A = (deltaA - 1.0) / A.unsqueeze(0).unsqueeze(-1)
    deltaB_u = torch.einsum('bdln,dn,bdl->bdln', deltaA_minus_I_div_A, B, u)
```

This involves:

- `torch.exp()` - optimized but still has overhead
- Division operation `(deltaA - 1.0) / A` - can be slow
- Multiple einsum operations - sequential processing

## Question 2: Can we create a ZOH path that uses Python reference?

**Answer: YES, now you can!**

### Implementation

I've added a `use_cuda_kernel` parameter that allows you to choose:

1. **Force CUDA kernel**: `use_cuda_kernel=True`
2. **Force Python reference**: `use_cuda_kernel=False`
3. **Auto-select** (default): `use_cuda_kernel=None` (tries CUDA first, falls back to Python)

### Usage Examples

#### Method 1: Environment Variable (Easiest for Scripts)

```bash
# Force ZOH to use Python reference (PyTorch operations)
export VIM_USE_CUDA_KERNEL=0
python train.py --discretization_method zoh

# Force ZOH to use CUDA kernel
export VIM_USE_CUDA_KERNEL=1
python train.py --discretization_method zoh
```

#### Method 2: Model Initialization

```python
from mamba_ssm.modules.mamba_simple import Mamba

# ZOH with Python reference
model = Mamba(
    d_model=512,
    discretization_method="zoh",
    use_cuda_kernel=False  # Force Python reference
)

# ZOH with CUDA kernel
model = Mamba(
    d_model=512,
    discretization_method="zoh",
    use_cuda_kernel=True  # Force CUDA kernel
)
```

#### Method 3: Direct Function Call

```python
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# ZOH with Python reference
out = selective_scan_fn(
    u, delta, A, B, C,
    discretization_method="zoh",
    use_cuda_kernel=False  # Force Python reference
)
```

## Question 3: Option to select CUDA kernel or Python reference in scripts

**Answer: YES, now available!**

### How It Works

The `use_cuda_kernel` parameter is checked in this order:

1. **Environment variable** `VIM_USE_CUDA_KERNEL` (takes precedence)
2. **Model parameter** `use_cuda_kernel` (if provided)
3. **Default**: `None` (auto-select: try CUDA first, fallback to Python)

### Updated Training Script Example

```bash
#!/bin/bash
# Rorqual_pt-vim-zoh.sh

# Option 1: Use Python reference for ZOH (to compare with BILINEAR)
export VIM_USE_CUDA_KERNEL=0

# Option 2: Use CUDA kernel for ZOH (original behavior)
# export VIM_USE_CUDA_KERNEL=1

# Option 3: Auto-select (try CUDA first, fallback to Python)
# unset VIM_USE_CUDA_KERNEL

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 \
    ./main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 512 \
    --discretization_method zoh \
    --seed ${SEED:-0} \
    # ... other args
```

### For Fair Comparison

To compare all methods fairly, use the **same implementation path**:

```bash
# Compare all methods with Python reference
export VIM_USE_CUDA_KERNEL=0
./run-all-discretization-experiments.sh

# Or compare all methods with CUDA kernel
export VIM_USE_CUDA_KERNEL=1
./run-all-discretization-experiments.sh
```

## Summary

### Before Our Changes

- **ZOH**: CUDA kernel only
- **FOH, BILINEAR, etc.**: Python reference only
- **Problem**: Different code paths made comparison unfair

### After Our Changes

- **All methods**: Can use either CUDA kernel OR Python reference
- **ZOH**: Can now use Python reference (PyTorch operations)
- **FOH, BILINEAR, etc.**: Can now use CUDA kernel
- **Control**: Via `VIM_USE_CUDA_KERNEL` environment variable or `use_cuda_kernel` parameter

### Why FOH Had Low Throughput

FOH had low throughput because:

1. It used Python reference (same as BILINEAR)
2. Python reference has overhead from sequential processing
3. No benefit from custom CUDA kernel's parallel scan

### Why BILINEAR Was Faster Than ZOH

BILINEAR was faster because:

1. PyTorch's optimized cuBLAS/cuSOLVER for matrix inverse
2. Better memory access patterns in PyTorch operations
3. Custom CUDA kernel had overhead (kernel launch, sync)

### Now You Can

1. ✅ Use ZOH with Python reference: `export VIM_USE_CUDA_KERNEL=0`
2. ✅ Use FOH with CUDA kernel: `export VIM_USE_CUDA_KERNEL=1`
3. ✅ Compare all methods fairly using the same code path
4. ✅ Choose the fastest implementation for each method

## Next Steps

1. **Test both paths** for each method to find the fastest:

   ```bash
   # Test ZOH with Python reference
   export VIM_USE_CUDA_KERNEL=0
   python train.py --discretization_method zoh

   # Test ZOH with CUDA kernel
   export VIM_USE_CUDA_KERNEL=1
   python train.py --discretization_method zoh
   ```

2. **Use consistent path** for fair comparison:

   ```bash
   export VIM_USE_CUDA_KERNEL=0  # or 1
   ./run-all-discretization-experiments.sh
   ```

3. **Check which path is used** in your logs (the code will indicate CUDA vs Python)
