# Custom CUDA Kernels for Discretization Methods

This branch (`custom_cuda`) extends the Vision Mamba implementation to support custom CUDA kernels for all discretization methods, not just ZOH. This addresses the performance discrepancy where BILINEAR and other methods were faster than ZOH because they used optimized PyTorch operations while ZOH used a custom CUDA kernel.

## Changes Made

### 1. Profiling Script (`profile_discretization.py`)

A comprehensive profiling script that measures:

- Individual operation performance (exp vs matrix inverse)
- Full discretization method performance via `selective_scan_fn`
- Throughput comparisons between methods

**Usage:**

```bash
python profile_discretization.py --batch 8 --dim 512 --seqlen 1024 --dstate 16 --n_trials 100
```

### 2. CUDA Kernel Extensions

#### Modified Files:

- **`mamba-1p1p1/csrc/selective_scan/selective_scan.h`**: Added `DiscretizationMethod` enum and `discretization_method` field to `SSMParamsBase`
- **`mamba-1p1p1/csrc/selective_scan/selective_scan.cpp`**: Updated `selective_scan_fwd` to accept `discretization_method` parameter
- **`mamba-1p1p1/csrc/selective_scan/selective_scan_fwd_kernel.cuh`**: Modified to use discretization helper function
- **`mamba-1p1p1/csrc/selective_scan/discretization_kernels.cuh`**: New file containing CUDA implementations for all discretization methods
- **`mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py`**: Updated to pass discretization method to CUDA kernel

#### New File:

- **`mamba-1p1p1/csrc/selective_scan/discretization_kernels.cuh`**: Contains `compute_discretization()` template function that implements:
  - **ZOH**: `A_d = exp(A*delta)`, `B_d = delta * B`
  - **FOH**: `A_d = exp(A*delta)`, `B_d = (exp(A*delta) - 1) / A * B`
  - **BILINEAR**: `A_d = (1 - A*delta/2) / (1 + A*delta/2)`, `B_d = delta * B / (1 + A*delta/2)` (scalar approximation)
  - **POLY**: `A_d = exp(A*delta)`, `B_d = delta*B + delta^2*A*B/2 + delta^3*A^2*B/6`
  - **HIGHORDER**: `A_d = exp(A*delta)`, `B_d = delta*B + delta^2*A*B/2`
  - **RK4**: `A_d = exp(A*delta)`, `B_d = delta*B + delta^2*A*B/2 + delta^3*A^2*B/6 + delta^4*A^3*B/24`

## Implementation Details

### Scalar vs Matrix Operations

The current implementation uses scalar approximations for all methods. This works because:

1. The SSM processes each state dimension independently
2. For diagonal A matrices (common in SSMs), scalar operations are exact
3. This avoids expensive matrix inversions in the hot path

### BILINEAR Special Case

BILINEAR requires matrix inversions for the general case: `A_d = (I + A*delta/2)^-1 * (I - A*delta/2)`. The current implementation uses a scalar approximation:

- `A_d = (1 - A*delta/2) / (1 + A*delta/2)`
- `B_d = delta * B / (1 + A*delta/2)`

This is exact when A is diagonal. For non-diagonal A, a full matrix inversion kernel would be needed, which is more complex and may not provide speedup over PyTorch's optimized BLAS routines.

## Building

To rebuild the CUDA extensions with the new kernels:

```bash
cd mamba-1p1p1
python setup.py build_ext --inplace
# Or if using pip:
pip install -e .
```

## Testing

1. **Run the profiling script:**

   ```bash
   python profile_discretization.py --profile_ops --profile_scan
   ```

2. **Test individual methods:**

   ```python
   from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
   import torch

   # Create test tensors
   u = torch.randn(8, 512, 1024, device='cuda')
   delta = torch.randn(8, 512, 1024, device='cuda') * 0.1
   A = torch.randn(512, 16, device='cuda') * 0.1
   B = torch.randn(512, 16, device='cuda')
   C = torch.randn(512, 16, device='cuda')

   # Test each method
   for method in ["zoh", "foh", "bilinear", "poly", "highorder", "rk4"]:
       out = selective_scan_fn(u, delta, A, B, C, discretization_method=method)
       print(f"{method}: {out.shape}")
   ```

## Performance Expectations

With custom CUDA kernels for all methods:

- **ZOH**: Should maintain current performance (baseline)
- **FOH, POLY, HIGHORDER, RK4**: Should see performance improvements as they now use optimized CUDA kernels instead of Python reference
- **BILINEAR**: May see improvement, but the scalar approximation may not match PyTorch's optimized matrix operations for non-diagonal A

## Future Improvements

1. **Full Matrix BILINEAR**: Implement a specialized CUDA kernel for small matrix inversions (dstate x dstate, typically 16x16) to handle non-diagonal A matrices
2. **Backward Pass**: Extend the backward pass kernels to support all discretization methods
3. **Benchmarking**: Add comprehensive benchmarks comparing CUDA kernel vs Python reference for each method

## Notes

- The implementation falls back to the Python reference implementation if the CUDA kernel fails
- Complex A matrices are currently handled with a simplified approach (falls back to ZOH-like behavior for some methods)
- The scalar approximation for BILINEAR is exact for diagonal A matrices, which is the common case in SSMs
