# Performance Verification Summary

This document confirms which of the potential performance issues apply to your Vision Mamba discretization implementation.

## Confirmed Issues

### ✅ 1. Implementation/Path Differences (CONFIRMED - PRIMARY ISSUE)

**Status: CONFIRMED - This is the main cause**

- **ZOH**: Uses optimized CUDA kernel (`selective_scan_cuda.fwd`) with `exp2f()` operations
- **BILINEAR/FOH/POLY/HIGHORDER/RK4**: Previously used Python reference implementation (`selective_scan_ref`) with PyTorch operations
- **Impact**: The Python reference uses optimized BLAS/LAPACK (cuBLAS/cuSOLVER) which can be faster than the custom CUDA kernel for certain operations

**Evidence:**

- `selective_scan_interface.py` lines 80-104: Different code paths based on `discretization_method`
- Before our changes: Only ZOH used CUDA kernel, others fell back to Python
- After our changes: All methods attempt CUDA kernel first, with fallback

**Fix Applied:**

- Custom CUDA kernels now implemented for all methods
- All methods use the same CUDA kernel path (with method-specific discretization logic)

### ✅ 2. GPU Synchronization (PARTIALLY CONFIRMED)

**Status: NEEDS VERIFICATION IN YOUR BENCHMARKS**

**Current State:**

- `profile_discretization.py` uses `torch.cuda.synchronize()` correctly (lines 32, 37, 40, 59, 63, etc.)
- `vim/engine.py` uses `torch.cuda.synchronize()` after backward pass (line 101)
- Training scripts may not all use synchronization consistently

**Recommendation:**

- Run `verify_performance_issues.py --skip-checks 1,3,4,5,6,7` to check your specific setup
- Ensure all throughput measurements include `torch.cuda.synchronize()` before/after timing

### ⚠️ 3. Threading/BLAS/OpenMP Differences (POTENTIAL ISSUE)

**Status: INCONSISTENT ACROSS SCRIPTS**

**Evidence:**

- Some scripts use `OMP_NUM_THREADS=4` (CIFAR-100 scripts)
- Some scripts use `OMP_NUM_THREADS=16` (detection/segmentation scripts)
- Some scripts use `OMP_NUM_THREADS=1` (single GPU scripts)

**Files with OMP_NUM_THREADS:**

- `vim/scripts/CVIS/cifar100/*.sh`: `OMP_NUM_THREADS=4`
- `det/scripts/discretization/CVIS/*.sh`: `OMP_NUM_THREADS=16`
- `seg/scripts/discretization/CVIS/*.sh`: `OMP_NUM_THREADS=16`
- `vim/scripts/CVIS/pt-vim-rk4.sh`: `OMP_NUM_THREADS=1`

**Recommendation:**

- Use consistent `OMP_NUM_THREADS` across all methods for fair comparison
- Set `OMP_NUM_THREADS=1` for GPU-bound workloads (CPU threads don't help GPU ops)

### ❌ 4. Different Code Paths/Preprocessing (NOT CONFIRMED)

**Status: NOT AN ISSUE**

- All methods use the same input preprocessing (contiguous checks, reshaping)
- Same batch size, dtype, and tensor shapes
- No method-specific preprocessing differences found

### ⚠️ 5. Caching/Warm-up (POTENTIAL ISSUE)

**Status: NEEDS VERIFICATION**

**Potential Issues:**

- First few iterations may be slower due to:
  - CUDA kernel JIT compilation
  - CPU frequency scaling
  - Memory allocation
  - cuDNN algorithm selection

**Recommendation:**

- Skip first 5-10 iterations in throughput measurements
- Use `verify_performance_issues.py` to check warm-up effects

### ❌ 6. Memory Access/Vectorization (NOT APPLICABLE)

**Status: NOT AN ISSUE FOR THIS CODEBASE**

- All operations use PyTorch/CUDA tensors (vectorized by default)
- No per-pixel Python loops in discretization code
- Memory access patterns are similar across methods

## Quick Checks to Run

### 1. Verify Identical Inputs

```bash
python verify_performance_issues.py --skip-checks 2,3,4,5,6,7
```

### 2. Check GPU Synchronization

```bash
python verify_performance_issues.py --skip-checks 1,3,4,5,6,7
```

### 3. Check Threading Configuration

```bash
python verify_performance_issues.py --skip-checks 1,2,4,5,6,7
```

### 4. Verify Code Paths

```bash
python verify_performance_issues.py --skip-checks 1,2,3,5,6,7
```

### 5. Check Warm-up Effects

```bash
python verify_performance_issues.py --skip-checks 1,2,3,4,6,7
```

### 6. Micro-benchmark Operations

```bash
python verify_performance_issues.py --skip-checks 1,2,3,4,5,7
```

### 7. Full Method Comparison

```bash
python verify_performance_issues.py --skip-checks 1,2,3,4,5,6
```

### Run All Checks

```bash
python verify_performance_issues.py
```

## Recommendations

### Immediate Actions

1. **Rebuild CUDA Extensions**: After our custom kernel changes

   ```bash
   cd mamba-1p1p1
   python setup.py build_ext --inplace
   ```

2. **Standardize Threading**: Use consistent `OMP_NUM_THREADS=1` for GPU workloads

   ```bash
   # Add to all training scripts:
   export OMP_NUM_THREADS=1
   ```

3. **Verify Synchronization**: Ensure all throughput measurements use:

   ```python
   torch.cuda.synchronize()
   start = time.perf_counter()
   # ... operation ...
   torch.cuda.synchronize()
   end = time.perf_counter()
   ```

4. **Skip Warmup**: In benchmarks, skip first 5-10 iterations:
   ```python
   for i in range(n_trials + n_warmup):
       if i >= n_warmup:
           # Start timing here
   ```

### Long-term Improvements

1. **Profile Individual Operations**: Use `verify_performance_issues.py` check 6 to identify bottlenecks
2. **Compare CUDA vs Python Paths**: Verify CUDA kernels are actually being used
3. **Benchmark with Consistent Settings**: Use same OMP_NUM_THREADS, batch size, etc. for all methods

## Expected Results After Fixes

With custom CUDA kernels for all methods:

- **ZOH**: Should maintain current performance (baseline)
- **FOH, POLY, HIGHORDER, RK4**: Should see performance improvements (now use CUDA kernels)
- **BILINEAR**: May see improvement, but scalar approximation may not match PyTorch's optimized matrix ops

The +63.76% throughput gain for BILINEAR was likely due to:

1. PyTorch's optimized cuBLAS/cuSOLVER being faster than the custom CUDA kernel's `exp2f()` path
2. Better memory access patterns in PyTorch's matrix operations
3. Potential overhead in the custom CUDA kernel (kernel launch, synchronization)

With our new implementation, all methods use the same CUDA kernel infrastructure, so differences should be due to the actual discretization math, not implementation paths.
