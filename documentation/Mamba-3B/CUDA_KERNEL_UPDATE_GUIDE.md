# CUDA Kernel Update Guide for Full A Matrices (Mamba-3B)

## Overview

This document outlines the changes needed to update CUDA kernels to support full A matrices (block-diagonal + low-rank structure) instead of just diagonal A matrices.

## Current Status

- ✅ Python reference implementation updated to handle full A matrices
- ✅ Interface detects full A matrices and forces Python path
- ⚠️ CUDA kernels still expect diagonal A (shape: `d_inner, d_state`)
- ⚠️ Full A matrices have shape `(d_inner, d_state, d_state)`

## Required Changes

### 1. Update SSMParamsBase Structure

**File**: `csrc/selective_scan/selective_scan.h` or similar

Add a flag to indicate if A is full matrix:
```cpp
struct SSMParamsBase {
    // ... existing fields ...
    bool is_full_A_matrix;  // New: true if A is (d_inner, d_state, d_state), false if (d_inner, d_state)
    int A_matrix_stride;     // New: stride for accessing A matrices when full
};
```

### 2. Update Kernel Interface

**File**: `csrc/selective_scan/selective_scan_fwd_kernel.cuh`

#### Current Implementation (Diagonal A):
```cpp
weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
// A[state_idx * params.A_dstate_stride + r * params.A_d_stride] gives diagonal element
```

#### New Implementation (Full A):
```cpp
if (params.is_full_A_matrix) {
    // A is (d_inner, d_state, d_state)
    // Access A[d, i, j] for full matrix
    weight_t *A_matrix = reinterpret_cast<weight_t *>(params.A_ptr) + 
                         dim_id * params.dstate * params.dstate * params.A_d_stride;
} else {
    // Original diagonal A access
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
}
```

### 3. Update Discretization Operations

**File**: `csrc/selective_scan/discretization_kernels.cuh`

#### ZOH (Zero Order Hold)

**Current (Diagonal)**:
```cpp
// For diagonal A: exp(delta * A) is element-wise
float A_val = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
float deltaA_val = exp2f(delta_val * A_val * kLog2e);
```

**New (Full Matrix)**:
```cpp
if (params.is_full_A_matrix) {
    // Matrix exponential: exp(delta * A[d])
    // Use batched matrix exponential or iterative method
    // For small matrices (d_state <= 16), can use series expansion:
    // exp(A) = I + A + A²/2! + A³/3! + ...
    // Or use CUDA's cuSOLVER for larger matrices
} else {
    // Original diagonal code
}
```

#### Bilinear Transform

**Current**: Already handles matrices for bilinear, but assumes A is diagonal

**New**: Update to handle full A matrices directly:
```cpp
if (params.is_full_A_matrix) {
    // A is already full matrix, use directly
    // (I - ΔA/2)⁻¹(I + ΔA/2) where A is full matrix
} else {
    // Convert diagonal A to matrix form for bilinear
}
```

### 4. Update State Update Operations

**File**: `csrc/selective_scan/selective_scan_fwd_kernel.cuh`

#### Current (Diagonal):
```cpp
// Element-wise: x = deltaA * x + deltaB_u
thread_data[i] = make_float2(deltaA_val * running_prefix.x + deltaB_u_val, 
                             deltaA_val * running_prefix.y + ...);
```

#### New (Full Matrix):
```cpp
if (params.is_full_A_matrix) {
    // Matrix-vector multiplication: x = deltaA @ x + deltaB_u
    // deltaA is (dstate, dstate), x is (dstate,), result is (dstate,)
    // Use shared memory for matrix-vector product
    float x_new[dstate];
    for (int j = 0; j < dstate; j++) {
        x_new[j] = 0.0f;
        for (int k = 0; k < dstate; k++) {
            x_new[j] += deltaA[j * dstate + k] * x[k];
        }
        x_new[j] += deltaB_u[j];
    }
    // Update x with x_new
} else {
    // Original diagonal code
}
```

### 5. Memory Layout Considerations

For full A matrices, memory layout options:

**Option 1: Contiguous per-channel**
```
A[d, :, :] is contiguous for each channel d
Stride: A_d_stride = dstate * dstate
```

**Option 2: Interleaved**
```
All A[d, i, j] for fixed (i, j) are contiguous
Stride: A_d_stride = d_inner
```

**Recommendation**: Use Option 1 (contiguous per-channel) for better cache locality.

### 6. Performance Optimizations

#### Block-Diagonal Structure
When A is block-diagonal + low-rank, optimize operations:

```cpp
if (params.is_block_diagonal_A) {
    // Exploit block structure
    // For block-diagonal: only compute within blocks
    // For low-rank: use Sherman-Morrison formula or direct computation
    // A = blockdiag(A₁, ..., Aₖ) + UVᵀ
    // exp(A) ≈ exp(blockdiag(...)) + correction from UVᵀ
}
```

#### Shared Memory Usage
- Store A matrix in shared memory when d_state is small (≤ 16)
- Use registers for very small blocks (4×4 or 8×8)
- Consider texture memory for larger matrices

### 7. Backward Pass Updates

**File**: `csrc/selective_scan/selective_scan_bwd_*.cu`

Similar changes needed for backward pass:
- Handle full A matrices in gradient computation
- Matrix-vector products instead of element-wise operations
- Chain rule through matrix exponential

### 8. Testing Strategy

1. **Unit Tests**: Test with small matrices (4×4, 8×8) first
2. **Numerical Stability**: Compare with Python reference implementation
3. **Performance Benchmarks**: Compare diagonal vs full A performance
4. **Gradient Checks**: Verify backward pass correctness

### 9. Implementation Priority

1. **High Priority**: ZOH discretization with full A matrices
2. **Medium Priority**: Bilinear transform (already matrix-based)
3. **Low Priority**: FOH, RK4, polynomial (less commonly used)

### 10. Example Kernel Skeleton

```cpp
template <typename Ktraits>
__global__ void selective_scan_fwd_kernel(SSMParamsBase params) {
    // ... existing setup ...
    
    if (params.is_full_A_matrix) {
        // Load full A matrix for this channel
        weight_t A_matrix[MAX_DSTATE * MAX_DSTATE];
        // Load A[dim_id] into shared memory or registers
        
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            // Compute matrix exponential: exp(delta * A)
            // Use iterative method or cuSOLVER
            
            // Matrix-vector product for state update
            // x_new = deltaA @ x_old + deltaB_u
        }
    } else {
        // Original diagonal A code
    }
}
```

## Migration Path

1. **Phase 1**: Add detection and Python fallback (✅ Done)
2. **Phase 2**: Implement ZOH with full A in CUDA
3. **Phase 3**: Optimize with block-diagonal structure
4. **Phase 4**: Add other discretization methods
5. **Phase 5**: Performance tuning and optimization

## References

- Matrix Exponential: [scipy.linalg.expm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)
- CUDA cuSOLVER: [cuSOLVER API Reference](https://docs.nvidia.com/cuda/cusolver/index.html)
- Block-Diagonal Matrices: [Efficient Operations on Block-Diagonal Matrices](https://en.wikipedia.org/wiki/Block_matrix)

