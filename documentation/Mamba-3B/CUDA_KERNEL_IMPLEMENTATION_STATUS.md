# CUDA Kernel Implementation Status for Full A Matrices (Mamba-3B)

## Current Status

### ✅ Completed

1. **Header Structure Updates**

   - Added `is_full_A_matrix`, `block_size`, `low_rank_rank` flags to `SSMParamsBase`
   - Added `A_matrix_stride` for full A matrix access
   - Created `matrix_ops.cuh` with helper functions for matrix operations

2. **Python Interface**

   - Detection of full A matrices (3D vs 2D)
   - Automatic fallback to Python reference when full A detected
   - Full support in Python reference implementation

3. **C++ Interface Updates**

   - Detection of full A matrices in `selective_scan.cpp`
   - Parameter structure in place
   - Error handling for unsupported full A matrices

4. **Kernel Structure**
   - Added conditional path for full A matrices in forward kernel
   - Matrix operation helpers created

### ✅ Partially Implemented

1. **Forward Kernel (`selective_scan_fwd_kernel.cuh`)**
   - ✅ Full A matrix path structure complete
   - ✅ Matrix loading into shared memory
   - ✅ Matrix exponential computation (Taylor series)
   - ✅ Matrix-vector multiplication in scan loop
   - ✅ Sequential time-step processing with full state vector
   - ✅ ZOH discretization fully implemented
   - ⚠️ Only ZOH discretization supported (other methods fall back to Python)

### ❌ Not Yet Implemented

1. **Other Discretization Methods for Full A**
   - FOH (First Order Hold) - needs matrix exponential with different formula
   - Bilinear Transform - needs matrix inversion and multiplication
   - RK4 (Runge-Kutta 4) - needs multiple matrix operations per step
   - Polynomial Interpolation - needs bidirectional scan with full A
   - High-order methods - needs extended matrix operations
2. **Block-Diagonal + Low-Rank Optimization**

   - Currently constructs full matrix (loses efficiency)
   - Need specialized kernels for block-diagonal operations
   - Need low-rank matrix-vector multiplication optimization
   - Need to exploit structure: `(blockdiag + UV^T) @ x = blockdiag @ x + U @ (V^T @ x)`

3. **Backward Kernel**

   - No updates for full A matrices
   - Gradient computation through matrix exponential needed

4. **Discretization Methods**
   - Only ZOH structure started
   - FOH, Bilinear, RK4, etc. need full A matrix support

## Implementation Challenges

### 1. Architecture Mismatch

**Problem**: Current kernel processes one state dimension at a time (`state_idx` loop), but full A matrices require processing all dimensions together.

**Solution Options**:

- **Option A**: Restructure kernel to process entire state vector per thread block
- **Option B**: Use separate kernel launch for full A matrices
- **Option C**: Hybrid approach - process in blocks of state dimensions

### 2. Matrix Exponential

**Problem**: Computing `exp(delta * A)` for each time step is expensive.

**Solution Options**:

- Use Taylor series (already implemented in `matrix_ops.cuh`)
- Use Padé approximation (more accurate, similar cost)
- Pre-compute for small matrices, use cuSOLVER for large ones
- Exploit block-diagonal structure to compute block-wise

### 3. Memory Access Patterns

**Problem**: Full A matrices are large (dstate² per channel).

**Solution**:

- Use shared memory for small matrices (dstate ≤ 16)
- Use texture memory for larger matrices
- Exploit block-diagonal structure to reduce memory access

### 4. Block-Diagonal + Low-Rank Optimization

**Problem**: Current implementation constructs full matrix, losing efficiency.

**Solution**:

- Implement specialized kernels for block-diagonal operations
- Use Sherman-Morrison formula for low-rank updates
- Combine operations: `(blockdiag + UV^T) @ x = blockdiag @ x + U @ (V^T @ x)`

## Next Steps

### Phase 1: Basic Full A Matrix Support (ZOH only) ✅ COMPLETE

1. ✅ Complete matrix loading in kernel
2. ✅ Implement matrix exponential using Taylor series
3. ✅ Implement matrix-vector multiplication in scan loop
4. ✅ Test with small matrices (dstate ≤ 8)

### Phase 2: Other Discretization Methods (IN PROGRESS)

1. **FOH (First Order Hold)**: Extend matrix exponential to handle FOH formula
   - Formula: `B_d = A^(-2) * (exp(A*Δ) - I - A*Δ) * B`
   - Need matrix inversion and matrix-matrix operations
2. **Bilinear Transform**: Implement matrix inversion and multiplication
   - Formula: `(I - ΔA/2)⁻¹(I + ΔA/2)`
   - Already matrix-based, but needs full A support
3. **RK4**: Implement 4-stage Runge-Kutta with full A matrices
   - Requires multiple matrix-vector products per time step
4. **Polynomial/High-order**: Extend to bidirectional scan with full A

### Phase 3: Block-Diagonal + Low-Rank Optimization

1. Add specialized block-diagonal matrix-vector multiplication
2. Implement low-rank matrix-vector multiplication: `U @ (V^T @ x)`
3. Combine operations efficiently without constructing full matrix
4. Optimize memory access patterns for block structure

### Phase 4: Backward Pass

1. Implement gradient computation through matrix exponential
2. Handle full A matrices in backward kernel
3. Implement matrix-vector products for gradients
4. Handle block-diagonal + low-rank in backward
5. Test gradient correctness with numerical checks

### Phase 5: Performance Optimization

1. Optimize shared memory usage
2. Add texture memory support for larger matrices
3. Tune matrix exponential computation (Padé approximation vs Taylor)
4. Profile and optimize hot paths
5. Compare performance vs Python reference

## Testing Strategy

1. **Unit Tests**: Small matrices (4×4, 8×8) first
2. **Numerical Tests**: Compare with Python reference
3. **Performance Tests**: Compare diagonal vs full A
4. **Gradient Tests**: Verify backward pass

## Current Status

**ZOH Discretization**: ✅ Fully implemented in CUDA kernels

- Full A matrices work with ZOH discretization
- Uses CUDA kernels (fast path)
- Matrix exponential via Taylor series
- Sequential time-step processing with full state vector

**Other Discretization Methods**: ⚠️ Fall back to Python reference

- FOH, Bilinear, RK4, Poly, High-order automatically use Python path
- Python reference works correctly but is slower
- CUDA implementation needed for performance

## Files Modified

1. `csrc/selective_scan/selective_scan.h` - Added flags to SSMParamsBase
2. `csrc/selective_scan/selective_scan.cpp` - Detection and parameter setting
3. `csrc/selective_scan/selective_scan_fwd_kernel.cuh` - Structure for full A path
4. `csrc/selective_scan/matrix_ops.cuh` - Helper functions (NEW)
5. `mamba_ssm/ops/selective_scan_interface.py` - Python detection and fallback
6. `mamba_ssm/modules/mamba_simple.py` - Full A matrix construction

## Estimated Implementation Time

- **Phase 1** (Basic ZOH): ✅ COMPLETE
- **Phase 2** (Other discretization methods): 5-7 days
  - FOH: 1-2 days
  - Bilinear: 1 day
  - RK4: 1-2 days
  - Poly/High-order: 1-2 days
- **Phase 3** (Block-diagonal + Low-rank optimization): 3-4 days
- **Phase 4** (Backward pass): 3-4 days
- **Phase 5** (Performance optimization): 2-3 days

**Remaining**: ~2-3 weeks for full implementation

## Notes

- The structure is in place for easy extension
- Matrix operation helpers are ready to use
- Python fallback ensures functionality while CUDA kernels are being developed
- Block-diagonal + low-rank structure can provide significant speedup when fully optimized
