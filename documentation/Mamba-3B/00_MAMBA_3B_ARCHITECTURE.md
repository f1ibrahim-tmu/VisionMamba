# Mamba-3B Architecture: Block-Diagonal + Low-Rank A Matrix

## Overview

Mamba-3B introduces architectural improvements to the original Vision Mamba architecture. The first major improvement is updating the A matrix parameterization from a simple diagonal structure to a block-diagonal + low-rank structure.

## 3.1 Structured State Transitions: Block-Diagonal + Low-Rank A

### 3.1.1 Motivation

The original Mamba architecture uses a diagonal A matrix, which:
- **Limits cross-channel dynamics**: Diagonal matrices only allow independent evolution of each state dimension
- **Restricts expressiveness**: Cannot model complex interactions between state dimensions

A dense A matrix would be more expressive but:
- **Computationally prohibitive**: O(d_state²) operations per channel
- **Memory intensive**: O(d_state²) parameters per channel

The block-diagonal + low-rank structure provides a middle ground:
- **Enables cross-channel dynamics**: Blocks allow interactions within groups
- **Maintains efficiency**: Much cheaper than dense matrices
- **Flexible expressiveness**: Low-rank component captures global patterns

### 3.1.2 Parameterization

The new A matrix is defined as:

```
A = blockdiag(A₁, ..., Aₖ) + UVᵀ
```

Where:
- **Aₖ ∈ R^(dₖ × dₖ)**: Small block matrices (e.g., 4×4, 8×8) placed on the diagonal
- **U, V ∈ R^(N × r)**: Low-rank factors with rank r << N
- **N = d_state**: The state dimension

### 3.1.3 Computational Cost

#### Block-Diagonal Construction
- **Complexity**: O(K × block_size² × d_inner)
- Where K = d_state / block_size (number of blocks)
- Example: d_state=16, block_size=4 → K=4 blocks

#### Low-Rank Component (UVᵀ)
- **Complexity**: O(d_inner × d_state × r)
- Where r << d_state (typically r = 2-4)
- Example: d_state=16, r=2 → 32 operations per channel

#### Total Cost
- **Per channel**: O(K × block_size² + d_state × r)
- **Total**: O(d_inner × (K × block_size² + d_state × r))
- **Comparison to dense**: O(d_inner × d_state²)
- **Speedup**: Approximately (d_state²) / (K × block_size² + d_state × r)

**Example Calculation** (d_state=16, block_size=4, r=2, d_inner=768):
- Block-diagonal: 4 × 16 = 64 operations per channel
- Low-rank: 16 × 2 = 32 operations per channel
- Total: 96 operations per channel
- Dense would be: 16² = 256 operations per channel
- **Speedup: ~2.67×**

### 3.1.4 Memory Footprint

#### Block-Diagonal Parameters
- **Storage**: K × block_size² × d_inner parameters
- Example: 4 × 16 × 768 = 49,152 parameters

#### Low-Rank Parameters
- **Storage**: 2 × d_state × r × d_inner parameters (U and V)
- Example: 2 × 16 × 2 × 768 = 49,152 parameters

#### Total Memory
- **Total**: d_inner × (K × block_size² + 2 × d_state × r)
- **Comparison to dense**: d_inner × d_state²
- **Reduction**: Approximately 50% when r << d_state

**Example Calculation** (d_state=16, block_size=4, r=2, d_inner=768):
- Block-diagonal: 49,152 parameters
- Low-rank: 49,152 parameters
- **Total: 98,304 parameters**
- Dense would be: 196,608 parameters
- **Memory reduction: 50%**

### 3.1.5 Integration with Existing Mamba Kernels

#### Current Implementation
- The new A matrix structure is implemented in `mamba_simple.py`
- For backward compatibility, the full matrix is constructed and then the diagonal is extracted
- This allows the code to work with existing CUDA kernels without modification

#### Kernel Updates Needed
To fully leverage the block-diagonal + low-rank structure, the following kernels need updates:

1. **selective_scan_fn**: Update to handle full A matrices instead of just diagonals
2. **mamba_inner_fn**: Modify to use block-diagonal + low-rank operations
3. **CUDA kernels**: Optimize matrix-vector products for the new structure

#### Benefits of Full Integration
- **Better performance**: Direct operations on block-diagonal + low-rank structure
- **Preserved expressiveness**: Full matrix information retained (not just diagonal)
- **Optimized memory access**: Block structure enables better cache utilization

### 3.1.6 Usage

To use the new A matrix structure, initialize Mamba with:

```python
mamba = Mamba(
    d_model=768,
    d_state=16,
    use_block_diagonal_lowrank=True,  # Enable new structure
    block_size=4,                      # Size of each block (4×4)
    low_rank_rank=2,                   # Rank of low-rank component
)
```

To use the original diagonal structure:

```python
mamba = Mamba(
    d_model=768,
    d_state=16,
    use_block_diagonal_lowrank=False,  # Use original diagonal A
)
```

### 3.1.7 Implementation Details

#### Initialization
- Block matrices are initialized with negative diagonal values (similar to original S4D)
- Low-rank factors U and V are initialized with small random values (std=0.01)
- All parameters are marked for no weight decay (consistent with original)

#### Matrix Construction
The `_construct_A_matrix()` method:
1. Constructs block-diagonal matrix by placing blocks on the diagonal
2. Computes low-rank component UVᵀ using batch matrix multiplication
3. Adds components together: A = blockdiag(A₁, ..., Aₖ) + UVᵀ
4. Extracts diagonal for backward compatibility (TODO: update kernels)

#### Bidirectional Support
- Both forward and backward (bidirectional) A matrices support the new structure
- Separate parameter sets for A and A_b when using bidirectional Mamba

## Future Improvements

1. **Kernel Optimization**: Update CUDA kernels to directly use block-diagonal + low-rank structure
2. **Adaptive Block Sizes**: Allow different block sizes for different layers
3. **Rank Adaptation**: Dynamically adjust low-rank rank based on layer depth
4. **Sparse Patterns**: Explore structured sparsity patterns within blocks

## References

- Original Mamba paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- S4D: [Diagonal State Spaces are as Effective as Structured State Spaces](https://arxiv.org/abs/2203.14343)

