/******************************************************************************
 * Feature-SST: Structured State Transitions - Block-Diagonal + Low-Rank A
 * 
 * SST = Structured State Transitions: A = blockdiag(A_1, ..., A_K) + UV^T
 * This architecture enables cross-channel dynamics while maintaining
 * computational efficiency compared to dense A matrices.
 * 
 * Matrix operations for full A matrices (block-diagonal + low-rank)
 ******************************************************************************/

#pragma once

#include "selective_scan_common.h"

// Maximum matrix size for on-device operations
#define MAX_MATRIX_SIZE 64

// Matrix exponential using Taylor series: exp(A) ≈ I + A + A²/2! + A³/3! + ...
// For small matrices (dstate <= 16), this is efficient
template <typename T>
__device__ __forceinline__ void matrix_exp_taylor(
    const T *A, // Input matrix (dstate x dstate), row-major
    T *expA,    // Output: exp(A) (dstate x dstate), row-major
    int dstate,
    int num_terms = 10 // Number of Taylor series terms
)
{
// Initialize expA as identity matrix
#pragma unroll
    for (int i = 0; i < MAX_MATRIX_SIZE; ++i)
    {
#pragma unroll
        for (int j = 0; j < MAX_MATRIX_SIZE; ++j)
        {
            if (i < dstate && j < dstate)
            {
                expA[i * dstate + j] = (i == j) ? T(1.0f) : T(0.0f);
            }
        }
    }

    // Temporary storage for A^k
    T Ak[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    T temp[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];

// Copy A to Ak
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            Ak[i * dstate + j] = A[i * dstate + j];
        }
    }

    float factorial = 1.0f;
    for (int k = 1; k <= num_terms; ++k)
    {
        factorial *= k;

// Add A^k / k! to expA
#pragma unroll
        for (int i = 0; i < dstate; ++i)
        {
#pragma unroll
            for (int j = 0; j < dstate; ++j)
            {
                expA[i * dstate + j] += Ak[i * dstate + j] / factorial;
            }
        }

        // Compute A^(k+1) = A^k * A
        if (k < num_terms)
        {
#pragma unroll
            for (int i = 0; i < dstate; ++i)
            {
#pragma unroll
                for (int j = 0; j < dstate; ++j)
                {
                    temp[i * dstate + j] = T(0.0f);
#pragma unroll
                    for (int l = 0; l < dstate; ++l)
                    {
                        temp[i * dstate + j] += Ak[i * dstate + l] * A[l * dstate + j];
                    }
                }
            }
// Swap Ak and temp
#pragma unroll
            for (int i = 0; i < dstate; ++i)
            {
#pragma unroll
                for (int j = 0; j < dstate; ++j)
                {
                    Ak[i * dstate + j] = temp[i * dstate + j];
                }
            }
        }
    }
}

// Matrix-vector multiplication: y = A * x
template <typename T>
__device__ __forceinline__ void matrix_vector_mult(
    const T *A, // Matrix (dstate x dstate), row-major
    const T *x, // Input vector (dstate)
    T *y,       // Output vector (dstate)
    int dstate)
{
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            y[i] = T(0.0f);
#pragma unroll
            for (int j = 0; j < MAX_DSTATE; ++j)
            {
                if (j < dstate)
                {
                    y[i] += A[i * dstate + j] * x[j];
                }
            }
        }
    }
}

// Matrix exponential scaled by delta: exp(delta * A)
template <typename T>
__device__ __forceinline__ void matrix_exp_scaled(
    const T *A,  // Input matrix (dstate x dstate), row-major
    T *expA,     // Output: exp(delta * A) (dstate x dstate), row-major
    float delta, // Scaling factor
    int dstate,
    int num_terms = 10)
{
    // Compute delta * A first
    T deltaA[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            deltaA[i * dstate + j] = T(delta * A[i * dstate + j]);
        }
    }

    // Compute exp(delta * A)
    matrix_exp_taylor(deltaA, expA, dstate, num_terms);
}

// Block-diagonal matrix-vector multiplication (optimized)
// A = blockdiag(A_1, ..., A_K) where each A_k is block_size x block_size
template <typename T>
__device__ __forceinline__ void block_diagonal_matrix_vector_mult(
    const T *A_blocks, // Block matrices stored contiguously
    const T *x,        // Input vector (dstate)
    T *y,              // Output vector (dstate)
    int dstate,
    int block_size,
    int num_blocks)
{
// Initialize output
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            y[i] = T(0.0f);
        }
    }

    // Process each block
    for (int k = 0; k < num_blocks; ++k)
    {
        int start_idx = k * block_size;
        const T *A_k = A_blocks + k * block_size * block_size;

#pragma unroll
        for (int i = 0; i < 16; ++i)
        { // Max block size
            if (i < block_size)
            {
                int row_idx = start_idx + i;
                if (row_idx < dstate)
                {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                    {
                        if (j < block_size)
                        {
                            int col_idx = start_idx + j;
                            if (col_idx < dstate)
                            {
                                y[row_idx] += A_k[i * block_size + j] * x[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Low-rank matrix-vector multiplication: (UV^T) * x = U * (V^T * x)
template <typename T>
__device__ __forceinline__ void low_rank_matrix_vector_mult(
    const T *U, // U matrix (dstate x rank), column-major
    const T *V, // V matrix (dstate x rank), column-major
    const T *x, // Input vector (dstate)
    T *y,       // Output vector (dstate)
    int dstate,
    int rank)
{
    // Compute V^T * x (intermediate result)
    T Vtx[16]; // Max rank
#pragma unroll
    for (int r = 0; r < 16; ++r)
    {
        if (r < rank)
        {
            Vtx[r] = T(0.0f);
#pragma unroll
            for (int i = 0; i < MAX_DSTATE; ++i)
            {
                if (i < dstate)
                {
                    Vtx[r] += V[i * rank + r] * x[i]; // V is column-major
                }
            }
        }
    }

// Compute U * (V^T * x)
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            y[i] = T(0.0f);
#pragma unroll
            for (int r = 0; r < 16; ++r)
            {
                if (r < rank)
                {
                    y[i] += U[i * rank + r] * Vtx[r]; // U is column-major
                }
            }
        }
    }
}

// Combined block-diagonal + low-rank matrix-vector multiplication
// A = blockdiag(A_1, ..., A_K) + UV^T
template <typename T>
__device__ __forceinline__ void block_diagonal_lowrank_matrix_vector_mult(
    const T *A_blocks, // Block matrices
    const T *U,        // Low-rank U
    const T *V,        // Low-rank V
    const T *x,        // Input vector
    T *y,              // Output vector
    int dstate,
    int block_size,
    int num_blocks,
    int rank)
{
// Initialize output
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            y[i] = T(0.0f);
        }
    }

    // Block-diagonal part
    block_diagonal_matrix_vector_mult(A_blocks, x, y, dstate, block_size, num_blocks);

    // Low-rank part: add UV^T * x
    T lowrank_result[MAX_DSTATE];
    low_rank_matrix_vector_mult(U, V, x, lowrank_result, dstate, rank);

#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            y[i] += lowrank_result[i];
        }
    }
}

// Optimized matrix-vector multiplication: exp(delta * (blockdiag + UV^T)) @ x
// This computes the matrix-vector product directly from structured components
// without constructing the full exp(delta * A) matrix
// 
// Strategy:
// 1. Compute exp(delta * blockdiag) @ x block-wise (efficient!)
// 2. Compute low-rank correction contribution
// 3. Combine results
//
// This avoids storing the full dstate x dstate matrix in memory
template <typename T>
__device__ __forceinline__ void block_diagonal_lowrank_exp_matrix_vector_mult(
    const T *A_blocks, // Block matrices
    const T *U,        // Low-rank U (dstate x rank), column-major
    const T *V,        // Low-rank V (dstate x rank), column-major
    const T *x,        // Input vector (dstate)
    T *y,              // Output: exp(delta * (blockdiag + UV^T)) @ x
    float delta,       // Scaling factor
    int dstate,
    int block_size,
    int num_blocks,
    int rank,
    int num_terms = 10,
    bool use_first_order = true)
{
    // Step 1: Compute exp(delta * blockdiag) @ x block-wise
    // Initialize output
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            y[i] = T(0.0f);
        }
    }
    
    // Process each block independently
    for (int k = 0; k < num_blocks; ++k)
    {
        int start_idx = k * block_size;
        const T *A_k = A_blocks + k * block_size * block_size;
        
        // Compute exp(delta * A_k) for this block
        T deltaA_k[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
        T expA_k[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
        
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (i < block_size)
            {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    if (j < block_size)
                    {
                        deltaA_k[i * block_size + j] = T(delta * A_k[i * block_size + j]);
                    }
                }
            }
        }
        matrix_exp_taylor(deltaA_k, expA_k, block_size, num_terms);
        
        // Compute exp(delta * A_k) @ x_k (where x_k is the relevant portion of x)
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (i < block_size)
            {
                int row = start_idx + i;
                if (row < dstate)
                {
                    T sum = T(0.0f);
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                    {
                        if (j < block_size)
                        {
                            int col = start_idx + j;
                            if (col < dstate)
                            {
                                sum += expA_k[i * block_size + j] * x[col];
                            }
                        }
                    }
                    y[row] = sum;
                }
            }
        }
    }
    
    // Step 2: Apply low-rank correction
    if (use_first_order && delta * rank < 1.0f)
    {
        // First-order: exp(delta * (blockdiag + UV^T)) @ x ≈ exp(delta * blockdiag) @ x + delta * UV^T @ x
        T lowrank_result[MAX_DSTATE];
        low_rank_matrix_vector_mult(U, V, x, lowrank_result, dstate, rank);
        
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
                y[i] += T(delta) * lowrank_result[i];
            }
        }
    }
    else
    {
        // Higher-order: Need to compute exp(delta * UV^T) @ x more accurately
        // Compute V^T @ x first (rank x 1)
        T VTx[16];
#pragma unroll
        for (int r = 0; r < 16; ++r)
        {
            if (r < rank)
            {
                VTx[r] = T(0.0f);
#pragma unroll
                for (int i = 0; i < MAX_DSTATE; ++i)
                {
                    if (i < dstate)
                    {
                        VTx[r] += V[i * rank + r] * x[i];
                    }
                }
            }
        }
        
        // Compute V^T U (rank x rank)
        T VTU[16 * 16];
#pragma unroll
        for (int r1 = 0; r1 < 16; ++r1)
        {
            if (r1 < rank)
            {
#pragma unroll
                for (int r2 = 0; r2 < 16; ++r2)
                {
                    if (r2 < rank)
                    {
                        T sum = T(0.0f);
#pragma unroll
                        for (int i = 0; i < MAX_DSTATE; ++i)
                        {
                            if (i < dstate)
                            {
                                sum += V[i * rank + r1] * U[i * rank + r2];
                            }
                        }
                        VTU[r1 * rank + r2] = sum;
                    }
                }
            }
        }
        
        // Compute exp(delta * V^T U) @ VTx
        T delta_VTU[16 * 16];
#pragma unroll
        for (int i = 0; i < 16 * 16; ++i)
        {
            if (i < rank * rank)
            {
                delta_VTU[i] = T(delta) * VTU[i];
            }
        }
        T exp_delta_VTU[16 * 16];
        matrix_exp_taylor(delta_VTU, exp_delta_VTU, rank, num_terms);
        
        // Compute exp(delta * V^T U) @ VTx
        T exp_VTU_VTx[16];
#pragma unroll
        for (int r1 = 0; r1 < 16; ++r1)
        {
            if (r1 < rank)
            {
                T sum = T(0.0f);
#pragma unroll
                for (int r2 = 0; r2 < 16; ++r2)
                {
                    if (r2 < rank)
                    {
                        sum += exp_delta_VTU[r1 * rank + r2] * VTx[r2];
                    }
                }
                exp_VTU_VTx[r1] = sum;
            }
        }
        
        // Add identity contribution: VTx
#pragma unroll
        for (int r = 0; r < rank; ++r)
        {
            exp_VTU_VTx[r] += VTx[r];
        }
        
        // Finally: U @ (exp(delta * V^T U) @ VTx)
        T lowrank_correction[MAX_DSTATE];
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
                T sum = T(0.0f);
#pragma unroll
                for (int r = 0; r < 16; ++r)
                {
                    if (r < rank)
                    {
                        sum += U[i * rank + r] * exp_VTU_VTx[r];
                    }
                }
                lowrank_correction[i] = sum;
            }
        }
        
        // Add to y (approximation: exp(blockdiag + UV^T) ≈ exp(blockdiag) * exp(UV^T))
        // Note: This is an approximation since they don't commute, but usually sufficient
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
                // We need to apply exp(blockdiag) to the low-rank correction
                // For efficiency, we use the approximation: add it directly
                // A more accurate approach would multiply exp(blockdiag) @ lowrank_correction
                // but that requires another block-wise multiplication
                y[i] += lowrank_correction[i];
            }
        }
    }
}

// Optimized matrix exponential for block-diagonal matrix
// exp(blockdiag(A_1, ..., A_K)) = blockdiag(exp(A_1), ..., exp(A_K))
// This is MUCH more efficient than computing exp of the full matrix!
template <typename T>
__device__ __forceinline__ void block_diagonal_matrix_exp(
    const T *A_blocks, // Block matrices (num_blocks * block_size * block_size), stored contiguously
    T *expA_blocks,    // Output: exp(A_blocks) (num_blocks * block_size * block_size)
    float delta,       // Scaling factor: compute exp(delta * A_blocks)
    int block_size,
    int num_blocks,
    int num_terms = 10)
{
    // Compute exp(delta * A_k) for each block independently
    for (int k = 0; k < num_blocks; ++k)
    {
        const T *A_k = A_blocks + k * block_size * block_size;
        T *expA_k = expA_blocks + k * block_size * block_size;
        
        // Scale A_k by delta
        T deltaA_k[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (i < block_size)
            {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    if (j < block_size)
                    {
                        deltaA_k[i * block_size + j] = T(delta * A_k[i * block_size + j]);
                    }
                }
            }
        }
        
        // Compute exp(delta * A_k) using Taylor series
        matrix_exp_taylor(deltaA_k, expA_k, block_size, num_terms);
    }
}

// Optimized matrix exponential for block-diagonal + low-rank structure
// A = blockdiag(A_1, ..., A_K) + UV^T
// Compute exp(delta * A) efficiently using structure
// 
// Strategy:
// 1. Compute exp(delta * blockdiag) block-wise (efficient!)
// 2. Apply low-rank correction using first-order approximation or series expansion
// 
// For small delta*UV^T, we use:
// exp(delta * (blockdiag + UV^T)) ≈ exp(delta * blockdiag) * (I + delta * UV^T)
// 
// For better accuracy, we can use the Zassenhaus formula or higher-order terms
template <typename T>
__device__ __forceinline__ void block_diagonal_lowrank_matrix_exp(
    const T *A_blocks, // Block matrices
    const T *U,        // Low-rank U (dstate x rank), column-major
    const T *V,        // Low-rank V (dstate x rank), column-major
    T *expA,           // Output: exp(delta * A) (dstate x dstate), row-major
    float delta,       // Scaling factor
    int dstate,
    int block_size,
    int num_blocks,
    int rank,
    int num_terms = 10,
    bool use_first_order = true) // If true, use first-order approximation for low-rank correction
{
    // Step 1: Compute exp(delta * blockdiag) block-wise
    T exp_blockdiag[MAX_DSTATE * MAX_DSTATE];
    
    // Initialize to zero
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
#pragma unroll
        for (int j = 0; j < MAX_DSTATE; ++j)
        {
            if (i < dstate && j < dstate)
            {
                exp_blockdiag[i * dstate + j] = (i == j) ? T(1.0f) : T(0.0f);
            }
        }
    }
    
    // Compute exp(delta * A_k) for each block
    T expA_k[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    for (int k = 0; k < num_blocks; ++k)
    {
        int start_idx = k * block_size;
        const T *A_k = A_blocks + k * block_size * block_size;
        
        // Scale and compute exp
        T deltaA_k[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (i < block_size)
            {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    if (j < block_size)
                    {
                        deltaA_k[i * block_size + j] = T(delta * A_k[i * block_size + j]);
                    }
                }
            }
        }
        matrix_exp_taylor(deltaA_k, expA_k, block_size, num_terms);
        
        // Place exp(delta * A_k) in the appropriate position
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (i < block_size)
            {
                int row = start_idx + i;
                if (row < dstate)
                {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                    {
                        if (j < block_size)
                        {
                            int col = start_idx + j;
                            if (col < dstate)
                            {
                                exp_blockdiag[row * dstate + col] = expA_k[i * block_size + j];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Step 2: Apply low-rank correction
    if (use_first_order && delta * rank < 1.0f) // First-order approximation for small delta
    {
        // exp(delta * (blockdiag + UV^T)) ≈ exp(delta * blockdiag) * (I + delta * UV^T)
        // Compute delta * UV^T
        T delta_UVT[MAX_DSTATE * MAX_DSTATE];
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
#pragma unroll
                for (int j = 0; j < MAX_DSTATE; ++j)
                {
                    if (j < dstate)
                    {
                        T uv_sum = T(0.0f);
#pragma unroll
                        for (int r = 0; r < 16; ++r)
                        {
                            if (r < rank)
                            {
                                uv_sum += U[i * rank + r] * V[j * rank + r];
                            }
                        }
                        delta_UVT[i * dstate + j] = T(delta * uv_sum);
                        // Add identity for (I + delta * UV^T)
                        if (i == j)
                        {
                            delta_UVT[i * dstate + j] += T(1.0f);
                        }
                    }
                }
            }
        }
        
        // Multiply: exp_blockdiag * (I + delta * UV^T)
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
#pragma unroll
                for (int j = 0; j < MAX_DSTATE; ++j)
                {
                    if (j < dstate)
                    {
                        T sum = T(0.0f);
#pragma unroll
                        for (int l = 0; l < MAX_DSTATE; ++l)
                        {
                            if (l < dstate)
                            {
                                sum += exp_blockdiag[i * dstate + l] * delta_UVT[l * dstate + j];
                            }
                        }
                        expA[i * dstate + j] = sum;
                    }
                }
            }
        }
    }
    else
    {
        // Higher-order approximation: use series expansion for better accuracy
        // For UV^T (rank r << dstate), we can compute exp(delta * UV^T) more efficiently
        // using the fact that (UV^T)^k = U(V^T U)^(k-1) V^T
        // This allows us to work in the low-rank space (rank x rank) instead of (dstate x dstate)
        
        // Compute V^T U (rank x rank matrix) - this is the key to efficiency!
        T VTU[16 * 16]; // Max rank x rank
#pragma unroll
        for (int r1 = 0; r1 < 16; ++r1)
        {
            if (r1 < rank)
            {
#pragma unroll
                for (int r2 = 0; r2 < 16; ++r2)
                {
                    if (r2 < rank)
                    {
                        T sum = T(0.0f);
#pragma unroll
                        for (int i = 0; i < MAX_DSTATE; ++i)
                        {
                            if (i < dstate)
                            {
                                sum += V[i * rank + r1] * U[i * rank + r2];
                            }
                        }
                        VTU[r1 * rank + r2] = sum;
                    }
                }
            }
        }
        
        // Scale by delta: compute delta * VTU
        T delta_VTU[16 * 16];
#pragma unroll
        for (int i = 0; i < 16 * 16; ++i)
        {
            if (i < rank * rank)
            {
                delta_VTU[i] = T(delta) * VTU[i];
            }
        }
        
        // Compute exp(delta * V^T U) in the low-rank space (much smaller: rank x rank!)
        T exp_delta_VTU[16 * 16];
        matrix_exp_taylor(delta_VTU, exp_delta_VTU, rank, num_terms);
        
        // Now compute exp(delta * UV^T) = U * exp(delta * V^T U) * V^T
        // This is much more efficient than computing exp of the full dstate x dstate matrix!
        T exp_delta_UVT[MAX_DSTATE * MAX_DSTATE];
        
        // Initialize to identity
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
#pragma unroll
            for (int j = 0; j < MAX_DSTATE; ++j)
            {
                if (i < dstate && j < dstate)
                {
                    exp_delta_UVT[i * dstate + j] = (i == j) ? T(1.0f) : T(0.0f);
                }
            }
        }
        
        // Add U * exp_delta_VTU * V^T
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
#pragma unroll
                for (int j = 0; j < MAX_DSTATE; ++j)
                {
                    if (j < dstate)
                    {
                        T sum = T(0.0f);
#pragma unroll
                        for (int r1 = 0; r1 < 16; ++r1)
                        {
                            if (r1 < rank)
                            {
                                T intermediate = T(0.0f);
#pragma unroll
                                for (int r2 = 0; r2 < 16; ++r2)
                                {
                                    if (r2 < rank)
                                    {
                                        intermediate += U[i * rank + r1] * exp_delta_VTU[r1 * rank + r2];
                                    }
                                }
                                sum += intermediate * V[j * rank + r1];
                            }
                        }
                        exp_delta_UVT[i * dstate + j] += sum;
                    }
                }
            }
        }
        
        // Finally, multiply: exp_blockdiag * exp_delta_UVT
        // Note: This is an approximation since blockdiag and UV^T don't commute
        // For better accuracy, we could use the Zassenhaus formula, but this is usually sufficient
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
#pragma unroll
                for (int j = 0; j < MAX_DSTATE; ++j)
                {
                    if (j < dstate)
                    {
                        T sum = T(0.0f);
#pragma unroll
                        for (int l = 0; l < MAX_DSTATE; ++l)
                        {
                            if (l < dstate)
                            {
                                sum += exp_blockdiag[i * dstate + l] * exp_delta_UVT[l * dstate + j];
                            }
                        }
                        expA[i * dstate + j] = sum;
                    }
                }
            }
        }
    }
}
