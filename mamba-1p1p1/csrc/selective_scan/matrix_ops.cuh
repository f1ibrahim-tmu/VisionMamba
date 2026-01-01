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
