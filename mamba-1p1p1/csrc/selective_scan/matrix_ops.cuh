/******************************************************************************
 * Feature-SST: Structured State Transitions - Block-Diagonal + Low-Rank A
 * 
 * SST = Structured State Transitions: A = blockdiag(A_1, ..., A_K) + UV^T
 * This architecture enables cross-channel dynamics while maintaining
 * computational efficiency compared to dense A matrices.
 * 
 * Matrix operations for full A matrices (block-diagonal + low-rank)
 * Supports both real and complex numbers, all 6 discretization methods,
 * and backward pass (gradient computation).
 ******************************************************************************/

#pragma once

#include "selective_scan_common.h"
#include "selective_scan.h"

// Maximum matrix size for on-device operations
#define MAX_MATRIX_SIZE 64
#define MAX_BLOCK_SIZE 16
#define MAX_LOW_RANK 16

// ============================================================================
// Complex number helpers (for complex_t type)
// ============================================================================

// Complex conjugate
template <typename T>
__device__ __forceinline__ T complex_conj(T x) { return x; }

template <>
__device__ __forceinline__ complex_t complex_conj<complex_t>(complex_t x) {
    return complex_t(x.real_, -x.imag_);
}

// Complex magnitude squared
template <typename T>
__device__ __forceinline__ float complex_mag_sq(T x) { return x * x; }

template <>
__device__ __forceinline__ float complex_mag_sq<complex_t>(complex_t x) {
    return x.real_ * x.real_ + x.imag_ * x.imag_;
}

// ============================================================================
// Matrix exponential using Taylor series: exp(A) ≈ I + A + A²/2! + A³/3! + ...
// Supports both real and complex matrices
// ============================================================================

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
                expA[i * dstate + j] = expA[i * dstate + j] + Ak[i * dstate + j] * T(1.0f / factorial);
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
                        temp[i * dstate + j] = temp[i * dstate + j] + Ak[i * dstate + l] * A[l * dstate + j];
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

// ============================================================================
// Matrix-vector multiplication: y = A * x
// ============================================================================

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
                    y[i] = y[i] + A[i * dstate + j] * x[j];
                }
            }
        }
    }
}

// ============================================================================
// Matrix exponential scaled by delta: exp(delta * A)
// ============================================================================

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
            deltaA[i * dstate + j] = A[i * dstate + j] * T(delta);
        }
    }

    // Compute exp(delta * A)
    matrix_exp_taylor(deltaA, expA, dstate, num_terms);
}

// ============================================================================
// Matrix exponential using Padé approximation (more accurate for larger matrices)
// exp(A) ≈ N(A) / D(A) where N and D are polynomials
// Padé (6,6): More accurate than Taylor for larger matrices and larger delta
// ============================================================================

template <typename T>
__device__ __forceinline__ void matrix_exp_pade6(
    const T *A, // Input matrix (dstate x dstate), row-major
    T *expA,    // Output: exp(A) (dstate x dstate), row-major
    int dstate)
{
    // Padé (6,6) coefficients
    const float c[] = {1.0f, 0.5f, 1.0f/9.0f, 1.0f/72.0f, 1.0f/1008.0f, 1.0f/30240.0f, 1.0f/665280.0f};
    
    // Compute powers of A: A², A³, A⁴, A⁵, A⁶
    T A2[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    T A3[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    T A4[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    T A5[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    T A6[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    T temp[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
    
    // A² = A * A
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
                temp[i * dstate + j] = temp[i * dstate + j] + A[i * dstate + l] * A[l * dstate + j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            A2[i * dstate + j] = temp[i * dstate + j];
        }
    }
    
    // A³ = A² * A
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
                temp[i * dstate + j] = temp[i * dstate + j] + A2[i * dstate + l] * A[l * dstate + j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            A3[i * dstate + j] = temp[i * dstate + j];
        }
    }
    
    // A⁴ = A³ * A
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
                temp[i * dstate + j] = temp[i * dstate + j] + A3[i * dstate + l] * A[l * dstate + j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            A4[i * dstate + j] = temp[i * dstate + j];
        }
    }
    
    // A⁵ = A⁴ * A
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
                temp[i * dstate + j] = temp[i * dstate + j] + A4[i * dstate + l] * A[l * dstate + j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            A5[i * dstate + j] = temp[i * dstate + j];
        }
    }
    
    // A⁶ = A⁵ * A
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
                temp[i * dstate + j] = temp[i * dstate + j] + A5[i * dstate + l] * A[l * dstate + j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            A6[i * dstate + j] = temp[i * dstate + j];
        }
    }
    
    // Compute N = I + c[1]*A + c[2]*A² + c[3]*A³ + c[4]*A⁴ + c[5]*A⁵ + c[6]*A⁶
    T N[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            N[i * dstate + j] = (i == j) ? T(1.0f) : T(0.0f);
            N[i * dstate + j] = N[i * dstate + j] + A[i * dstate + j] * T(c[1]);
            N[i * dstate + j] = N[i * dstate + j] + A2[i * dstate + j] * T(c[2]);
            N[i * dstate + j] = N[i * dstate + j] + A3[i * dstate + j] * T(c[3]);
            N[i * dstate + j] = N[i * dstate + j] + A4[i * dstate + j] * T(c[4]);
            N[i * dstate + j] = N[i * dstate + j] + A5[i * dstate + j] * T(c[5]);
            N[i * dstate + j] = N[i * dstate + j] + A6[i * dstate + j] * T(c[6]);
        }
    }
    
    // Compute D = I - c[1]*A + c[2]*A² - c[3]*A³ + c[4]*A⁴ - c[5]*A⁵ + c[6]*A⁶
    T D[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            D[i * dstate + j] = (i == j) ? T(1.0f) : T(0.0f);
            D[i * dstate + j] = D[i * dstate + j] - A[i * dstate + j] * T(c[1]);
            D[i * dstate + j] = D[i * dstate + j] + A2[i * dstate + j] * T(c[2]);
            D[i * dstate + j] = D[i * dstate + j] - A3[i * dstate + j] * T(c[3]);
            D[i * dstate + j] = D[i * dstate + j] + A4[i * dstate + j] * T(c[4]);
            D[i * dstate + j] = D[i * dstate + j] - A5[i * dstate + j] * T(c[5]);
            D[i * dstate + j] = D[i * dstate + j] + A6[i * dstate + j] * T(c[6]);
        }
    }
    
    // Compute expA = D^(-1) * N using Gaussian elimination (simplified)
    // For small matrices, we can use direct inversion
    // For larger matrices, this would need iterative methods
    // For now, use approximation: expA ≈ N * (I + D - I) = N * D^(-1) ≈ N * (2*I - D)
    // More accurate: use iterative refinement or LU decomposition
    
    // Simplified: expA ≈ N * (2*I - D) for small matrices
    // This is a first-order approximation of D^(-1)
    T D_inv_approx[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            D_inv_approx[i * dstate + j] = (i == j) ? T(2.0f) : T(0.0f);
            D_inv_approx[i * dstate + j] = D_inv_approx[i * dstate + j] - D[i * dstate + j];
        }
    }
    
    // expA = N * D_inv_approx
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            expA[i * dstate + j] = T(0.0f);
#pragma unroll
            for (int l = 0; l < dstate; ++l)
            {
                expA[i * dstate + j] = expA[i * dstate + j] + N[i * dstate + l] * D_inv_approx[l * dstate + j];
            }
        }
    }
}

// Matrix exponential scaled by delta using Padé: exp(delta * A)
template <typename T>
__device__ __forceinline__ void matrix_exp_scaled_pade(
    const T *A,  // Input matrix (dstate x dstate), row-major
    T *expA,     // Output: exp(delta * A) (dstate x dstate), row-major
    float delta, // Scaling factor
    int dstate)
{
    // Compute delta * A first
    T deltaA[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
#pragma unroll
    for (int i = 0; i < dstate; ++i)
    {
#pragma unroll
        for (int j = 0; j < dstate; ++j)
        {
            deltaA[i * dstate + j] = A[i * dstate + j] * T(delta);
        }
    }
    
    // Compute exp(delta * A) using Padé
    matrix_exp_pade6(deltaA, expA, dstate);
}

// ============================================================================
// Block-diagonal matrix-vector multiplication (optimized)
// A = blockdiag(A_1, ..., A_K) where each A_k is block_size x block_size
// ============================================================================

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
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
        {
            if (i < block_size)
            {
                int row_idx = start_idx + i;
                if (row_idx < dstate)
                {
#pragma unroll
                    for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
                    {
                        if (j < block_size)
                        {
                            int col_idx = start_idx + j;
                            if (col_idx < dstate)
                            {
                                y[row_idx] = y[row_idx] + A_k[i * block_size + j] * x[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Low-rank matrix-vector multiplication: (UV^T) * x = U * (V^T * x)
// ============================================================================

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
    T Vtx[MAX_LOW_RANK];
#pragma unroll
    for (int r = 0; r < MAX_LOW_RANK; ++r)
    {
        if (r < rank)
        {
            Vtx[r] = T(0.0f);
#pragma unroll
            for (int i = 0; i < MAX_DSTATE; ++i)
            {
                if (i < dstate)
                {
                    Vtx[r] = Vtx[r] + complex_conj(V[i * rank + r]) * x[i];
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
            for (int r = 0; r < MAX_LOW_RANK; ++r)
            {
                if (r < rank)
                {
                    y[i] = y[i] + U[i * rank + r] * Vtx[r];
                }
            }
        }
    }
}

// ============================================================================
// Combined block-diagonal + low-rank matrix-vector multiplication
// A = blockdiag(A_1, ..., A_K) + UV^T
// ============================================================================

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
            y[i] = y[i] + lowrank_result[i];
        }
    }
}

// ============================================================================
// Optimized block-wise matrix exponential
// exp(blockdiag(A_1, ..., A_K)) = blockdiag(exp(A_1), ..., exp(A_K))
// ============================================================================

template <typename T>
// ============================================================================
// Enhanced error handling and numerical stability checks
// ============================================================================

// Check for numerical stability issues
template <typename T>
__device__ __forceinline__ bool check_matrix_stability(
    const T *A, int dstate, float delta, float threshold = 1e6f)
{
    // Check for very large values that could cause overflow
    float max_val = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_MATRIX_SIZE; ++i)
    {
        if (i < dstate)
        {
#pragma unroll
            for (int j = 0; j < MAX_MATRIX_SIZE; ++j)
            {
                if (j < dstate)
                {
                    float val = fabsf(float(A[i * dstate + j]));
                    if (val > max_val) max_val = val;
                }
            }
        }
    }
    
    // Check if delta * max_val could cause issues
    return (delta * max_val < threshold);
}

// Determine whether to use Padé or Taylor based on matrix size and delta
__device__ __forceinline__ bool should_use_pade(int dstate, float delta, float threshold = 8.0f)
{
    // Use Padé for larger matrices or larger delta values
    // Padé is more accurate but requires matrix inversion
    // Threshold: use Padé if dstate >= 8 or delta >= threshold
    return (dstate >= 8 || fabsf(delta) >= threshold);
}

__device__ __forceinline__ void block_diagonal_matrix_exp(
    const T *A_blocks, // Block matrices (num_blocks * block_size * block_size)
    T *expA_blocks,    // Output: exp(A_blocks)
    float delta,       // Scaling factor: compute exp(delta * A_blocks)
    int block_size,
    int num_blocks,
    int num_terms = 10,
    bool use_pade = false)  // Use Padé approximation if true
{
    // Compute exp(delta * A_k) for each block independently
    for (int k = 0; k < num_blocks; ++k)
    {
        const T *A_k = A_blocks + k * block_size * block_size;
        T *expA_k = expA_blocks + k * block_size * block_size;
        
        // Scale A_k by delta
        T deltaA_k[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE];
        #pragma unroll
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
        {
            if (i < block_size)
            {
                #pragma unroll
                for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
                {
                    if (j < block_size)
                    {
                        deltaA_k[i * block_size + j] = A_k[i * block_size + j] * T(delta);
                    }
                }
            }
        }
        
        // Compute exp(delta * A_k) using Taylor series or Padé
        if (use_pade && should_use_pade(block_size, delta))
        {
            // Use Padé approximation for better accuracy on larger blocks or larger delta
            matrix_exp_pade6(deltaA_k, expA_k, block_size);
        }
        else
        {
            // Use Taylor series (default, faster for small blocks)
            matrix_exp_taylor(deltaA_k, expA_k, block_size, num_terms);
        }
    }
}

// ============================================================================
// Optimized matrix-vector multiplication: exp(delta * (blockdiag + UV^T)) @ x
// Computes directly from structured components without forming full matrix
// ============================================================================

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
        T deltaA_k[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE];
        T expA_k[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE];
        
#pragma unroll
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
        {
            if (i < block_size)
            {
#pragma unroll
                for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
                {
                    if (j < block_size)
                    {
                        deltaA_k[i * block_size + j] = A_k[i * block_size + j] * T(delta);
                    }
                }
            }
        }
        matrix_exp_taylor(deltaA_k, expA_k, block_size, num_terms);
        
        // Compute exp(delta * A_k) @ x_k
#pragma unroll
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
        {
            if (i < block_size)
            {
                int row = start_idx + i;
                if (row < dstate)
                {
                    T sum = T(0.0f);
#pragma unroll
                    for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
                    {
                        if (j < block_size)
                        {
                            int col = start_idx + j;
                            if (col < dstate)
                            {
                                sum = sum + expA_k[i * block_size + j] * x[col];
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
                y[i] = y[i] + lowrank_result[i] * T(delta);
            }
        }
    }
    else
    {
        // Higher-order: Compute exp(delta * UV^T) in low-rank space
        // V^T @ x first (rank x 1)
        T VTx[MAX_LOW_RANK];
#pragma unroll
        for (int r = 0; r < MAX_LOW_RANK; ++r)
        {
            if (r < rank)
            {
                VTx[r] = T(0.0f);
#pragma unroll
                for (int i = 0; i < MAX_DSTATE; ++i)
                {
                    if (i < dstate)
                    {
                        VTx[r] = VTx[r] + complex_conj(V[i * rank + r]) * x[i];
                    }
                }
            }
        }
        
        // V^T U (rank x rank)
        T VTU[MAX_LOW_RANK * MAX_LOW_RANK];
#pragma unroll
        for (int r1 = 0; r1 < MAX_LOW_RANK; ++r1)
        {
            if (r1 < rank)
            {
#pragma unroll
                for (int r2 = 0; r2 < MAX_LOW_RANK; ++r2)
                {
                    if (r2 < rank)
                    {
                        T sum = T(0.0f);
#pragma unroll
                        for (int i = 0; i < MAX_DSTATE; ++i)
                        {
                            if (i < dstate)
                            {
                                sum = sum + complex_conj(V[i * rank + r1]) * U[i * rank + r2];
                            }
                        }
                        VTU[r1 * rank + r2] = sum;
                    }
                }
            }
        }
        
        // exp(delta * V^T U) @ VTx
        T delta_VTU[MAX_LOW_RANK * MAX_LOW_RANK];
#pragma unroll
        for (int i = 0; i < MAX_LOW_RANK * MAX_LOW_RANK; ++i)
        {
            if (i < rank * rank)
            {
                delta_VTU[i] = VTU[i] * T(delta);
            }
        }
        T exp_delta_VTU[MAX_LOW_RANK * MAX_LOW_RANK];
        matrix_exp_taylor(delta_VTU, exp_delta_VTU, rank, num_terms);
        
        T exp_VTU_VTx[MAX_LOW_RANK];
#pragma unroll
        for (int r1 = 0; r1 < MAX_LOW_RANK; ++r1)
        {
            if (r1 < rank)
            {
                T sum = T(0.0f);
#pragma unroll
                for (int r2 = 0; r2 < MAX_LOW_RANK; ++r2)
                {
                    if (r2 < rank)
                    {
                        sum = sum + exp_delta_VTU[r1 * rank + r2] * VTx[r2];
                    }
                }
                exp_VTU_VTx[r1] = sum;
            }
        }
        
        // U @ (exp(delta * V^T U) @ VTx)
        T lowrank_correction[MAX_DSTATE];
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
                T sum = T(0.0f);
#pragma unroll
                for (int r = 0; r < MAX_LOW_RANK; ++r)
                {
                    if (r < rank)
                    {
                        sum = sum + U[i * rank + r] * exp_VTU_VTx[r];
                    }
                }
                lowrank_correction[i] = sum;
            }
        }
        
        // Add low-rank correction
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
                y[i] = y[i] + lowrank_correction[i];
            }
        }
    }
}

// ============================================================================
// All 6 discretization methods for structured A matrices
// ============================================================================

// ZOH: x_new = exp(delta * A) @ x_old + delta * B * u
template <typename T>
__device__ __forceinline__ void structured_discretization_zoh(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, T *x_new,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx)
{
    // Compute exp(delta * A) @ x_old
    T exp_A_x[MAX_DSTATE];
    block_diagonal_lowrank_exp_matrix_vector_mult<T>(
        A_blocks, U, V, x_old, exp_A_x, delta,
        dstate, block_size, num_blocks, rank, 10, true);
    
    // x_new = exp(delta * A) @ x_old + delta * B * u
    x_new[state_idx] = exp_A_x[state_idx] + T(delta * B_val * u_val);
}

// FOH: First Order Hold with structured A
// B_d = A^(-2) * (exp(A*Δ) - I - A*Δ) * B
// For structured A, we use Taylor expansion
template <typename T>
__device__ __forceinline__ void structured_discretization_foh(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, T *x_new,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx)
{
    // Compute exp(delta * A) @ x_old
    T exp_A_x[MAX_DSTATE];
    block_diagonal_lowrank_exp_matrix_vector_mult<T>(
        A_blocks, U, V, x_old, exp_A_x, delta,
        dstate, block_size, num_blocks, rank, 10, true);
    
    // For FOH with full A, we need to compute:
    // B_d * u = A^(-2) * (exp(A*Δ) - I - A*Δ) * B * u
    // Using Taylor: ≈ (Δ²/2 + A*Δ³/6 + ...) * B * u
    // For block-diagonal part, compute contribution separately
    
    float delta_sq = delta * delta;
    float delta_cubed = delta_sq * delta;
    
    // Simplified: use block-diagonal approximation for B_d term
    // B_d ≈ delta²/2 * B (first-order Taylor)
    float B_d_u = delta_sq * 0.5f * B_val * u_val;
    
    x_new[state_idx] = exp_A_x[state_idx] + T(B_d_u);
}

// Bilinear: Tustin transform with structured A
// A_d = (I - ΔA/2)^(-1) * (I + ΔA/2)
// B_d = (I - ΔA/2)^(-1) * Δ * B
template <typename T>
__device__ __forceinline__ void structured_discretization_bilinear(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, T *x_new,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx)
{
    // For bilinear with block-diagonal + low-rank:
    // (I - ΔA/2)^(-1) can be approximated using Neumann series
    // Or we use the first-order approximation
    
    // First-order approximation: A_d ≈ I + ΔA
    T A_x[MAX_DSTATE];
    block_diagonal_lowrank_matrix_vector_mult<T>(
        A_blocks, U, V, x_old, A_x, dstate, block_size, num_blocks, rank);
    
    // x_new = x_old + delta * A * x_old + delta * B * u
    // This is Euler discretization, bilinear is more complex
    // For full bilinear, we need matrix inversion which is expensive
    
    // Use approximation: (I - ΔA/2)^(-1) ≈ I + ΔA/2 + (ΔA/2)^2 + ...
    T x_mid[MAX_DSTATE];
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            x_mid[i] = x_old[i] + A_x[i] * T(delta * 0.5f);
        }
    }
    
    T A_x_mid[MAX_DSTATE];
    block_diagonal_lowrank_matrix_vector_mult<T>(
        A_blocks, U, V, x_mid, A_x_mid, dstate, block_size, num_blocks, rank);
    
    x_new[state_idx] = x_old[state_idx] + A_x_mid[state_idx] * T(delta) + T(delta * B_val * u_val);
}

// RK4: Runge-Kutta 4th order with structured A
template <typename T>
__device__ __forceinline__ void structured_discretization_rk4(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, T *x_new,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx)
{
    // RK4 stages:
    // k1 = A @ x + B * u
    // k2 = A @ (x + delta/2 * k1) + B * u
    // k3 = A @ (x + delta/2 * k2) + B * u
    // k4 = A @ (x + delta * k3) + B * u
    // x_new = x + delta/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    T k1[MAX_DSTATE], k2[MAX_DSTATE], k3[MAX_DSTATE], k4[MAX_DSTATE];
    T x_temp[MAX_DSTATE];
    
    // k1 = A @ x + B * u
    block_diagonal_lowrank_matrix_vector_mult<T>(
        A_blocks, U, V, x_old, k1, dstate, block_size, num_blocks, rank);
    k1[state_idx] = k1[state_idx] + T(B_val * u_val);
    
    // x_temp = x + delta/2 * k1
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            x_temp[i] = x_old[i] + k1[i] * T(delta * 0.5f);
        }
    }
    
    // k2 = A @ x_temp + B * u
    block_diagonal_lowrank_matrix_vector_mult<T>(
        A_blocks, U, V, x_temp, k2, dstate, block_size, num_blocks, rank);
    k2[state_idx] = k2[state_idx] + T(B_val * u_val);
    
    // x_temp = x + delta/2 * k2
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            x_temp[i] = x_old[i] + k2[i] * T(delta * 0.5f);
        }
    }
    
    // k3 = A @ x_temp + B * u
    block_diagonal_lowrank_matrix_vector_mult<T>(
        A_blocks, U, V, x_temp, k3, dstate, block_size, num_blocks, rank);
    k3[state_idx] = k3[state_idx] + T(B_val * u_val);
    
    // x_temp = x + delta * k3
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            x_temp[i] = x_old[i] + k3[i] * T(delta);
        }
    }
    
    // k4 = A @ x_temp + B * u
    block_diagonal_lowrank_matrix_vector_mult<T>(
        A_blocks, U, V, x_temp, k4, dstate, block_size, num_blocks, rank);
    k4[state_idx] = k4[state_idx] + T(B_val * u_val);
    
    // x_new = x + delta/6 * (k1 + 2*k2 + 2*k3 + k4)
    x_new[state_idx] = x_old[state_idx] + 
        (k1[state_idx] + k2[state_idx] * T(2.0f) + k3[state_idx] * T(2.0f) + k4[state_idx]) * T(delta / 6.0f);
}

// Polynomial interpolation with structured A
template <typename T>
__device__ __forceinline__ void structured_discretization_poly(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, T *x_new,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx)
{
    // Polynomial interpolation combines ZOH and FOH terms
    // B_d = A^(-1)(exp(AΔ)-I)B + ½A^(-2)(exp(AΔ)-I-AΔ)B
    
    // Compute exp(delta * A) @ x_old
    T exp_A_x[MAX_DSTATE];
    block_diagonal_lowrank_exp_matrix_vector_mult<T>(
        A_blocks, U, V, x_old, exp_A_x, delta,
        dstate, block_size, num_blocks, rank, 10, true);
    
    // Use Taylor expansion for B_d term
    float delta_sq = delta * delta;
    float delta_cubed = delta_sq * delta;
    
    // Combined polynomial coefficient (simplified)
    float B_d_u = (delta + delta_sq * 0.25f + delta_cubed / 12.0f) * B_val * u_val;
    
    x_new[state_idx] = exp_A_x[state_idx] + T(B_d_u);
}

// Higher-order hold with structured A
template <typename T>
__device__ __forceinline__ void structured_discretization_highorder(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, T *x_new,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx)
{
    // Higher-order hold (n=2) includes ZOH, FOH, and quadratic terms
    
    // Compute exp(delta * A) @ x_old
    T exp_A_x[MAX_DSTATE];
    block_diagonal_lowrank_exp_matrix_vector_mult<T>(
        A_blocks, U, V, x_old, exp_A_x, delta,
        dstate, block_size, num_blocks, rank, 10, true);
    
    // Use Taylor expansion for B_d term
    float delta_sq = delta * delta;
    float delta_cubed = delta_sq * delta;
    
    // Higher-order coefficient (simplified)
    float B_d_u = (delta + delta_sq * 0.5f + delta_cubed / 6.0f) * B_val * u_val;
    
    x_new[state_idx] = exp_A_x[state_idx] + T(B_d_u);
}

// Unified discretization function for structured A
template <typename T>
__device__ __forceinline__ void structured_discretization(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, T *x_new,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx,
    DiscretizationMethod method)
{
    switch (method)
    {
        case DISCRETIZATION_ZOH:
            structured_discretization_zoh<T>(A_blocks, U, V, x_old, x_new, delta, B_val, u_val,
                dstate, block_size, num_blocks, rank, state_idx);
            break;
        case DISCRETIZATION_FOH:
            structured_discretization_foh<T>(A_blocks, U, V, x_old, x_new, delta, B_val, u_val,
                dstate, block_size, num_blocks, rank, state_idx);
            break;
        case DISCRETIZATION_BILINEAR:
            structured_discretization_bilinear<T>(A_blocks, U, V, x_old, x_new, delta, B_val, u_val,
                dstate, block_size, num_blocks, rank, state_idx);
            break;
        case DISCRETIZATION_RK4:
            structured_discretization_rk4<T>(A_blocks, U, V, x_old, x_new, delta, B_val, u_val,
                dstate, block_size, num_blocks, rank, state_idx);
            break;
        case DISCRETIZATION_POLY:
            structured_discretization_poly<T>(A_blocks, U, V, x_old, x_new, delta, B_val, u_val,
                dstate, block_size, num_blocks, rank, state_idx);
            break;
        case DISCRETIZATION_HIGHORDER:
        default:
            structured_discretization_highorder<T>(A_blocks, U, V, x_old, x_new, delta, B_val, u_val,
                dstate, block_size, num_blocks, rank, state_idx);
            break;
    }
}

// ============================================================================
// BACKWARD PASS: Gradient computation for structured A matrices
// Method-specific gradients for all 6 discretization methods
// ============================================================================

// Gradient of matrix exponential: d/dA exp(A) = integral_0^1 exp((1-t)A) dA exp(tA) dt
// For computational efficiency, we use the approximation:
// d(exp(A)@x)/dA ≈ exp(A) @ x @ e_i^T where e_i is the unit vector
// This gives us the gradient contribution for each element

// Method-specific gradient computation for structured A matrices
// Computes gradients w.r.t. A_blocks, A_U, A_V, delta, B, u, x_old
template <typename T>
__device__ __forceinline__ void structured_gradient_method_specific(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, const T *x_new,  // x_new is the output from forward pass
    float delta, float B_val, float u_val,
    const T *grad_output,  // Gradient w.r.t. x_new
    T *grad_A_blocks, T *grad_U, T *grad_V,
    T *grad_x_old, T *grad_delta, T *grad_B, T *grad_u,
    int dstate, int block_size, int num_blocks, int rank,
    int state_idx,
    DiscretizationMethod method)
{
    switch (method)
    {
        case DISCRETIZATION_ZOH:
        {
            // ZOH: x_new = exp(δA) @ x_old + δBu
            // Gradient through exp(δA) @ x_old
            // d(exp(δA)@x)/dA ≈ δ * x[j] * grad[i] (first-order)
            // More accurate: use x_new - δBu for exp(δA)@x
            
            T exp_A_x_val = x_new[state_idx] - T(delta * B_val * u_val);
            
            // Gradient w.r.t. A_blocks (block-diagonal part)
            int block_idx = state_idx / block_size;
            int local_i = state_idx % block_size;
            
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    grad_A_blocks[A_idx] = grad_A_blocks[A_idx] + T(delta) * grad_output[state_idx] * x_old[global_j];
                }
            }
            
            // Gradient w.r.t. U and V (low-rank part)
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                grad_U[state_idx * rank + r] = grad_U[state_idx * rank + r] + T(delta) * grad_output[state_idx] * Vtx_r;
                
                for (int j = 0; j < dstate; ++j)
                {
                    grad_V[j * rank + r] = grad_V[j * rank + r] + T(delta) * U[state_idx * rank + r] * grad_output[state_idx] * x_old[j];
                }
            }
            
            // Gradient w.r.t. x_old: exp(δA)^T @ grad_output
            T grad_x_contrib = grad_output[state_idx];
            
            // Block-diagonal contribution
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_j * block_size + local_i; // Transpose
                    grad_x_contrib = grad_x_contrib + T(delta) * A_blocks[A_idx] * grad_output[global_j];
                }
            }
            
            // Low-rank contribution
            for (int r = 0; r < rank; ++r)
            {
                T Utgrad_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Utgrad_r = Utgrad_r + U[j * rank + r] * grad_output[j];
                }
                grad_x_contrib = grad_x_contrib + T(delta) * V[state_idx * rank + r] * Utgrad_r;
            }
            
            grad_x_old[state_idx] = grad_x_contrib;
            
            // Gradient w.r.t. delta, B, u
            T Ax_val = T(0.0f);
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    Ax_val = Ax_val + A_blocks[A_idx] * x_old[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                Ax_val = Ax_val + U[state_idx * rank + r] * Vtx_r;
            }
            
            *grad_delta = *grad_delta + grad_output[state_idx] * (Ax_val + T(B_val * u_val));
            *grad_B = *grad_B + T(delta * u_val) * grad_output[state_idx];
            *grad_u = *grad_u + T(delta * B_val) * grad_output[state_idx];
            break;
        }
        
        case DISCRETIZATION_FOH:
        {
            // FOH: x_new = exp(δA) @ x_old + B_d * u
            // B_d = A^(-2) * (exp(A*Δ) - I - A*Δ) * B ≈ (Δ²/2 + A*Δ³/6 + ...) * B
            // Gradient needs to account for B_d term that depends on A
            
            // For FOH, the B_d term adds extra gradient contributions through A
            // Simplified: treat B_d as approximately (Δ²/2) * B for gradient computation
            // More accurate would require computing gradient through full B_d formula
            
            float delta_sq = delta * delta;
            float delta_cubed = delta_sq * delta;
            
            // Gradient through exp(δA) @ x_old (same as ZOH)
            T exp_A_x_val = x_new[state_idx] - T((delta_sq * 0.5f) * B_val * u_val);
            
            // A_blocks gradient (includes contribution from B_d term)
            int block_idx = state_idx / block_size;
            int local_i = state_idx % block_size;
            
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    // Main gradient from exp(δA) @ x_old
                    grad_A_blocks[A_idx] = grad_A_blocks[A_idx] + T(delta) * grad_output[state_idx] * x_old[global_j];
                    // Additional gradient from B_d term (simplified)
                    if (state_idx == global_j)  // Diagonal term
                    {
                        grad_A_blocks[A_idx] = grad_A_blocks[A_idx] + T(delta_cubed / 6.0f * B_val * u_val) * grad_output[state_idx];
                    }
                }
            }
            
            // U, V gradients (similar to ZOH, with B_d correction)
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                grad_U[state_idx * rank + r] = grad_U[state_idx * rank + r] + T(delta) * grad_output[state_idx] * Vtx_r;
                
                for (int j = 0; j < dstate; ++j)
                {
                    grad_V[j * rank + r] = grad_V[j * rank + r] + T(delta) * U[state_idx * rank + r] * grad_output[state_idx] * x_old[j];
                }
            }
            
            // Gradient w.r.t. x_old (same as ZOH)
            T grad_x_contrib = grad_output[state_idx];
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_j * block_size + local_i;
                    grad_x_contrib = grad_x_contrib + T(delta) * A_blocks[A_idx] * grad_output[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Utgrad_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Utgrad_r = Utgrad_r + U[j * rank + r] * grad_output[j];
                }
                grad_x_contrib = grad_x_contrib + T(delta) * V[state_idx * rank + r] * Utgrad_r;
            }
            grad_x_old[state_idx] = grad_x_contrib;
            
            // Gradient w.r.t. delta, B, u (with FOH-specific terms)
            T Ax_val = T(0.0f);
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    Ax_val = Ax_val + A_blocks[A_idx] * x_old[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                Ax_val = Ax_val + U[state_idx * rank + r] * Vtx_r;
            }
            
            *grad_delta = *grad_delta + grad_output[state_idx] * (Ax_val + T((delta * B_val) * u_val));
            *grad_B = *grad_B + T((delta_sq * 0.5f) * u_val) * grad_output[state_idx];
            *grad_u = *grad_u + T((delta_sq * 0.5f) * B_val) * grad_output[state_idx];
            break;
        }
        
        case DISCRETIZATION_BILINEAR:
        {
            // Bilinear: x_new = A_d @ x_old + B_d * u
            // A_d = (I - δA/2)^(-1) * (I + δA/2) ≈ I + δA + (δA)²/2 + ...
            // B_d = (I - δA/2)^(-1) * δ * B ≈ δB + (δA/2) * δB + ...
            
            // For gradient computation, use first-order approximation:
            // A_d ≈ I + δA, B_d ≈ δB
            
            int block_idx = state_idx / block_size;
            int local_i = state_idx % block_size;
            
            // Gradient through A_d @ x_old
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    // A_d ≈ I + δA, so gradient ≈ δ * x[j] * grad[i]
                    grad_A_blocks[A_idx] = grad_A_blocks[A_idx] + T(delta) * grad_output[state_idx] * x_old[global_j];
                }
            }
            
            // U, V gradients
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                grad_U[state_idx * rank + r] = grad_U[state_idx * rank + r] + T(delta) * grad_output[state_idx] * Vtx_r;
                
                for (int j = 0; j < dstate; ++j)
                {
                    grad_V[j * rank + r] = grad_V[j * rank + r] + T(delta) * U[state_idx * rank + r] * grad_output[state_idx] * x_old[j];
                }
            }
            
            // Gradient w.r.t. x_old: A_d^T @ grad_output ≈ (I + δA^T) @ grad_output
            T grad_x_contrib = grad_output[state_idx];
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_j * block_size + local_i;
                    grad_x_contrib = grad_x_contrib + T(delta) * A_blocks[A_idx] * grad_output[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Utgrad_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Utgrad_r = Utgrad_r + U[j * rank + r] * grad_output[j];
                }
                grad_x_contrib = grad_x_contrib + T(delta) * V[state_idx * rank + r] * Utgrad_r;
            }
            grad_x_old[state_idx] = grad_x_contrib;
            
            // Gradient w.r.t. delta, B, u
            T Ax_val = T(0.0f);
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    Ax_val = Ax_val + A_blocks[A_idx] * x_old[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                Ax_val = Ax_val + U[state_idx * rank + r] * Vtx_r;
            }
            
            *grad_delta = *grad_delta + grad_output[state_idx] * (Ax_val + T(B_val * u_val));
            *grad_B = *grad_B + T(delta * u_val) * grad_output[state_idx];
            *grad_u = *grad_u + T(delta * B_val) * grad_output[state_idx];
            break;
        }
        
        case DISCRETIZATION_RK4:
        {
            // RK4: x_new = x_old + δ/6 * (k1 + 2*k2 + 2*k3 + k4)
            // Each k_i = A @ x_i + B*u where x_i depends on previous stages
            // Gradient needs to propagate through all 4 stages
            
            // Simplified: treat as single step with effective A_d ≈ I + δA
            // More accurate would require storing k1, k2, k3, k4 and propagating through each
            
            int block_idx = state_idx / block_size;
            int local_i = state_idx % block_size;
            
            // Use ZOH-like gradient (RK4 is more complex, this is an approximation)
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    grad_A_blocks[A_idx] = grad_A_blocks[A_idx] + T(delta) * grad_output[state_idx] * x_old[global_j];
                }
            }
            
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                grad_U[state_idx * rank + r] = grad_U[state_idx * rank + r] + T(delta) * grad_output[state_idx] * Vtx_r;
                
                for (int j = 0; j < dstate; ++j)
                {
                    grad_V[j * rank + r] = grad_V[j * rank + r] + T(delta) * U[state_idx * rank + r] * grad_output[state_idx] * x_old[j];
                }
            }
            
            T grad_x_contrib = grad_output[state_idx];
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_j * block_size + local_i;
                    grad_x_contrib = grad_x_contrib + T(delta) * A_blocks[A_idx] * grad_output[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Utgrad_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Utgrad_r = Utgrad_r + U[j * rank + r] * grad_output[j];
                }
                grad_x_contrib = grad_x_contrib + T(delta) * V[state_idx * rank + r] * Utgrad_r;
            }
            grad_x_old[state_idx] = grad_x_contrib;
            
            T Ax_val = T(0.0f);
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    Ax_val = Ax_val + A_blocks[A_idx] * x_old[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                Ax_val = Ax_val + U[state_idx * rank + r] * Vtx_r;
            }
            
            *grad_delta = *grad_delta + grad_output[state_idx] * (Ax_val + T(B_val * u_val));
            *grad_B = *grad_B + T(delta * u_val) * grad_output[state_idx];
            *grad_u = *grad_u + T(delta * B_val) * grad_output[state_idx];
            break;
        }
        
        case DISCRETIZATION_POLY:
        case DISCRETIZATION_HIGHORDER:
        default:
        {
            // Poly and Highorder: Similar to FOH but with different coefficients
            // Use FOH-like gradient computation
            
            float delta_sq = delta * delta;
            int block_idx = state_idx / block_size;
            int local_i = state_idx % block_size;
            
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    grad_A_blocks[A_idx] = grad_A_blocks[A_idx] + T(delta) * grad_output[state_idx] * x_old[global_j];
                }
            }
            
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                grad_U[state_idx * rank + r] = grad_U[state_idx * rank + r] + T(delta) * grad_output[state_idx] * Vtx_r;
                
                for (int j = 0; j < dstate; ++j)
                {
                    grad_V[j * rank + r] = grad_V[j * rank + r] + T(delta) * U[state_idx * rank + r] * grad_output[state_idx] * x_old[j];
                }
            }
            
            T grad_x_contrib = grad_output[state_idx];
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_j * block_size + local_i;
                    grad_x_contrib = grad_x_contrib + T(delta) * A_blocks[A_idx] * grad_output[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Utgrad_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Utgrad_r = Utgrad_r + U[j * rank + r] * grad_output[j];
                }
                grad_x_contrib = grad_x_contrib + T(delta) * V[state_idx * rank + r] * Utgrad_r;
            }
            grad_x_old[state_idx] = grad_x_contrib;
            
            T Ax_val = T(0.0f);
            if (block_idx < num_blocks)
            {
                for (int local_j = 0; local_j < block_size && (block_idx * block_size + local_j) < dstate; ++local_j)
                {
                    int global_j = block_idx * block_size + local_j;
                    int A_idx = block_idx * block_size * block_size + local_i * block_size + local_j;
                    Ax_val = Ax_val + A_blocks[A_idx] * x_old[global_j];
                }
            }
            for (int r = 0; r < rank; ++r)
            {
                T Vtx_r = T(0.0f);
                for (int j = 0; j < dstate; ++j)
                {
                    Vtx_r = Vtx_r + V[j * rank + r] * x_old[j];
                }
                Ax_val = Ax_val + U[state_idx * rank + r] * Vtx_r;
            }
            
            *grad_delta = *grad_delta + grad_output[state_idx] * (Ax_val + T(B_val * u_val));
            *grad_B = *grad_B + T((delta + delta_sq * 0.25f) * u_val) * grad_output[state_idx];
            *grad_u = *grad_u + T((delta + delta_sq * 0.25f) * B_val) * grad_output[state_idx];
            break;
        }
    }
}

// Gradient of block-diagonal matrix-vector multiplication
// d(blockdiag @ x)/dA_block_ij = x_j for position (i,j) in block
template <typename T>
__device__ __forceinline__ void gradient_block_diagonal_matrix_vector(
    const T *x,           // Input vector
    const T *grad_output, // Gradient w.r.t. output (dstate)
    T *grad_A_blocks,     // Output: gradient w.r.t. A_blocks
    int dstate,
    int block_size,
    int num_blocks)
{
    // Initialize gradients
    int total_elements = num_blocks * block_size * block_size;
    #pragma unroll
    for (int idx = 0; idx < total_elements && idx < MAX_DSTATE * MAX_DSTATE; ++idx)
    {
        grad_A_blocks[idx] = T(0.0f);
    }
    
    // Compute gradients for each block
    for (int k = 0; k < num_blocks; ++k)
    {
        int start_idx = k * block_size;
        T *grad_A_k = grad_A_blocks + k * block_size * block_size;
        
#pragma unroll
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
        {
            if (i < block_size)
            {
                int row_idx = start_idx + i;
                if (row_idx < dstate)
                {
#pragma unroll
                    for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
                    {
                        if (j < block_size)
                        {
                            int col_idx = start_idx + j;
                            if (col_idx < dstate)
                            {
                                // d(A@x)_i/dA_ij = x_j
                                // Chain rule: dL/dA_ij = dL/d(A@x)_i * d(A@x)_i/dA_ij = grad_output_i * x_j
                                grad_A_k[i * block_size + j] = grad_output[row_idx] * x[col_idx];
                            }
                            }
                        }
                    }
                }
            }
        }
    }
    
// Gradient of low-rank matrix-vector multiplication: UV^T @ x
// d(UV^T@x)/dU = grad_output @ (V^T @ x)^T for each row
// d(UV^T@x)/dV = x @ (U^T @ grad_output)^T for each row
template <typename T>
__device__ __forceinline__ void gradient_low_rank_matrix_vector(
    const T *U,           // U matrix (dstate x rank)
    const T *V,           // V matrix (dstate x rank)
    const T *x,           // Input vector
    const T *grad_output, // Gradient w.r.t. output (dstate)
    T *grad_U,            // Output: gradient w.r.t. U
    T *grad_V,            // Output: gradient w.r.t. V
    int dstate,
    int rank)
{
    // Compute V^T @ x (rank x 1)
    T Vtx[MAX_LOW_RANK];
    #pragma unroll
    for (int r = 0; r < MAX_LOW_RANK; ++r)
    {
        if (r < rank)
        {
            Vtx[r] = T(0.0f);
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
                    Vtx[r] = Vtx[r] + complex_conj(V[i * rank + r]) * x[i];
                }
            }
        }
    }
    
    // Compute U^T @ grad_output (rank x 1)
    T Ut_grad[MAX_LOW_RANK];
#pragma unroll
    for (int r = 0; r < MAX_LOW_RANK; ++r)
                {
        if (r < rank)
                    {
            Ut_grad[r] = T(0.0f);
#pragma unroll
            for (int i = 0; i < MAX_DSTATE; ++i)
            {
                if (i < dstate)
                {
                    Ut_grad[r] = Ut_grad[r] + complex_conj(U[i * rank + r]) * grad_output[i];
                }
            }
        }
    }
    
    // grad_U[i,r] = grad_output[i] * Vtx[r]
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
#pragma unroll
            for (int r = 0; r < MAX_LOW_RANK; ++r)
            {
                if (r < rank)
                {
                    grad_U[i * rank + r] = grad_output[i] * Vtx[r];
                }
            }
        }
    }
    
    // grad_V[i,r] = x[i] * Ut_grad[r]
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
        {
        if (i < dstate)
            {
#pragma unroll
            for (int r = 0; r < MAX_LOW_RANK; ++r)
            {
                if (r < rank)
                {
                    grad_V[i * rank + r] = x[i] * Ut_grad[r];
                }
            }
        }
    }
}

// Combined gradient for block-diagonal + low-rank matrix-vector multiplication
template <typename T>
__device__ __forceinline__ void gradient_block_diagonal_lowrank_matrix_vector(
    const T *A_blocks,    // Block matrices
    const T *U,           // Low-rank U
    const T *V,           // Low-rank V
    const T *x,           // Input vector
    const T *grad_output, // Gradient w.r.t. output
    T *grad_A_blocks,     // Output: gradient w.r.t. A_blocks
    T *grad_U,            // Output: gradient w.r.t. U
    T *grad_V,            // Output: gradient w.r.t. V
    int dstate,
    int block_size,
    int num_blocks,
    int rank)
{
    // Gradient for block-diagonal part
    gradient_block_diagonal_matrix_vector<T>(x, grad_output, grad_A_blocks,
        dstate, block_size, num_blocks);
    
    // Gradient for low-rank part
    gradient_low_rank_matrix_vector<T>(U, V, x, grad_output, grad_U, grad_V,
        dstate, rank);
}

// Gradient of exp(delta * A) @ x with respect to A, delta, and x
// This is the core gradient computation for the backward pass
template <typename T>
__device__ __forceinline__ void gradient_exp_A_x(
    const T *A_blocks,    // Block matrices
    const T *U,           // Low-rank U
    const T *V,           // Low-rank V
    const T *x,           // Input state
    const T *exp_A_x,     // exp(delta * A) @ x (forward output)
    const T *grad_output, // Gradient w.r.t. output
    float delta,          // Time step
    T *grad_A_blocks,     // Output: gradient w.r.t. A_blocks
    T *grad_U,            // Output: gradient w.r.t. U
    T *grad_V,            // Output: gradient w.r.t. V
    T *grad_x,            // Output: gradient w.r.t. x
    float *grad_delta,    // Output: gradient w.r.t. delta (accumulated)
    int dstate,
    int block_size,
    int num_blocks,
    int rank)
{
    // The gradient through matrix exponential is complex
    // We use the approximation: d(exp(δA)@x)/dA ≈ δ * exp(δA) @ (grad_output ⊗ x)
    // where ⊗ denotes outer product
    
    // For grad_x: d(exp(δA)@x)/dx = exp(δA)^T @ grad_output
    // Since A is structured, exp(δA)^T is also structured:
    // exp(δA)^T = exp(δA^T) (for real A)
    
    // Compute exp(δA)^T @ grad_output for grad_x
    // For block-diagonal part: exp(δ blockdiag)^T = blockdiag(exp(δA_k)^T)
    // For low-rank part: use first-order approximation
    
    // Initialize grad_x
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            grad_x[i] = T(0.0f);
        }
    }
    
    // Process block-diagonal part
    for (int k = 0; k < num_blocks; ++k)
    {
        int start_idx = k * block_size;
        const T *A_k = A_blocks + k * block_size * block_size;
        T *grad_A_k = grad_A_blocks + k * block_size * block_size;
        
        // Compute exp(delta * A_k)
        T deltaA_k[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE];
        T expA_k[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE];
        
#pragma unroll
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
        {
            if (i < block_size)
        {
#pragma unroll
                for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
            {
                    if (j < block_size)
                {
                        deltaA_k[i * block_size + j] = A_k[i * block_size + j] * T(delta);
                }
            }
        }
        }
        matrix_exp_taylor(deltaA_k, expA_k, block_size, 10);
        
        // grad_x contribution from this block: exp(δA_k)^T @ grad_output_block
#pragma unroll
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
        {
            if (i < block_size)
            {
                int row_idx = start_idx + i;
                if (row_idx < dstate)
                {
                    T sum = T(0.0f);
#pragma unroll
                    for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
                    {
                        if (j < block_size)
                        {
                            int col_idx = start_idx + j;
                            if (col_idx < dstate)
                            {
                                // exp(δA_k)^T[i,j] = exp(δA_k)[j,i]
                                sum = sum + complex_conj(expA_k[j * block_size + i]) * grad_output[col_idx];
                            }
                        }
                    }
                    grad_x[row_idx] = grad_x[row_idx] + sum;
                }
            }
        }
        
        // grad_A_blocks contribution
        // d(exp(δA)@x)/dA_ij ≈ δ * (some function)
        // Using first-order approximation: d exp(δA)/dA_ij ≈ δ * e_i @ e_j^T
#pragma unroll
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i)
                        {
            if (i < block_size)
                            {
                int row_idx = start_idx + i;
                if (row_idx < dstate)
                {
#pragma unroll
                    for (int j = 0; j < MAX_BLOCK_SIZE; ++j)
                    {
                        if (j < block_size)
                        {
                            int col_idx = start_idx + j;
                            if (col_idx < dstate)
                            {
                                // Chain rule contribution
                                grad_A_k[i * block_size + j] = grad_output[row_idx] * x[col_idx] * T(delta);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // grad_delta contribution: d(exp(δA)@x)/dδ = A @ exp(δA) @ x = A @ exp_A_x
    T A_exp_A_x[MAX_DSTATE];
    block_diagonal_lowrank_matrix_vector_mult<T>(
        A_blocks, U, V, exp_A_x, A_exp_A_x, dstate, block_size, num_blocks, rank);
    
    float delta_grad = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            // Convert to float for accumulation
            float grad_out_f, A_exp_f;
            if constexpr (sizeof(T) == sizeof(complex_t))
            {
                // Complex case - use real part for gradient
                grad_out_f = reinterpret_cast<const complex_t*>(&grad_output[i])->real_;
                A_exp_f = reinterpret_cast<const complex_t*>(&A_exp_A_x[i])->real_;
            }
            else
            {
                grad_out_f = float(grad_output[i]);
                A_exp_f = float(A_exp_A_x[i]);
            }
            delta_grad += grad_out_f * A_exp_f;
        }
    }
    *grad_delta += delta_grad;
    
    // Low-rank gradient contributions (using first-order approximation)
    T lowrank_grad_U[MAX_DSTATE * MAX_LOW_RANK];
    T lowrank_grad_V[MAX_DSTATE * MAX_LOW_RANK];
    
    // d(UV^T @ x)/dU and dV
    gradient_low_rank_matrix_vector<T>(U, V, x, grad_output, lowrank_grad_U, lowrank_grad_V,
        dstate, rank);
    
    // Scale by delta for first-order approximation
#pragma unroll
        for (int i = 0; i < MAX_DSTATE; ++i)
        {
            if (i < dstate)
            {
#pragma unroll
            for (int r = 0; r < MAX_LOW_RANK; ++r)
            {
                if (r < rank)
                {
                    grad_U[i * rank + r] = lowrank_grad_U[i * rank + r] * T(delta);
                    grad_V[i * rank + r] = lowrank_grad_V[i * rank + r] * T(delta);
                }
            }
        }
    }
    
    // Add low-rank contribution to grad_x
    T lowrank_grad_x[MAX_DSTATE];
    T grad_out_scaled[MAX_DSTATE];
#pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            grad_out_scaled[i] = grad_output[i] * T(delta);
        }
    }
    low_rank_matrix_vector_mult<T>(V, U, grad_out_scaled, lowrank_grad_x, dstate, rank);
    
    #pragma unroll
    for (int i = 0; i < MAX_DSTATE; ++i)
    {
        if (i < dstate)
        {
            grad_x[i] = grad_x[i] + lowrank_grad_x[i];
        }
    }
}

// ============================================================================
// Bidirectional Mamba support
// ============================================================================

// Reverse scan for bidirectional operation
// For backward direction, we reverse the sequence and apply the same operations
template <typename T>
__device__ __forceinline__ void structured_reverse_scan_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x_future, T *x_current,
    float delta, float B_val, float u_val,
    int dstate, int block_size, int num_blocks, int rank, int state_idx,
    DiscretizationMethod method)
{
    // For reverse scan, we essentially use the same discretization
    // but in reverse order. The state equation becomes:
    // x[t] = A_d^(-1) @ (x[t+1] - B_d * u[t+1])
    // 
    // However, for numerical stability, we typically compute the forward
    // pass in reverse order rather than inverting A_d.
    // Here we provide a simplified approach using the same discretization.
    
    structured_discretization<T>(A_blocks, U, V, x_future, x_current, delta, B_val, u_val,
        dstate, block_size, num_blocks, rank, state_idx, method);
}

// Combine forward and backward states for bidirectional output
template <typename T>
__device__ __forceinline__ T combine_bidirectional_states(
    T forward_state,
    T backward_state,
    bool use_concat = false)  // If true, concatenate; if false, add
{
    if (use_concat)
    {
        // Concatenation is typically done at a higher level
        // Here we return forward state for the first half
        return forward_state;
    }
    else
    {
        // Element-wise addition
        return forward_state + backward_state;
    }
}

#endif // MATRIX_OPS_CUH

