/******************************************************************************
 * Feature-SST: Structured State Transitions - Block-Diagonal + Low-Rank A
 * 
 * SST = Structured State Transitions: A = blockdiag(A_1, ..., A_K) + UV^T
 * This file contains comprehensive matrix operations for:
 * 1. All 6 discretization methods (ZOH, FOH, Bilinear, RK4, Poly, High-order)
 * 2. Complex number support
 * 3. Backward pass gradient computation
 * 4. Bidirectional Mamba support
 ******************************************************************************/

#pragma once

#include "selective_scan_common.h"
#include "selective_scan.h"
#include <cuda_runtime.h>

// Maximum dimensions for on-device operations
#define SST_MAX_DSTATE 64
#define SST_MAX_BLOCK_SIZE 16
#define SST_MAX_RANK 16
#define SST_MAX_BLOCKS 16

//=============================================================================
// PART 1: Complex Number Operations
//=============================================================================

// Complex type wrapper for CUDA
template <typename T>
struct ComplexSST {
    T real;
    T imag;
    
    __device__ __host__ ComplexSST() : real(T(0)), imag(T(0)) {}
    __device__ __host__ ComplexSST(T r) : real(r), imag(T(0)) {}
    __device__ __host__ ComplexSST(T r, T i) : real(r), imag(i) {}
    
    __device__ __host__ ComplexSST operator+(const ComplexSST& other) const {
        return ComplexSST(real + other.real, imag + other.imag);
    }
    
    __device__ __host__ ComplexSST operator-(const ComplexSST& other) const {
        return ComplexSST(real - other.real, imag - other.imag);
    }
    
    __device__ __host__ ComplexSST operator*(const ComplexSST& other) const {
        return ComplexSST(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
    
    __device__ __host__ ComplexSST operator*(T scalar) const {
        return ComplexSST(real * scalar, imag * scalar);
    }
    
    __device__ __host__ ComplexSST operator/(T scalar) const {
        return ComplexSST(real / scalar, imag / scalar);
    }
    
    __device__ __host__ ComplexSST conj() const {
        return ComplexSST(real, -imag);
    }
    
    __device__ __host__ T norm_sq() const {
        return real * real + imag * imag;
    }
    
    __device__ __host__ ComplexSST operator/(const ComplexSST& other) const {
        T denom = other.norm_sq();
        return ComplexSST(
            (real * other.real + imag * other.imag) / denom,
            (imag * other.real - real * other.imag) / denom
        );
    }
    
    __device__ __host__ ComplexSST& operator+=(const ComplexSST& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
};

using ComplexFloat = ComplexSST<float>;

// Complex exponential: exp(z) = exp(real) * (cos(imag) + i*sin(imag))
__device__ __forceinline__ ComplexFloat cexpf_sst(ComplexFloat z) {
    float exp_real = expf(z.real);
    float s, c;
    sincosf(z.imag, &s, &c);
    return ComplexFloat(exp_real * c, exp_real * s);
}

// Complex exp2: exp2(z) = 2^z
__device__ __forceinline__ ComplexFloat cexp2f_sst(ComplexFloat z) {
    float exp_real = exp2f(z.real);
    float angle = z.imag * 0.693147180559945309417f; // ln(2)
    float s, c;
    sincosf(angle, &s, &c);
    return ComplexFloat(exp_real * c, exp_real * s);
}

//=============================================================================
// PART 2: Basic Matrix Operations (Real and Complex)
//=============================================================================

// Initialize matrix to identity
template <typename T>
__device__ __forceinline__ void sst_matrix_identity(T *A, int n) {
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < SST_MAX_DSTATE; ++j) {
                if (j < n) {
                    A[i * n + j] = (i == j) ? T(1) : T(0);
                }
            }
        }
    }
}

// Initialize matrix to zero
template <typename T>
__device__ __forceinline__ void sst_matrix_zero(T *A, int n) {
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < SST_MAX_DSTATE; ++j) {
                if (j < n) {
                    A[i * n + j] = T(0);
                }
            }
        }
    }
}

// Matrix copy: B = A
template <typename T>
__device__ __forceinline__ void sst_matrix_copy(const T *A, T *B, int n) {
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < SST_MAX_DSTATE; ++j) {
                if (j < n) {
                    B[i * n + j] = A[i * n + j];
                }
            }
        }
    }
}

// Matrix addition: C = A + B
template <typename T>
__device__ __forceinline__ void sst_matrix_add(const T *A, const T *B, T *C, int n) {
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < SST_MAX_DSTATE; ++j) {
                if (j < n) {
                    C[i * n + j] = A[i * n + j] + B[i * n + j];
                }
            }
        }
    }
}

// Matrix scalar multiplication: B = scalar * A
template <typename T>
__device__ __forceinline__ void sst_matrix_scale(const T *A, T *B, float scalar, int n) {
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < SST_MAX_DSTATE; ++j) {
                if (j < n) {
                    B[i * n + j] = T(scalar) * A[i * n + j];
                }
            }
        }
    }
}

// Matrix multiplication: C = A * B
template <typename T>
__device__ __forceinline__ void sst_matrix_mult(const T *A, const T *B, T *C, int n) {
    T temp[SST_MAX_DSTATE * SST_MAX_DSTATE];
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < SST_MAX_DSTATE; ++j) {
                if (j < n) {
                    T sum = T(0);
                    #pragma unroll
                    for (int k = 0; k < SST_MAX_DSTATE; ++k) {
                        if (k < n) {
                            sum = sum + A[i * n + k] * B[k * n + j];
                        }
                    }
                    temp[i * n + j] = sum;
                }
            }
        }
    }
    sst_matrix_copy(temp, C, n);
}

// Matrix-vector multiplication: y = A * x
template <typename T>
__device__ __forceinline__ void sst_matvec_mult(const T *A, const T *x, T *y, int n) {
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < n) {
            T sum = T(0);
            #pragma unroll
            for (int j = 0; j < SST_MAX_DSTATE; ++j) {
                if (j < n) {
                    sum = sum + A[i * n + j] * x[j];
                }
            }
            y[i] = sum;
        }
    }
}

//=============================================================================
// PART 3: Matrix Exponential (Taylor Series and Padé Approximation)
//=============================================================================

// Matrix exponential using Taylor series: exp(A) ≈ I + A + A²/2! + A³/3! + ...
template <typename T>
__device__ __forceinline__ void sst_matrix_exp_taylor(
    const T *A, T *expA, int n, int num_terms = 10)
{
    // Initialize expA = I
    sst_matrix_identity(expA, n);
    
    // A^k storage
    T Ak[SST_MAX_DSTATE * SST_MAX_DSTATE];
    T temp[SST_MAX_DSTATE * SST_MAX_DSTATE];
    sst_matrix_copy(A, Ak, n);
    
    float factorial = 1.0f;
    for (int k = 1; k <= num_terms; ++k) {
        factorial *= k;
        
        // Add A^k / k! to expA
        #pragma unroll
        for (int i = 0; i < n; ++i) {
            #pragma unroll
            for (int j = 0; j < n; ++j) {
                expA[i * n + j] = expA[i * n + j] + Ak[i * n + j] / T(factorial);
            }
        }
        
        // Compute A^(k+1) = A^k * A
        if (k < num_terms) {
            sst_matrix_mult(Ak, A, temp, n);
            sst_matrix_copy(temp, Ak, n);
        }
    }
}

// Matrix exponential scaled by delta: exp(delta * A)
template <typename T>
__device__ __forceinline__ void sst_matrix_exp_scaled(
    const T *A, T *expA, float delta, int n, int num_terms = 10)
{
    T deltaA[SST_MAX_DSTATE * SST_MAX_DSTATE];
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int j = 0; j < n; ++j) {
            deltaA[i * n + j] = T(delta) * A[i * n + j];
        }
    }
    sst_matrix_exp_taylor(deltaA, expA, n, num_terms);
}

// Padé approximation for matrix exponential (more accurate for larger matrices)
// exp(A) ≈ N(A) / D(A) where N and D are polynomials
// For order p=6: N = I + A/2 + A²/9 + A³/72 + A⁴/1008 + A⁵/30240 + A⁶/665280
//                D = I - A/2 + A²/9 - A³/72 + A⁴/1008 - A⁵/30240 + A⁶/665280
template <typename T>
__device__ __forceinline__ void sst_matrix_exp_pade6(
    const T *A, T *expA, int n)
{
    // Padé coefficients for order 6
    const float c[] = {1.0f, 0.5f, 1.0f/9.0f, 1.0f/72.0f, 1.0f/1008.0f, 1.0f/30240.0f, 1.0f/665280.0f};
    
    // Compute powers of A
    T A2[SST_MAX_DSTATE * SST_MAX_DSTATE];
    T A3[SST_MAX_DSTATE * SST_MAX_DSTATE];
    T A4[SST_MAX_DSTATE * SST_MAX_DSTATE];
    T A5[SST_MAX_DSTATE * SST_MAX_DSTATE];
    T A6[SST_MAX_DSTATE * SST_MAX_DSTATE];
    
    sst_matrix_mult(A, A, A2, n);
    sst_matrix_mult(A2, A, A3, n);
    sst_matrix_mult(A3, A, A4, n);
    sst_matrix_mult(A4, A, A5, n);
    sst_matrix_mult(A5, A, A6, n);
    
    // Compute N = I + c[1]*A + c[2]*A² + c[3]*A³ + c[4]*A⁴ + c[5]*A⁵ + c[6]*A⁶
    T N[SST_MAX_DSTATE * SST_MAX_DSTATE];
    sst_matrix_identity(N, n);
    
    // Compute D = I - c[1]*A + c[2]*A² - c[3]*A³ + c[4]*A⁴ - c[5]*A⁵ + c[6]*A⁶
    T D[SST_MAX_DSTATE * SST_MAX_DSTATE];
    sst_matrix_identity(D, n);
    
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            N[idx] = N[idx] + T(c[1]) * A[idx] + T(c[2]) * A2[idx] + T(c[3]) * A3[idx] + 
                     T(c[4]) * A4[idx] + T(c[5]) * A5[idx] + T(c[6]) * A6[idx];
            D[idx] = D[idx] - T(c[1]) * A[idx] + T(c[2]) * A2[idx] - T(c[3]) * A3[idx] + 
                     T(c[4]) * A4[idx] - T(c[5]) * A5[idx] + T(c[6]) * A6[idx];
        }
    }
    
    // Compute expA = D^(-1) * N using Gauss-Jordan elimination
    // For small matrices, this is acceptable
    // Copy D to augmented matrix [D | I]
    T aug[SST_MAX_DSTATE * 2 * SST_MAX_DSTATE];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i * (2 * n) + j] = D[i * n + j];
            aug[i * (2 * n) + n + j] = (i == j) ? T(1) : T(0);
        }
    }
    
    // Gauss-Jordan elimination
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            // Using magnitude for comparison
            float mag_cur = aug[pivot * (2 * n) + col] * aug[pivot * (2 * n) + col];
            float mag_new = aug[row * (2 * n) + col] * aug[row * (2 * n) + col];
            if (mag_new > mag_cur) {
                pivot = row;
            }
        }
        
        // Swap rows
        if (pivot != col) {
            for (int j = 0; j < 2 * n; ++j) {
                T tmp = aug[col * (2 * n) + j];
                aug[col * (2 * n) + j] = aug[pivot * (2 * n) + j];
                aug[pivot * (2 * n) + j] = tmp;
            }
        }
        
        // Scale pivot row
        T pivot_val = aug[col * (2 * n) + col];
        if (pivot_val * pivot_val > 1e-10f) {
            for (int j = 0; j < 2 * n; ++j) {
                aug[col * (2 * n) + j] = aug[col * (2 * n) + j] / pivot_val;
            }
        }
        
        // Eliminate column
        for (int row = 0; row < n; ++row) {
            if (row != col) {
                T factor = aug[row * (2 * n) + col];
                for (int j = 0; j < 2 * n; ++j) {
                    aug[row * (2 * n) + j] = aug[row * (2 * n) + j] - factor * aug[col * (2 * n) + j];
                }
            }
        }
    }
    
    // Extract D^(-1)
    T Dinv[SST_MAX_DSTATE * SST_MAX_DSTATE];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Dinv[i * n + j] = aug[i * (2 * n) + n + j];
        }
    }
    
    // Compute expA = D^(-1) * N
    sst_matrix_mult(Dinv, N, expA, n);
}

//=============================================================================
// PART 4: Block-Diagonal + Low-Rank Matrix Operations
//=============================================================================

// Block-diagonal matrix-vector multiplication: y = blockdiag(A_1, ..., A_K) * x
template <typename T>
__device__ __forceinline__ void sst_block_diagonal_matvec(
    const T *A_blocks, const T *x, T *y,
    int dstate, int block_size, int num_blocks)
{
    // Initialize output
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            y[i] = T(0);
        }
    }
    
    // Process each block
    for (int k = 0; k < num_blocks; ++k) {
        int start_idx = k * block_size;
        const T *A_k = A_blocks + k * block_size * block_size;
        
        #pragma unroll
        for (int i = 0; i < SST_MAX_BLOCK_SIZE; ++i) {
            if (i < block_size) {
                int row = start_idx + i;
                if (row < dstate) {
                    T sum = T(0);
                    #pragma unroll
                    for (int j = 0; j < SST_MAX_BLOCK_SIZE; ++j) {
                        if (j < block_size) {
                            int col = start_idx + j;
                            if (col < dstate) {
                                sum = sum + A_k[i * block_size + j] * x[col];
                            }
                        }
                    }
                    y[row] = sum;
                }
            }
        }
    }
}

// Low-rank matrix-vector multiplication: y = (UV^T) * x = U * (V^T * x)
template <typename T>
__device__ __forceinline__ void sst_lowrank_matvec(
    const T *U, const T *V, const T *x, T *y,
    int dstate, int rank)
{
    // Compute V^T * x (result is rank x 1)
    T Vtx[SST_MAX_RANK];
    #pragma unroll
    for (int r = 0; r < SST_MAX_RANK; ++r) {
        if (r < rank) {
            T sum = T(0);
            #pragma unroll
            for (int i = 0; i < SST_MAX_DSTATE; ++i) {
                if (i < dstate) {
                    sum = sum + V[i * rank + r] * x[i];
                }
            }
            Vtx[r] = sum;
        }
    }
    
    // Compute y = U * (V^T * x)
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            T sum = T(0);
            #pragma unroll
            for (int r = 0; r < SST_MAX_RANK; ++r) {
                if (r < rank) {
                    sum = sum + U[i * rank + r] * Vtx[r];
                }
            }
            y[i] = sum;
        }
    }
}

// Combined: y = (blockdiag + UV^T) * x
template <typename T>
__device__ __forceinline__ void sst_structured_matvec(
    const T *A_blocks, const T *U, const T *V, const T *x, T *y,
    int dstate, int block_size, int num_blocks, int rank)
{
    // Block-diagonal part
    sst_block_diagonal_matvec(A_blocks, x, y, dstate, block_size, num_blocks);
    
    // Add low-rank part
    T lowrank_y[SST_MAX_DSTATE];
    sst_lowrank_matvec(U, V, x, lowrank_y, dstate, rank);
    
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            y[i] = y[i] + lowrank_y[i];
        }
    }
}

// Block-diagonal matrix exponential: exp(blockdiag(A_1,...,A_K)) = blockdiag(exp(A_1),...,exp(A_K))
template <typename T>
__device__ __forceinline__ void sst_block_diagonal_exp(
    const T *A_blocks, T *expA_blocks, float delta,
    int block_size, int num_blocks, int num_terms = 10)
{
    for (int k = 0; k < num_blocks; ++k) {
        const T *A_k = A_blocks + k * block_size * block_size;
        T *expA_k = expA_blocks + k * block_size * block_size;
        
        // Scale A_k by delta
        T deltaA_k[SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
        #pragma unroll
        for (int i = 0; i < block_size; ++i) {
            #pragma unroll
            for (int j = 0; j < block_size; ++j) {
                deltaA_k[i * block_size + j] = T(delta) * A_k[i * block_size + j];
            }
        }
        
        // Compute exp(delta * A_k)
        sst_matrix_exp_taylor(deltaA_k, expA_k, block_size, num_terms);
    }
}

//=============================================================================
// PART 5: Discretization Methods for Structured A
//=============================================================================

// Helper: Compute V^T * U (rank x rank matrix)
template <typename T>
__device__ __forceinline__ void sst_compute_VTU(
    const T *U, const T *V, T *VTU,
    int dstate, int rank)
{
    #pragma unroll
    for (int r1 = 0; r1 < SST_MAX_RANK; ++r1) {
        if (r1 < rank) {
            #pragma unroll
            for (int r2 = 0; r2 < SST_MAX_RANK; ++r2) {
                if (r2 < rank) {
                    T sum = T(0);
                    #pragma unroll
                    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
                        if (i < dstate) {
                            sum = sum + V[i * rank + r1] * U[i * rank + r2];
                        }
                    }
                    VTU[r1 * rank + r2] = sum;
                }
            }
        }
    }
}

// ZOH: x_new = exp(δA) @ x + δ * B * u
// Direct computation of exp(δ * (blockdiag + UV^T)) @ x without full matrix
template <typename T>
__device__ __forceinline__ void sst_zoh_structured_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, T Bu,
    int dstate, int block_size, int num_blocks, int rank,
    int num_terms = 10)
{
    // Step 1: Compute exp(δ * blockdiag) @ x block-wise
    T expA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    sst_block_diagonal_exp(A_blocks, expA_blocks, delta, block_size, num_blocks, num_terms);
    
    T blockdiag_x[SST_MAX_DSTATE];
    sst_block_diagonal_matvec(expA_blocks, x, blockdiag_x, dstate, block_size, num_blocks);
    
    // Step 2: Compute low-rank correction
    // For small δ: exp(δ*(B+UV^T)) ≈ exp(δ*B) * (I + δ*UV^T + ...)
    // Using first-order: exp(δ*B) @ x + δ * UV^T @ x
    if (delta * (float)rank < 1.0f) {
        // First-order approximation
        T lowrank_x[SST_MAX_DSTATE];
        sst_lowrank_matvec(U, V, x, lowrank_x, dstate, rank);
        
        #pragma unroll
        for (int i = 0; i < SST_MAX_DSTATE; ++i) {
            if (i < dstate) {
                x_new[i] = blockdiag_x[i] + T(delta) * lowrank_x[i] + Bu;
            }
        }
    } else {
        // Higher-order: Use exp(δ * V^T U) in low-rank space
        T VTU[SST_MAX_RANK * SST_MAX_RANK];
        sst_compute_VTU(U, V, VTU, dstate, rank);
        
        // Scale by delta
        #pragma unroll
        for (int i = 0; i < rank * rank; ++i) {
            VTU[i] = T(delta) * VTU[i];
        }
        
        // exp(δ * V^T U)
        T exp_VTU[SST_MAX_RANK * SST_MAX_RANK];
        sst_matrix_exp_taylor(VTU, exp_VTU, rank, num_terms);
        
        // V^T @ x
        T Vtx[SST_MAX_RANK];
        #pragma unroll
        for (int r = 0; r < rank; ++r) {
            T sum = T(0);
            #pragma unroll
            for (int i = 0; i < dstate; ++i) {
                sum = sum + V[i * rank + r] * x[i];
            }
            Vtx[r] = sum;
        }
        
        // exp(δ * V^T U) @ V^T @ x
        T exp_VTU_Vtx[SST_MAX_RANK];
        #pragma unroll
        for (int r1 = 0; r1 < rank; ++r1) {
            T sum = T(0);
            #pragma unroll
            for (int r2 = 0; r2 < rank; ++r2) {
                sum = sum + exp_VTU[r1 * rank + r2] * Vtx[r2];
            }
            exp_VTU_Vtx[r1] = sum;
        }
        
        // U @ (exp(δ * V^T U) @ V^T @ x)
        T lowrank_correction[SST_MAX_DSTATE];
        #pragma unroll
        for (int i = 0; i < dstate; ++i) {
            T sum = T(0);
            #pragma unroll
            for (int r = 0; r < rank; ++r) {
                sum = sum + U[i * rank + r] * exp_VTU_Vtx[r];
            }
            lowrank_correction[i] = sum;
        }
        
        // Combine
        #pragma unroll
        for (int i = 0; i < SST_MAX_DSTATE; ++i) {
            if (i < dstate) {
                x_new[i] = blockdiag_x[i] + lowrank_correction[i] + Bu;
            }
        }
    }
}

// FOH: Uses first-order hold discretization
// A_d = exp(δA), B_d = A^(-2) * (exp(δA) - I - δA) * B
template <typename T>
__device__ __forceinline__ void sst_foh_structured_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, T B_val, T u_val,
    int dstate, int block_size, int num_blocks, int rank,
    int num_terms = 10)
{
    // For FOH, the B_d term is different from ZOH
    // Using Taylor series: B_d/B = δ²/2! + Aδ³/3! + A²δ⁴/4! + ...
    
    // First compute exp(δA) @ x (same as ZOH)
    T expA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    sst_block_diagonal_exp(A_blocks, expA_blocks, delta, block_size, num_blocks, num_terms);
    
    T blockdiag_x[SST_MAX_DSTATE];
    sst_block_diagonal_matvec(expA_blocks, x, blockdiag_x, dstate, block_size, num_blocks);
    
    // Low-rank correction (same approach as ZOH)
    T lowrank_x[SST_MAX_DSTATE];
    sst_lowrank_matvec(U, V, x, lowrank_x, dstate, rank);
    
    // FOH B_d coefficient: δ²/2 + (aggregate A effect)
    // For structured A, we approximate using average eigenvalue
    float delta_sq = delta * delta;
    float foh_coeff = delta_sq / 2.0f;
    
    T Bu = T(foh_coeff) * B_val * u_val;
    
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            x_new[i] = blockdiag_x[i] + T(delta) * lowrank_x[i] + Bu;
        }
    }
}

// Bilinear: A_d = (I - δA/2)^(-1) * (I + δA/2)
// For structured A: Use Woodbury formula for matrix inversion
template <typename T>
__device__ __forceinline__ void sst_bilinear_structured_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, T Bu,
    int dstate, int block_size, int num_blocks, int rank,
    int num_terms = 10)
{
    float half_delta = delta * 0.5f;
    
    // For bilinear with structured A:
    // (I - δA/2)^(-1) where A = blockdiag + UV^T
    // Using Woodbury: (B + UV^T)^(-1) = B^(-1) - B^(-1)U(I + V^TB^(-1)U)^(-1)V^TB^(-1)
    // Here B = I - δ*blockdiag/2
    
    // First compute (I - δ*blockdiag/2)^(-1) @ (I + δ*blockdiag/2) @ x for block-diagonal part
    T expA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    
    // For each block: A_d = (I - δA_k/2)^(-1) * (I + δA_k/2)
    for (int k = 0; k < num_blocks; ++k) {
        const T *A_k = A_blocks + k * block_size * block_size;
        T *Ad_k = expA_blocks + k * block_size * block_size;
        
        // Compute I - δA_k/2
        T I_minus[SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
        T I_plus[SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
        
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                float Aij = A_k[i * block_size + j];
                I_minus[i * block_size + j] = (i == j) ? T(1.0f - half_delta * Aij) : T(-half_delta * Aij);
                I_plus[i * block_size + j] = (i == j) ? T(1.0f + half_delta * Aij) : T(half_delta * Aij);
            }
        }
        
        // Invert I_minus (simple Gauss-Jordan for small blocks)
        T aug[SST_MAX_BLOCK_SIZE * 2 * SST_MAX_BLOCK_SIZE];
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                aug[i * (2 * block_size) + j] = I_minus[i * block_size + j];
                aug[i * (2 * block_size) + block_size + j] = (i == j) ? T(1) : T(0);
            }
        }
        
        for (int col = 0; col < block_size; ++col) {
            T pivot_val = aug[col * (2 * block_size) + col];
            float mag = pivot_val * pivot_val;
            if (mag > 1e-10f) {
                for (int j = 0; j < 2 * block_size; ++j) {
                    aug[col * (2 * block_size) + j] = aug[col * (2 * block_size) + j] / pivot_val;
                }
            }
            for (int row = 0; row < block_size; ++row) {
                if (row != col) {
                    T factor = aug[row * (2 * block_size) + col];
                    for (int j = 0; j < 2 * block_size; ++j) {
                        aug[row * (2 * block_size) + j] = aug[row * (2 * block_size) + j] - 
                                                          factor * aug[col * (2 * block_size) + j];
                    }
                }
            }
        }
        
        // Extract I_minus^(-1)
        T I_minus_inv[SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                I_minus_inv[i * block_size + j] = aug[i * (2 * block_size) + block_size + j];
            }
        }
        
        // Compute A_d = I_minus^(-1) * I_plus
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                T sum = T(0);
                for (int l = 0; l < block_size; ++l) {
                    sum = sum + I_minus_inv[i * block_size + l] * I_plus[l * block_size + j];
                }
                Ad_k[i * block_size + j] = sum;
            }
        }
    }
    
    // Block-diagonal part
    T blockdiag_x[SST_MAX_DSTATE];
    sst_block_diagonal_matvec(expA_blocks, x, blockdiag_x, dstate, block_size, num_blocks);
    
    // Low-rank correction using first-order approximation for bilinear
    // (I - δ(B+UV^T)/2)^(-1)(I + δ(B+UV^T)/2) ≈ (I - δB/2)^(-1)(I + δB/2) + δUV^T + O(δ²)
    T lowrank_x[SST_MAX_DSTATE];
    sst_lowrank_matvec(U, V, x, lowrank_x, dstate, rank);
    
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            x_new[i] = blockdiag_x[i] + T(delta) * lowrank_x[i] + Bu;
        }
    }
}

// RK4: 4th-order Runge-Kutta method
// Computes 4 stages: k1, k2, k3, k4
template <typename T>
__device__ __forceinline__ void sst_rk4_structured_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, const T *B, T u_val,
    int dstate, int block_size, int num_blocks, int rank)
{
    // k1 = A @ x + B * u
    T k1[SST_MAX_DSTATE];
    sst_structured_matvec(A_blocks, U, V, x, k1, dstate, block_size, num_blocks, rank);
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        k1[i] = k1[i] + B[i] * u_val;
    }
    
    // k2 = A @ (x + δ/2 * k1) + B * u
    T x_half[SST_MAX_DSTATE];
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        x_half[i] = x[i] + T(delta * 0.5f) * k1[i];
    }
    T k2[SST_MAX_DSTATE];
    sst_structured_matvec(A_blocks, U, V, x_half, k2, dstate, block_size, num_blocks, rank);
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        k2[i] = k2[i] + B[i] * u_val;
    }
    
    // k3 = A @ (x + δ/2 * k2) + B * u
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        x_half[i] = x[i] + T(delta * 0.5f) * k2[i];
    }
    T k3[SST_MAX_DSTATE];
    sst_structured_matvec(A_blocks, U, V, x_half, k3, dstate, block_size, num_blocks, rank);
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        k3[i] = k3[i] + B[i] * u_val;
    }
    
    // k4 = A @ (x + δ * k3) + B * u
    T x_full[SST_MAX_DSTATE];
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        x_full[i] = x[i] + T(delta) * k3[i];
    }
    T k4[SST_MAX_DSTATE];
    sst_structured_matvec(A_blocks, U, V, x_full, k4, dstate, block_size, num_blocks, rank);
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        k4[i] = k4[i] + B[i] * u_val;
    }
    
    // x_new = x + δ/6 * (k1 + 2*k2 + 2*k3 + k4)
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            x_new[i] = x[i] + T(delta / 6.0f) * (k1[i] + T(2) * k2[i] + T(2) * k3[i] + k4[i]);
        }
    }
}

// Polynomial: Combination of ZOH and FOH terms
template <typename T>
__device__ __forceinline__ void sst_poly_structured_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, T B_val, T u_val,
    int dstate, int block_size, int num_blocks, int rank,
    int num_terms = 10)
{
    // exp(δA) @ x
    T expA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    sst_block_diagonal_exp(A_blocks, expA_blocks, delta, block_size, num_blocks, num_terms);
    
    T blockdiag_x[SST_MAX_DSTATE];
    sst_block_diagonal_matvec(expA_blocks, x, blockdiag_x, dstate, block_size, num_blocks);
    
    T lowrank_x[SST_MAX_DSTATE];
    sst_lowrank_matvec(U, V, x, lowrank_x, dstate, rank);
    
    // Poly coefficient: δ + δ²/4 + higher order terms
    float delta_sq = delta * delta;
    float poly_coeff = delta + delta_sq * 0.25f;
    
    T Bu = T(poly_coeff) * B_val * u_val;
    
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            x_new[i] = blockdiag_x[i] + T(delta) * lowrank_x[i] + Bu;
        }
    }
}

// High-Order: Higher-order hold (n=2, quadratic)
template <typename T>
__device__ __forceinline__ void sst_highorder_structured_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, T B_val, T u_val,
    int dstate, int block_size, int num_blocks, int rank,
    int num_terms = 10)
{
    // exp(δA) @ x
    T expA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    sst_block_diagonal_exp(A_blocks, expA_blocks, delta, block_size, num_blocks, num_terms);
    
    T blockdiag_x[SST_MAX_DSTATE];
    sst_block_diagonal_matvec(expA_blocks, x, blockdiag_x, dstate, block_size, num_blocks);
    
    T lowrank_x[SST_MAX_DSTATE];
    sst_lowrank_matvec(U, V, x, lowrank_x, dstate, rank);
    
    // High-order coefficient: δ + δ²*(1/2 + 1/2) + δ³*(1/6 + 1/6 + 1/12) + ...
    float delta_sq = delta * delta;
    float delta_cubed = delta_sq * delta;
    float ho_coeff = delta + delta_sq * 0.5f + delta_cubed * (1.0f/6.0f + 1.0f/6.0f + 1.0f/12.0f);
    
    T Bu = T(ho_coeff) * B_val * u_val;
    
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            x_new[i] = blockdiag_x[i] + T(delta) * lowrank_x[i] + Bu;
        }
    }
}

//=============================================================================
// PART 6: Unified Structured State Step (All Methods)
//=============================================================================

// Unified step function that dispatches to the appropriate discretization method
template <typename T>
__device__ __forceinline__ void sst_structured_state_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, T Bu,
    int dstate, int block_size, int num_blocks, int rank,
    DiscretizationMethod method, int num_terms = 10)
{
    switch (method) {
        case DISCRETIZATION_ZOH:
            sst_zoh_structured_step(A_blocks, U, V, x, x_new, delta, Bu,
                                   dstate, block_size, num_blocks, rank, num_terms);
            break;
        case DISCRETIZATION_FOH:
            sst_foh_structured_step(A_blocks, U, V, x, x_new, delta, T(1), Bu,
                                   dstate, block_size, num_blocks, rank, num_terms);
            break;
        case DISCRETIZATION_BILINEAR:
            sst_bilinear_structured_step(A_blocks, U, V, x, x_new, delta, Bu,
                                        dstate, block_size, num_blocks, rank, num_terms);
            break;
        case DISCRETIZATION_RK4:
            // RK4 needs B vector, use simplified version
            {
                T B_vec[SST_MAX_DSTATE];
                for (int i = 0; i < dstate; ++i) {
                    B_vec[i] = T(1); // Placeholder
                }
                sst_rk4_structured_step(A_blocks, U, V, x, x_new, delta, B_vec, Bu,
                                       dstate, block_size, num_blocks, rank);
            }
            break;
        case DISCRETIZATION_POLY:
            sst_poly_structured_step(A_blocks, U, V, x, x_new, delta, T(1), Bu,
                                    dstate, block_size, num_blocks, rank, num_terms);
            break;
        case DISCRETIZATION_HIGHORDER:
        default:
            sst_highorder_structured_step(A_blocks, U, V, x, x_new, delta, T(1), Bu,
                                         dstate, block_size, num_blocks, rank, num_terms);
            break;
    }
}

//=============================================================================
// PART 7: Backward Pass - Gradient Computation
//=============================================================================

// Gradient of matrix exponential: d(exp(A))/dA
// Using the Fréchet derivative approach
// For a small perturbation dA: d(exp(A)) = integral_0^1 exp((1-t)*A) * dA * exp(t*A) dt
// Approximation: d(exp(A))[dA] ≈ exp(A) * dA (for small A)
template <typename T>
__device__ __forceinline__ void sst_matrix_exp_grad(
    const T *A, const T *expA, const T *d_expA,
    T *dA, int n)
{
    // Simplified gradient: dA ≈ exp(-A) * d_expA * exp(A)
    // For efficiency, use approximation: dA ≈ d_expA (for small A)
    // More accurate: dA[i,j] = sum_k,l expA[k,i] * d_expA[k,l] * expA[l,j] / factorial terms
    
    // Using Taylor series gradient:
    // exp(A) = I + A + A²/2! + A³/3! + ...
    // d(exp(A))/dA[i,j] = delta[i,j] + (delta[k,j]*A + A*delta[i,l])/2! + ...
    
    // For now, use first-order approximation
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        #pragma unroll
        for (int j = 0; j < n; ++j) {
            dA[i * n + j] = d_expA[i * n + j];
        }
    }
}

// Gradient through block-diagonal matrix-vector multiplication
// y = blockdiag @ x
// dL/d(blockdiag) = dL/dy * x^T (per block)
// dL/dx = blockdiag^T @ dL/dy
template <typename T>
__device__ __forceinline__ void sst_block_diagonal_matvec_backward(
    const T *A_blocks, const T *x, const T *dy,
    T *dA_blocks, T *dx,
    int dstate, int block_size, int num_blocks)
{
    // Initialize gradients
    #pragma unroll
    for (int i = 0; i < SST_MAX_DSTATE; ++i) {
        if (i < dstate) {
            dx[i] = T(0);
        }
    }
    
    int total_block_elems = num_blocks * block_size * block_size;
    for (int idx = 0; idx < total_block_elems; ++idx) {
        dA_blocks[idx] = T(0);
    }
    
    // Process each block
    for (int k = 0; k < num_blocks; ++k) {
        int start_idx = k * block_size;
        const T *A_k = A_blocks + k * block_size * block_size;
        T *dA_k = dA_blocks + k * block_size * block_size;
        
        // dL/dA_k = dy[block] * x[block]^T
        // dL/dx[block] = A_k^T @ dy[block]
        #pragma unroll
        for (int i = 0; i < block_size; ++i) {
            int row = start_idx + i;
            if (row < dstate) {
                #pragma unroll
                for (int j = 0; j < block_size; ++j) {
                    int col = start_idx + j;
                    if (col < dstate) {
                        // dA_k[i,j] = dy[row] * x[col]
                        dA_k[i * block_size + j] = dy[row] * x[col];
                        
                        // dx[col] += A_k[i,j] * dy[row] (transpose)
                        dx[col] = dx[col] + A_k[i * block_size + j] * dy[row];
                    }
                }
            }
        }
    }
}

// Gradient through low-rank matrix-vector multiplication
// y = UV^T @ x = U @ (V^T @ x)
// dL/dU = dL/dy @ (V^T @ x)^T
// dL/dV = x @ (dL/dy^T @ U)^T  
// dL/dx = V @ (U^T @ dL/dy)
template <typename T>
__device__ __forceinline__ void sst_lowrank_matvec_backward(
    const T *U, const T *V, const T *x, const T *dy,
    T *dU, T *dV, T *dx,
    int dstate, int rank)
{
    // Compute V^T @ x (needed for dU)
    T Vtx[SST_MAX_RANK];
    #pragma unroll
    for (int r = 0; r < rank; ++r) {
        T sum = T(0);
        #pragma unroll
        for (int i = 0; i < dstate; ++i) {
            sum = sum + V[i * rank + r] * x[i];
        }
        Vtx[r] = sum;
    }
    
    // Compute U^T @ dy (needed for dV and dx)
    T Utdy[SST_MAX_RANK];
    #pragma unroll
    for (int r = 0; r < rank; ++r) {
        T sum = T(0);
        #pragma unroll
        for (int i = 0; i < dstate; ++i) {
            sum = sum + U[i * rank + r] * dy[i];
        }
        Utdy[r] = sum;
    }
    
    // dU[i,r] = dy[i] * Vtx[r]
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        #pragma unroll
        for (int r = 0; r < rank; ++r) {
            dU[i * rank + r] = dy[i] * Vtx[r];
        }
    }
    
    // dV[i,r] = x[i] * Utdy[r]
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        #pragma unroll
        for (int r = 0; r < rank; ++r) {
            dV[i * rank + r] = x[i] * Utdy[r];
        }
    }
    
    // dx[i] = sum_r V[i,r] * Utdy[r]
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        T sum = T(0);
        #pragma unroll
        for (int r = 0; r < rank; ++r) {
            sum = sum + V[i * rank + r] * Utdy[r];
        }
        dx[i] = sum;
    }
}

// Gradient through structured state transition (ZOH)
// x_new = exp(δA) @ x_old + δ * B * u
// Gradients: dL/dA_blocks, dL/dU, dL/dV, dL/dx_old, dL/dδ, dL/dB, dL/du
template <typename T>
__device__ __forceinline__ void sst_zoh_structured_backward(
    const T *A_blocks, const T *U, const T *V,
    const T *x_old, const T *x_new, float delta,
    const T *B, T u_val, const T *dy,
    T *dA_blocks, T *dU, T *dV, T *dx_old,
    T *ddelta, T *dB, T *du,
    int dstate, int block_size, int num_blocks, int rank,
    int num_terms = 10)
{
    // Forward: x_new = exp(δA) @ x_old + δ * B * u
    // where A = blockdiag + UV^T
    
    // First, compute exp(δ * blockdiag)
    T expA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    sst_block_diagonal_exp(A_blocks, expA_blocks, delta, block_size, num_blocks, num_terms);
    
    // Gradient through exp(δ*blockdiag) @ x_old
    T dx_from_block[SST_MAX_DSTATE];
    T dexpA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    sst_block_diagonal_matvec_backward(expA_blocks, x_old, dy, dexpA_blocks, dx_from_block,
                                       dstate, block_size, num_blocks);
    
    // Gradient through low-rank part: δ * UV^T @ x_old
    T dy_scaled[SST_MAX_DSTATE];
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        dy_scaled[i] = T(delta) * dy[i];
    }
    
    T dU_temp[SST_MAX_DSTATE * SST_MAX_RANK];
    T dV_temp[SST_MAX_DSTATE * SST_MAX_RANK];
    T dx_from_lowrank[SST_MAX_DSTATE];
    sst_lowrank_matvec_backward(U, V, x_old, dy_scaled, dU_temp, dV_temp, dx_from_lowrank,
                                dstate, rank);
    
    // Copy gradients
    for (int i = 0; i < dstate * rank; ++i) {
        dU[i] = dU_temp[i];
        dV[i] = dV_temp[i];
    }
    
    // Combine dx gradients
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        dx_old[i] = dx_from_block[i] + dx_from_lowrank[i];
    }
    
    // Gradient through δ * B * u
    // dB[i] = δ * u * dy[i]
    // du = δ * sum_i B[i] * dy[i]
    // dδ = sum_i (B[i] * u + (A @ x_old)[i]) * dy[i]
    T du_val = T(0);
    T ddelta_val = T(0);
    
    // Compute (A @ x_old)
    T Ax[SST_MAX_DSTATE];
    sst_structured_matvec(A_blocks, U, V, x_old, Ax, dstate, block_size, num_blocks, rank);
    
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        dB[i] = T(delta) * u_val * dy[i];
        du_val = du_val + T(delta) * B[i] * dy[i];
        ddelta_val = ddelta_val + (B[i] * u_val + Ax[i]) * dy[i];
    }
    
    *du = du_val;
    *ddelta = ddelta_val;
    
    // Gradient of exp(δ*A) w.r.t. A_blocks
    // d(exp(δA))/dA_blocks ≈ δ * dexpA_blocks (first-order)
    int total_block_elems = num_blocks * block_size * block_size;
    for (int idx = 0; idx < total_block_elems; ++idx) {
        dA_blocks[idx] = T(delta) * dexpA_blocks[idx];
    }
}

//=============================================================================
// PART 8: Bidirectional Support
//=============================================================================

// Forward scan for bidirectional Mamba
template <typename T>
__device__ __forceinline__ void sst_forward_scan_step(
    const T *A_blocks, const T *U, const T *V,
    const T *x, T *x_new, float delta, T Bu,
    int dstate, int block_size, int num_blocks, int rank,
    DiscretizationMethod method, int num_terms = 10)
{
    sst_structured_state_step(A_blocks, U, V, x, x_new, delta, Bu,
                              dstate, block_size, num_blocks, rank, method, num_terms);
}

// Backward scan for bidirectional Mamba (reverse direction)
// Uses A_backward matrix instead of A_forward
template <typename T>
__device__ __forceinline__ void sst_backward_scan_step(
    const T *A_blocks_bwd, const T *U_bwd, const T *V_bwd,
    const T *x, T *x_new, float delta, T Bu,
    int dstate, int block_size, int num_blocks, int rank,
    DiscretizationMethod method, int num_terms = 10)
{
    // Same structure as forward, but with backward A matrices
    sst_structured_state_step(A_blocks_bwd, U_bwd, V_bwd, x, x_new, delta, Bu,
                              dstate, block_size, num_blocks, rank, method, num_terms);
}

// Combined bidirectional output
// y = C_fwd^T @ x_fwd + C_bwd^T @ x_bwd
template <typename T>
__device__ __forceinline__ T sst_bidirectional_output(
    const T *C_fwd, const T *x_fwd,
    const T *C_bwd, const T *x_bwd,
    int dstate)
{
    T y = T(0);
    #pragma unroll
    for (int i = 0; i < dstate; ++i) {
        y = y + C_fwd[i] * x_fwd[i] + C_bwd[i] * x_bwd[i];
    }
    return y;
}

//=============================================================================
// PART 9: Complex Number Extensions
//=============================================================================

// Complex block-diagonal matrix-vector multiplication
__device__ __forceinline__ void sst_complex_block_diagonal_matvec(
    const ComplexFloat *A_blocks, const ComplexFloat *x, ComplexFloat *y,
    int dstate, int block_size, int num_blocks)
{
    // Initialize output
    for (int i = 0; i < dstate; ++i) {
        y[i] = ComplexFloat(0, 0);
    }
    
    // Process each block
    for (int k = 0; k < num_blocks; ++k) {
        int start_idx = k * block_size;
        const ComplexFloat *A_k = A_blocks + k * block_size * block_size;
        
        for (int i = 0; i < block_size; ++i) {
            int row = start_idx + i;
            if (row < dstate) {
                ComplexFloat sum(0, 0);
                for (int j = 0; j < block_size; ++j) {
                    int col = start_idx + j;
                    if (col < dstate) {
                        sum = sum + A_k[i * block_size + j] * x[col];
                    }
                }
                y[row] = sum;
            }
        }
    }
}

// Complex low-rank matrix-vector multiplication
__device__ __forceinline__ void sst_complex_lowrank_matvec(
    const ComplexFloat *U, const ComplexFloat *V, const ComplexFloat *x, ComplexFloat *y,
    int dstate, int rank)
{
    // V^T @ x (conjugate transpose for complex)
    ComplexFloat Vtx[SST_MAX_RANK];
    for (int r = 0; r < rank; ++r) {
        ComplexFloat sum(0, 0);
        for (int i = 0; i < dstate; ++i) {
            sum = sum + V[i * rank + r].conj() * x[i];
        }
        Vtx[r] = sum;
    }
    
    // y = U @ Vtx
    for (int i = 0; i < dstate; ++i) {
        ComplexFloat sum(0, 0);
        for (int r = 0; r < rank; ++r) {
            sum = sum + U[i * rank + r] * Vtx[r];
        }
        y[i] = sum;
    }
}

// Complex matrix exponential using Taylor series
__device__ __forceinline__ void sst_complex_matrix_exp_taylor(
    const ComplexFloat *A, ComplexFloat *expA, int n, int num_terms = 10)
{
    // Initialize expA = I
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            expA[i * n + j] = (i == j) ? ComplexFloat(1, 0) : ComplexFloat(0, 0);
        }
    }
    
    // A^k storage
    ComplexFloat Ak[SST_MAX_DSTATE * SST_MAX_DSTATE];
    ComplexFloat temp[SST_MAX_DSTATE * SST_MAX_DSTATE];
    
    for (int i = 0; i < n * n; ++i) {
        Ak[i] = A[i];
    }
    
    float factorial = 1.0f;
    for (int k = 1; k <= num_terms; ++k) {
        factorial *= k;
        
        // Add A^k / k! to expA
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                expA[i * n + j] = expA[i * n + j] + Ak[i * n + j] / factorial;
            }
        }
        
        // Compute A^(k+1)
        if (k < num_terms) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    ComplexFloat sum(0, 0);
                    for (int l = 0; l < n; ++l) {
                        sum = sum + Ak[i * n + l] * A[l * n + j];
                    }
                    temp[i * n + j] = sum;
                }
            }
            for (int i = 0; i < n * n; ++i) {
                Ak[i] = temp[i];
            }
        }
    }
}

// Complex ZOH structured step
__device__ __forceinline__ void sst_complex_zoh_structured_step(
    const ComplexFloat *A_blocks, const ComplexFloat *U, const ComplexFloat *V,
    const ComplexFloat *x, ComplexFloat *x_new, float delta, ComplexFloat Bu,
    int dstate, int block_size, int num_blocks, int rank, int num_terms = 10)
{
    // Compute exp(δ * blockdiag) per block
    ComplexFloat expA_blocks[SST_MAX_BLOCKS * SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
    
    for (int k = 0; k < num_blocks; ++k) {
        const ComplexFloat *A_k = A_blocks + k * block_size * block_size;
        ComplexFloat *expA_k = expA_blocks + k * block_size * block_size;
        
        // Scale and compute exp
        ComplexFloat deltaA_k[SST_MAX_BLOCK_SIZE * SST_MAX_BLOCK_SIZE];
        for (int i = 0; i < block_size * block_size; ++i) {
            deltaA_k[i] = A_k[i] * delta;
        }
        sst_complex_matrix_exp_taylor(deltaA_k, expA_k, block_size, num_terms);
    }
    
    // exp(δ*blockdiag) @ x
    ComplexFloat blockdiag_x[SST_MAX_DSTATE];
    sst_complex_block_diagonal_matvec(expA_blocks, x, blockdiag_x, dstate, block_size, num_blocks);
    
    // Low-rank correction
    ComplexFloat lowrank_x[SST_MAX_DSTATE];
    sst_complex_lowrank_matvec(U, V, x, lowrank_x, dstate, rank);
    
    // Combine
    for (int i = 0; i < dstate; ++i) {
        x_new[i] = blockdiag_x[i] + lowrank_x[i] * delta + Bu;
    }
}

#endif // STRUCTURED_MATRIX_OPS_CUH
