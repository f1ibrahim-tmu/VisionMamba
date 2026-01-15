/******************************************************************************
 * Feature-SST: Bidirectional Mamba - Structured State Transitions
 * 
 * SST = Structured State Transitions: A = blockdiag(A_1, ..., A_K) + UV^T
 * Bidirectional scanning with separate forward and backward A matrices.
 * Supports all 6 discretization methods and complex numbers.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>

#ifndef USE_ROCM
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "static_switch.h"
#include "discretization_kernels.cuh"
#include "matrix_ops.cuh"

// ============================================================================
// Bidirectional Mamba Kernel
// Processes both forward and backward directions in a single kernel launch
// ============================================================================

template <int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
          bool kIsVariableB_, bool kIsVariableC_,
          bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_bidirectional_kernel_traits
{
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
                                         !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2,
                                               !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
                                           !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                   sizeof(typename BlockLoadVecT::TempStorage),
                                                   (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                   (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                   sizeof(typename BlockStoreT::TempStorage),
                                                   sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bidirectional_fwd_kernel(SSMParamsBidirectional params)
{
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    extern __shared__ char smem_[];
    auto &smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage &>(smem_);
    auto &smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage &>(smem_);
    auto &smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage &>(smem_);
    auto &smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage *>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride + dim_id * kNRows * params.delta_d_stride;

    // Forward and backward direction outputs
    input_t *out_fwd = reinterpret_cast<input_t *>(params.out_fwd_ptr) + batch_id * params.out_batch_stride + dim_id * kNRows * params.out_d_stride;
    input_t *out_bwd = reinterpret_cast<input_t *>(params.out_bwd_ptr) + batch_id * params.out_batch_stride + dim_id * kNRows * params.out_d_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride + dim_id * kNRows * params.out_d_stride;

    // Forward direction A (standard or structured)
    float *A_blocks_fwd = nullptr;
    float *A_U_fwd = nullptr;
    float *A_V_fwd = nullptr;
    
    if (params.use_structured_A)
    {
        A_blocks_fwd = reinterpret_cast<float *>(params.A_blocks_ptr) + dim_id * params.A_block_stride;
        A_U_fwd = reinterpret_cast<float *>(params.A_U_ptr) + dim_id * params.A_U_stride;
        A_V_fwd = reinterpret_cast<float *>(params.A_V_ptr) + dim_id * params.A_V_stride;
    }

    // Backward direction A (separate or same as forward)
    float *A_blocks_bwd = nullptr;
    float *A_U_bwd = nullptr;
    float *A_V_bwd = nullptr;
    
    if (params.use_structured_A_b && params.A_b_blocks_ptr != nullptr)
    {
        A_blocks_bwd = reinterpret_cast<float *>(params.A_b_blocks_ptr) + dim_id * params.A_b_block_stride;
        A_U_bwd = reinterpret_cast<float *>(params.A_b_U_ptr) + dim_id * params.A_b_U_stride;
        A_V_bwd = reinterpret_cast<float *>(params.A_b_V_ptr) + dim_id * params.A_b_V_stride;
    }
    else if (params.use_structured_A)
    {
        // Use same A for both directions
        A_blocks_bwd = A_blocks_fwd;
        A_U_bwd = A_U_fwd;
        A_V_bwd = A_V_fwd;
    }

    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    
    scan_t *x_fwd = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;
    scan_t *x_bwd = params.x_bwd_ptr != nullptr 
        ? reinterpret_cast<scan_t *>(params.x_bwd_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate
        : nullptr;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr)
    {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr)
    {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    constexpr int kChunkSize = kNThreads * kNItems;

    // Bidirectional processing with structured A
    if (params.use_structured_A || params.use_structured_A_b)
    {
        // Load structured components into shared memory
        __shared__ float A_blocks_fwd_shared[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE * 16];
        __shared__ float A_U_fwd_shared[MAX_DSTATE * MAX_LOW_RANK];
        __shared__ float A_V_fwd_shared[MAX_DSTATE * MAX_LOW_RANK];
        __shared__ float A_blocks_bwd_shared[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE * 16];
        __shared__ float A_U_bwd_shared[MAX_DSTATE * MAX_LOW_RANK];
        __shared__ float A_V_bwd_shared[MAX_DSTATE * MAX_LOW_RANK];
        
        __shared__ float x_fwd_state[MAX_DSTATE];
        __shared__ float x_bwd_state[MAX_DSTATE];
        __shared__ float x_new_fwd[MAX_DSTATE];
        __shared__ float x_new_bwd[MAX_DSTATE];

        // Load forward A components
        if (A_blocks_fwd != nullptr)
        {
            int total_block_elements = params.num_blocks * params.block_size * params.block_size;
            for (int idx = threadIdx.x; idx < total_block_elements; idx += blockDim.x)
            {
                A_blocks_fwd_shared[idx] = A_blocks_fwd[idx];
            }
            for (int idx = threadIdx.x; idx < params.dstate * params.low_rank_rank; idx += blockDim.x)
            {
                A_U_fwd_shared[idx] = A_U_fwd[idx];
                A_V_fwd_shared[idx] = A_V_fwd[idx];
            }
        }

        // Load backward A components (if different from forward)
        if (A_blocks_bwd != nullptr && A_blocks_bwd != A_blocks_fwd)
        {
            int total_block_elements = params.num_blocks_b * params.block_size_b * params.block_size_b;
            for (int idx = threadIdx.x; idx < total_block_elements; idx += blockDim.x)
            {
                A_blocks_bwd_shared[idx] = A_blocks_bwd[idx];
            }
            for (int idx = threadIdx.x; idx < params.dstate * params.low_rank_rank_b; idx += blockDim.x)
            {
                A_U_bwd_shared[idx] = A_U_bwd[idx];
                A_V_bwd_shared[idx] = A_V_bwd[idx];
            }
        }
        else if (A_blocks_bwd == A_blocks_fwd)
        {
            // Copy forward to backward shared memory
            int total_block_elements = params.num_blocks * params.block_size * params.block_size;
            for (int idx = threadIdx.x; idx < total_block_elements; idx += blockDim.x)
            {
                A_blocks_bwd_shared[idx] = A_blocks_fwd_shared[idx];
            }
            for (int idx = threadIdx.x; idx < params.dstate * params.low_rank_rank; idx += blockDim.x)
            {
                A_U_bwd_shared[idx] = A_U_fwd_shared[idx];
                A_V_bwd_shared[idx] = A_V_fwd_shared[idx];
            }
        }

        // Initialize states
        if (threadIdx.x < params.dstate)
        {
            x_fwd_state[threadIdx.x] = 0.0f;
            x_bwd_state[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Process chunks
        for (int chunk = 0; chunk < params.n_chunks; ++chunk)
        {
            int chunk_bwd = params.n_chunks - 1 - chunk;  // Reverse chunk index for backward

            input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if constexpr (!kDirectIO)
                {
                    if (r > 0) { __syncthreads(); }
                }
                load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
                if constexpr (!kDirectIO) { __syncthreads(); }
                load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            }

            float delta_vals[kNRows][kNItems], out_vals_fwd[kNRows][kNItems], out_vals_bwd[kNRows][kNItems];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    float u_val = float(u_vals[r][i]);
                    delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                    if (params.delta_softplus)
                    {
                        delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                    }
                    out_vals_fwd[r][i] = D_val[r] * u_val;
                    out_vals_bwd[r][i] = D_val[r] * u_val;
                }
            }

            // Process time steps
            int chunk_start = chunk * kChunkSize;
            int chunk_end = (chunk_start + kChunkSize < params.seqlen) ? (chunk_start + kChunkSize) : params.seqlen;
            int chunk_len = chunk_end - chunk_start;

            for (int t_local = 0; t_local < chunk_len; ++t_local)
            {
                int t_bwd_local = chunk_len - 1 - t_local;  // Reverse time index
                
                int thread_id = t_local / kNItems;
                int item_id = t_local % kNItems;
                int thread_id_bwd = t_bwd_local / kNItems;
                int item_id_bwd = t_bwd_local % kNItems;

                float delta_val_fwd = 0.0f, u_val_fwd = 0.0f;
                float delta_val_bwd = 0.0f, u_val_bwd = 0.0f;

                __shared__ float delta_fwd_shared, u_fwd_shared;
                __shared__ float delta_bwd_shared, u_bwd_shared;

                if (threadIdx.x == thread_id && item_id < kNItems)
                {
                    delta_fwd_shared = delta_vals[0][item_id];
                    u_fwd_shared = float(u_vals[0][item_id]);
                }
                if (threadIdx.x == thread_id_bwd && item_id_bwd < kNItems)
                {
                    delta_bwd_shared = delta_vals[0][item_id_bwd];
                    u_bwd_shared = float(u_vals[0][item_id_bwd]);
                }
                __syncthreads();
                delta_val_fwd = delta_fwd_shared;
                u_val_fwd = u_fwd_shared;
                delta_val_bwd = delta_bwd_shared;
                u_val_bwd = u_bwd_shared;

                // Forward pass
                if (threadIdx.x < params.dstate)
                {
                    float B_val = 0.0f;
                    if constexpr (!kIsVariableB)
                    {
                        B_val = float(B[threadIdx.x * params.B_dstate_stride]);
                    }

                    float x_new_fwd_local[MAX_DSTATE];
                    structured_discretization<float>(
                        A_blocks_fwd_shared, A_U_fwd_shared, A_V_fwd_shared,
                        x_fwd_state, x_new_fwd_local,
                        delta_val_fwd, B_val, u_val_fwd,
                        params.dstate, params.block_size, params.num_blocks, params.low_rank_rank,
                        threadIdx.x, params.discretization_method);
                    x_new_fwd[threadIdx.x] = x_new_fwd_local[threadIdx.x];
                }
                __syncthreads();

                if (threadIdx.x < params.dstate)
                {
                    x_fwd_state[threadIdx.x] = x_new_fwd[threadIdx.x];
                }
                __syncthreads();

                // Backward pass
                if (threadIdx.x < params.dstate)
                {
                    float B_val = 0.0f;
                    if constexpr (!kIsVariableB)
                    {
                        B_val = float(B[threadIdx.x * params.B_dstate_stride]);
                    }

                    int block_size_b = params.use_structured_A_b ? params.block_size_b : params.block_size;
                    int num_blocks_b = params.use_structured_A_b ? params.num_blocks_b : params.num_blocks;
                    int rank_b = params.use_structured_A_b ? params.low_rank_rank_b : params.low_rank_rank;

                    float x_new_bwd_local[MAX_DSTATE];
                    structured_discretization<float>(
                        A_blocks_bwd_shared, A_U_bwd_shared, A_V_bwd_shared,
                        x_bwd_state, x_new_bwd_local,
                        delta_val_bwd, B_val, u_val_bwd,
                        params.dstate, block_size_b, num_blocks_b, rank_b,
                        threadIdx.x, params.discretization_method);
                    x_new_bwd[threadIdx.x] = x_new_bwd_local[threadIdx.x];
                }
                __syncthreads();

                if (threadIdx.x < params.dstate)
                {
                    x_bwd_state[threadIdx.x] = x_new_bwd[threadIdx.x];
                }
                __syncthreads();

                // Compute outputs
                __shared__ float y_fwd_shared[kChunkSize];
                __shared__ float y_bwd_shared[kChunkSize];
                
                if (threadIdx.x == 0)
                {
                    float y_fwd = 0.0f, y_bwd = 0.0f;
                    if constexpr (!kIsVariableC)
                    {
                        for (int s = 0; s < params.dstate; ++s)
                        {
                            float C_val = float(C[s * params.C_dstate_stride]);
                            y_fwd += C_val * x_fwd_state[s];
                            y_bwd += C_val * x_bwd_state[s];
                        }
                    }
                    y_fwd_shared[t_local] = y_fwd;
                    y_bwd_shared[t_bwd_local] = y_bwd;
                }
                __syncthreads();

                // Accumulate outputs
                int thread_for_t = t_local / kNItems;
                int item_for_t = t_local % kNItems;
                if (threadIdx.x == thread_for_t && item_for_t < kNItems)
                {
                    out_vals_fwd[0][item_for_t] += y_fwd_shared[t_local];
                }
                
                int thread_for_t_bwd = t_bwd_local / kNItems;
                int item_for_t_bwd = t_bwd_local % kNItems;
                if (threadIdx.x == thread_for_t_bwd && item_for_t_bwd < kNItems)
                {
                    out_vals_bwd[0][item_for_t_bwd] += y_bwd_shared[t_bwd_local];
                }
            }

            u += kChunkSize;
            delta += kChunkSize;

            // Combine forward and backward outputs
            float out_vals_combined[kNRows][kNItems];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    if (params.concat_bidirectional)
                    {
                        // Concatenation: store separately (combined at Python level)
                        out_vals_combined[r][i] = out_vals_fwd[r][i];  // Forward only here
                    }
                    else
                    {
                        // Addition
                        out_vals_combined[r][i] = out_vals_fwd[r][i] + out_vals_bwd[r][i];
                    }
                }
            }

            // Store outputs
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if constexpr (!kDirectIO)
                {
                    if (r > 0) { __syncthreads(); }
                }
                store_output<Ktraits>(out + r * params.out_d_stride + chunk * kChunkSize, out_vals_combined[r], smem_store, params.seqlen - chunk * kChunkSize);
                
                if (params.concat_bidirectional && out_fwd != nullptr)
                {
                    __syncthreads();
                    store_output<Ktraits>(out_fwd + r * params.out_d_stride + chunk * kChunkSize, out_vals_fwd[r], smem_store, params.seqlen - chunk * kChunkSize);
                    __syncthreads();
                    store_output<Ktraits>(out_bwd + r * params.out_d_stride + chunk * kChunkSize, out_vals_bwd[r], smem_store, params.seqlen - chunk * kChunkSize);
                }
            }

            // Store states
            if (threadIdx.x < params.dstate)
            {
                int r = 0;
                if constexpr (!kIsComplex)
                {
                    x_fwd[(r * params.n_chunks + chunk) * params.dstate + threadIdx.x] = make_float2(x_fwd_state[threadIdx.x], 0.0f);
                    if (x_bwd != nullptr)
                    {
                        x_bwd[(r * params.n_chunks + chunk_bwd) * params.dstate + threadIdx.x] = make_float2(x_bwd_state[threadIdx.x], 0.0f);
                    }
                }
                else
                {
                    x_fwd[(r * params.n_chunks + chunk) * params.dstate + threadIdx.x] = make_float4(x_fwd_state[threadIdx.x], 0.0f, 0.0f, 0.0f);
                    if (x_bwd != nullptr)
                    {
                        x_bwd[(r * params.n_chunks + chunk_bwd) * params.dstate + threadIdx.x] = make_float4(x_bwd_state[threadIdx.x], 0.0f, 0.0f, 0.0f);
                    }
                }
            }
        }
        return;
    }

    // Fallback to standard unidirectional processing if structured A not used
    // (This path would call the original kernel logic)
}

// ============================================================================
// Bidirectional Backward Pass Kernel
// Processes gradients for both forward and backward directions
// ============================================================================

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bidirectional_bwd_kernel(SSMParamsBidirectional params)
{
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    extern __shared__ char smem_[];
    auto &smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage &>(smem_);
    auto &smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage &>(smem_);
    auto &smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage &>(smem_);
    auto &smem_scan = reinterpret_cast<typename Ktraits::BlockScanT::TempStorage &>(smem_);

    int batch_id = blockIdx.x;
    int dim_id = blockIdx.y * kNRows;
    int group_id = threadIdx.y;

    constexpr int kChunkSize = kNThreads * kNItems;

    // Feature-SST: Bidirectional backward pass with structured A
    if (params.use_structured_A)
    {
        // Load forward and backward A components
        float *A_blocks_fwd = reinterpret_cast<float *>(params.A_blocks_ptr) + dim_id * params.A_block_stride;
        float *A_U_fwd = reinterpret_cast<float *>(params.A_U_ptr) + dim_id * params.A_U_stride;
        float *A_V_fwd = reinterpret_cast<float *>(params.A_V_ptr) + dim_id * params.A_V_stride;
        
        float *A_blocks_bwd = nullptr;
        float *A_U_bwd = nullptr;
        float *A_V_bwd = nullptr;
        
        if (params.use_structured_A_b && params.A_b_blocks_ptr != nullptr)
        {
            A_blocks_bwd = reinterpret_cast<float *>(params.A_b_blocks_ptr) + dim_id * params.A_b_block_stride;
            A_U_bwd = reinterpret_cast<float *>(params.A_b_U_ptr) + dim_id * params.A_b_U_stride;
            A_V_bwd = reinterpret_cast<float *>(params.A_b_V_ptr) + dim_id * params.A_b_V_stride;
        }
        else
        {
            // Use same A for both directions
            A_blocks_bwd = A_blocks_fwd;
            A_U_bwd = A_U_fwd;
            A_V_bwd = A_V_fwd;
        }
        
        // Load gradient outputs
        // For bidirectional backward, we need gradients w.r.t. forward and backward outputs
        // Note: SSMParamsBidirectional extends SSMParamsBase, which has dout_ptr
        input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride + dim_id * kNRows * params.dout_d_stride;
        input_t *dout_fwd = params.dout_fwd_ptr != nullptr
            ? reinterpret_cast<input_t *>(params.dout_fwd_ptr) + batch_id * params.dout_batch_stride + dim_id * kNRows * params.dout_d_stride
            : dout;  // Use combined gradient if not separate
        input_t *dout_bwd = params.dout_bwd_ptr != nullptr
            ? reinterpret_cast<input_t *>(params.dout_bwd_ptr) + batch_id * params.dout_batch_stride + dim_id * kNRows * params.dout_d_stride
            : dout;  // Use combined gradient if not separate
        
        // Load B and C
        weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
        input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
        weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
        input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
        
        // Load forward and backward states
        scan_t *x_fwd = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;
        scan_t *x_bwd = params.x_bwd_ptr != nullptr 
            ? reinterpret_cast<scan_t *>(params.x_bwd_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate
            : nullptr;
        
        // Shared memory for A components
        __shared__ float A_blocks_fwd_shared[16 * 16 * 16];
        __shared__ float A_U_fwd_shared[MAX_DSTATE * 16];
        __shared__ float A_V_fwd_shared[MAX_DSTATE * 16];
        __shared__ float A_blocks_bwd_shared[16 * 16 * 16];
        __shared__ float A_U_bwd_shared[MAX_DSTATE * 16];
        __shared__ float A_V_bwd_shared[MAX_DSTATE * 16];
        
        // Load A components cooperatively (same as forward)
        int total_block_elements = params.num_blocks * params.block_size * params.block_size;
        for (int idx = threadIdx.x; idx < total_block_elements; idx += blockDim.x)
        {
            if (idx < total_block_elements)
            {
                A_blocks_fwd_shared[idx] = A_blocks_fwd[idx];
                if (A_blocks_bwd != A_blocks_fwd)
                {
                    A_blocks_bwd_shared[idx] = A_blocks_bwd[idx];
                }
            }
        }
        
        int U_elements = params.dstate * params.low_rank_rank;
        for (int idx = threadIdx.x; idx < U_elements; idx += blockDim.x)
        {
            if (idx < U_elements)
            {
                A_U_fwd_shared[idx] = A_U_fwd[idx];
                A_V_fwd_shared[idx] = A_V_fwd[idx];
                if (A_blocks_bwd != A_blocks_fwd)
                {
                    A_U_bwd_shared[idx] = A_U_bwd[idx];
                    A_V_bwd_shared[idx] = A_V_bwd[idx];
                }
                else
                {
                    A_U_bwd_shared[idx] = A_U_fwd_shared[idx];
                    A_V_bwd_shared[idx] = A_V_fwd_shared[idx];
                }
            }
        }
        __syncthreads();
        
        // Shared state vectors
        __shared__ float x_fwd_state[MAX_DSTATE];
        __shared__ float x_bwd_state[MAX_DSTATE];
        __shared__ float grad_x_fwd_shared[MAX_DSTATE];
        __shared__ float grad_x_bwd_shared[MAX_DSTATE];
        
        // Initialize gradients
        if (threadIdx.x < params.dstate)
        {
            grad_x_fwd_shared[threadIdx.x] = 0.0f;
            grad_x_bwd_shared[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        // Process chunks in reverse order
        for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk)
        {
            int chunk_bwd = params.n_chunks - 1 - chunk;
            
            // Load gradient outputs for this chunk
            input_t dout_fwd_vals[kNRows][kNItems], dout_bwd_vals[kNRows][kNItems];
            __syncthreads();
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if constexpr (!kDirectIO)
                {
                    if (r > 0) { __syncthreads(); }
                }
                load_input<Ktraits>(dout_fwd + r * params.dout_d_stride + chunk * kChunkSize, dout_fwd_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
                load_input<Ktraits>(dout_bwd + r * params.dout_d_stride + chunk * kChunkSize, dout_bwd_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            }
            
            // Load forward and backward states for this chunk
            if (x_fwd != nullptr && threadIdx.x < params.dstate)
            {
                scan_t x_fwd_chunk = x_fwd[chunk * params.dstate + threadIdx.x];
                if constexpr (!kIsComplex)
                {
                    x_fwd_state[threadIdx.x] = x_fwd_chunk.x;
                }
                else
                {
                    x_fwd_state[threadIdx.x] = x_fwd_chunk.x; // Use real part
                }
            }
            if (x_bwd != nullptr && threadIdx.x < params.dstate)
            {
                scan_t x_bwd_chunk = x_bwd[chunk_bwd * params.dstate + threadIdx.x];
                if constexpr (!kIsComplex)
                {
                    x_bwd_state[threadIdx.x] = x_bwd_chunk.x;
                }
                else
                {
                    x_bwd_state[threadIdx.x] = x_bwd_chunk.x; // Use real part
                }
            }
            __syncthreads();
            
            // Process time steps in reverse order
            int chunk_start = chunk * kChunkSize;
            int chunk_end = (chunk_start + kChunkSize < params.seqlen) ? (chunk_start + kChunkSize) : params.seqlen;
            int chunk_len = chunk_end - chunk_start;
            
            // Load u and delta for gradient computation
            input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
            __syncthreads();
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if constexpr (!kDirectIO)
                {
                    if (r > 0) { __syncthreads(); }
                }
                input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride + dim_id * kNRows * params.u_d_stride;
                input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride + dim_id * kNRows * params.delta_d_stride;
                load_input<Ktraits>(u + r * params.u_d_stride + chunk * kChunkSize, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
                load_input<Ktraits>(delta + r * params.delta_d_stride + chunk * kChunkSize, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            }
            
            float delta_vals[kNRows][kNItems];
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
#pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    delta_vals[r][i] = float(delta_vals_load[r][i]) + (params.delta_bias_ptr != nullptr ? reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id + r] : 0.0f);
                    if (params.delta_softplus)
                    {
                        delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                    }
                }
            }
            
            // Gradient accumulators for A_fwd and A_bwd
            __shared__ float grad_A_fwd_blocks_shared[16 * 16 * 16];
            __shared__ float grad_A_fwd_U_shared[MAX_DSTATE * MAX_LOW_RANK];
            __shared__ float grad_A_fwd_V_shared[MAX_DSTATE * MAX_LOW_RANK];
            __shared__ float grad_A_bwd_blocks_shared[16 * 16 * 16];
            __shared__ float grad_A_bwd_U_shared[MAX_DSTATE * MAX_LOW_RANK];
            __shared__ float grad_A_bwd_V_shared[MAX_DSTATE * MAX_LOW_RANK];
            
            // Initialize gradient accumulators
            for (int idx = threadIdx.x; idx < total_block_elements; idx += blockDim.x)
            {
                if (idx < total_block_elements)
                {
                    grad_A_fwd_blocks_shared[idx] = 0.0f;
                    if (A_blocks_bwd != A_blocks_fwd)
                    {
                        grad_A_bwd_blocks_shared[idx] = 0.0f;
                    }
                }
            }
            for (int idx = threadIdx.x; idx < U_elements; idx += blockDim.x)
            {
                if (idx < U_elements)
                {
                    grad_A_fwd_U_shared[idx] = 0.0f;
                    grad_A_fwd_V_shared[idx] = 0.0f;
                    if (A_blocks_bwd != A_blocks_fwd)
                    {
                        grad_A_bwd_U_shared[idx] = 0.0f;
                        grad_A_bwd_V_shared[idx] = 0.0f;
                    }
                }
            }
            __syncthreads();
            
            for (int t_local = chunk_len - 1; t_local >= 0; --t_local)
            {
                int t_global = chunk_start + t_local;
                int t_bwd_local = chunk_len - 1 - t_local;
                
                // Get values for this time step
                float dout_fwd_val = 0.0f, dout_bwd_val = 0.0f;
                float delta_val = 0.0f, u_val = 0.0f;
                int thread_id = t_local / kNItems;
                int item_id = t_local % kNItems;
                
                if (threadIdx.x == thread_id && item_id < kNItems)
                {
                    dout_fwd_val = float(dout_fwd_vals[0][item_id]);
                    dout_bwd_val = float(dout_bwd_vals[0][item_id]);
                    delta_val = delta_vals[0][item_id];
                    u_val = float(u_vals[0][item_id]);
                }
                
                __shared__ float dout_fwd_shared, dout_bwd_shared, delta_val_shared, u_val_shared;
                if (threadIdx.x == thread_id)
                {
                    dout_fwd_shared = dout_fwd_val;
                    dout_bwd_shared = dout_bwd_val;
                    delta_val_shared = delta_val;
                    u_val_shared = u_val;
                }
                __syncthreads();
                dout_fwd_val = dout_fwd_shared;
                dout_bwd_val = dout_bwd_shared;
                delta_val = delta_val_shared;
                u_val = u_val_shared;
                
                // Compute gradients for forward direction
                if (threadIdx.x < params.dstate)
                {
                    // Gradient through output: dy/dx = C
                    float C_val = 0.0f;
                    if constexpr (!kIsVariableC)
                    {
                        C_val = float(C[threadIdx.x * params.C_dstate_stride]);
                    }
                    else
                    {
                        int C_offset = threadIdx.x * params.C_dstate_stride + t_global * (!kIsComplex ? 1 : 2);
                        C_val = float(Cvar[C_offset]);
                    }
                    grad_x_fwd_shared[threadIdx.x] += dout_fwd_val * C_val;
                }
                
                // Compute gradients for backward direction
                if (threadIdx.x < params.dstate)
                {
                    float C_b_val = 0.0f;
                    if (params.C_b_ptr != nullptr)
                    {
                        if constexpr (!kIsVariableC)
                        {
                            C_b_val = float(reinterpret_cast<weight_t *>(params.C_b_ptr)[threadIdx.x * params.C_dstate_stride]);
                        }
                        else
                        {
                            int C_b_offset = threadIdx.x * params.C_dstate_stride + t_global * (!kIsComplex ? 1 : 2);
                            C_b_val = float(reinterpret_cast<input_t *>(params.C_b_ptr)[C_b_offset]);
                        }
                    }
                    else
                    {
                        // Use same C as forward
                        if constexpr (!kIsVariableC)
                        {
                            C_b_val = float(C[threadIdx.x * params.C_dstate_stride]);
                        }
                        else
                        {
                            int C_offset = threadIdx.x * params.C_dstate_stride + t_global * (!kIsComplex ? 1 : 2);
                            C_b_val = float(Cvar[C_offset]);
                        }
                    }
                    grad_x_bwd_shared[threadIdx.x] += dout_bwd_val * C_b_val;
                }
                __syncthreads();
                
                // Gradient computation through state transitions for forward direction
                // Use method-specific gradients (same as unidirectional backward)
                if (threadIdx.x < params.dstate)
                {
                    int block_idx = threadIdx.x / params.block_size;
                    int local_i = threadIdx.x % params.block_size;
                    
                    float B_val = 0.0f;
                    if constexpr (!kIsVariableB)
                    {
                        B_val = float(B[threadIdx.x * params.B_dstate_stride]);
                    }
                    else
                    {
                        int B_offset = threadIdx.x * params.B_dstate_stride + t_global * (!kIsComplex ? 1 : 2);
                        B_val = float(Bvar[B_offset]);
                    }
                    
                    // Compute A @ x_old for delta gradient
                    float Ax_old_fwd = 0.0f;
                    if (block_idx < params.num_blocks)
                    {
                        for (int local_j = 0; local_j < params.block_size && (block_idx * params.block_size + local_j) < params.dstate; ++local_j)
                        {
                            int global_j = block_idx * params.block_size + local_j;
                            int A_idx = block_idx * params.block_size * params.block_size + local_i * params.block_size + local_j;
                            Ax_old_fwd += A_blocks_fwd_shared[A_idx] * x_fwd_state[global_j];
                        }
                    }
                    for (int r = 0; r < params.low_rank_rank; ++r)
                    {
                        float Vtx_r = 0.0f;
                        for (int j = 0; j < params.dstate; ++j)
                        {
                            Vtx_r += A_V_fwd_shared[j * params.low_rank_rank + r] * x_fwd_state[j];
                        }
                        Ax_old_fwd += A_U_fwd_shared[threadIdx.x * params.low_rank_rank + r] * Vtx_r;
                    }
                    
                    // Method-specific gradient coefficients
                    float delta_grad_coeff = 1.0f;
                    float B_grad_coeff = delta_val;
                    if (params.discretization_method == DISCRETIZATION_FOH)
                    {
                        float delta_sq = delta_val * delta_val;
                        B_grad_coeff = delta_sq * 0.5f;
                    }
                    else if (params.discretization_method == DISCRETIZATION_POLY || 
                             params.discretization_method == DISCRETIZATION_HIGHORDER)
                    {
                        float delta_sq = delta_val * delta_val;
                        B_grad_coeff = delta_val + delta_sq * 0.25f;
                    }
                    
                    // Gradient w.r.t. A_fwd_blocks
                    if (block_idx < params.num_blocks)
                    {
                        for (int local_j = 0; local_j < params.block_size && (block_idx * params.block_size + local_j) < params.dstate; ++local_j)
                        {
                            int global_j = block_idx * params.block_size + local_j;
                            int A_idx = block_idx * params.block_size * params.block_size + local_i * params.block_size + local_j;
                            float grad_A_contrib = delta_val * x_fwd_state[global_j] * grad_x_fwd_shared[threadIdx.x];
                            
                            // FOH: Additional gradient from B_d term
                            if (params.discretization_method == DISCRETIZATION_FOH && threadIdx.x == global_j)
                            {
                                float delta_cubed = delta_val * delta_val * delta_val;
                                grad_A_contrib += (delta_cubed / 6.0f * B_val * u_val) * grad_x_fwd_shared[threadIdx.x];
                            }
                            
                            atomicAdd(&grad_A_fwd_blocks_shared[A_idx], grad_A_contrib);
                        }
                    }
                    
                    // Gradient w.r.t. A_fwd_U and A_fwd_V
                    for (int r = 0; r < params.low_rank_rank; ++r)
                    {
                        float Vtx_r = 0.0f;
                        for (int j = 0; j < params.dstate; ++j)
                        {
                            Vtx_r += A_V_fwd_shared[j * params.low_rank_rank + r] * x_fwd_state[j];
                        }
                        atomicAdd(&grad_A_fwd_U_shared[threadIdx.x * params.low_rank_rank + r], 
                                  delta_val * Vtx_r * grad_x_fwd_shared[threadIdx.x]);
                        
                        for (int j = 0; j < params.dstate; ++j)
                        {
                            atomicAdd(&grad_A_fwd_V_shared[j * params.low_rank_rank + r],
                                      delta_val * A_U_fwd_shared[threadIdx.x * params.low_rank_rank + r] * grad_x_fwd_shared[threadIdx.x] * x_fwd_state[j]);
                        }
                    }
                    
                    // Gradient w.r.t. x_old: exp(delta*A_fwd)^T @ grad_x
                    float new_grad_x_fwd = grad_x_fwd_shared[threadIdx.x];
                    if (block_idx < params.num_blocks)
                    {
                        for (int local_j = 0; local_j < params.block_size && (block_idx * params.block_size + local_j) < params.dstate; ++local_j)
                        {
                            int global_j = block_idx * params.block_size + local_j;
                            int A_idx = block_idx * params.block_size * params.block_size + local_j * params.block_size + local_i; // Transpose
                            new_grad_x_fwd += delta_val * A_blocks_fwd_shared[A_idx] * grad_x_fwd_shared[global_j];
                        }
                    }
                    for (int r = 0; r < params.low_rank_rank; ++r)
                    {
                        float Utgrad_r = 0.0f;
                        for (int j = 0; j < params.dstate; ++j)
                        {
                            Utgrad_r += A_U_fwd_shared[j * params.low_rank_rank + r] * grad_x_fwd_shared[j];
                        }
                        new_grad_x_fwd += delta_val * A_V_fwd_shared[threadIdx.x * params.low_rank_rank + r] * Utgrad_r;
                    }
                    grad_x_fwd_shared[threadIdx.x] = new_grad_x_fwd;
                }
                
                // Gradient computation through state transitions for backward direction
                // Similar to forward but using A_bwd and x_bwd_state
                if (threadIdx.x < params.dstate)
                {
                    int block_idx = threadIdx.x / params.block_size;
                    int local_i = threadIdx.x % params.block_size;
                    
                    float B_b_val = 0.0f;
                    if (params.B_b_ptr != nullptr)
                    {
                        if constexpr (!kIsVariableB)
                        {
                            B_b_val = float(reinterpret_cast<weight_t *>(params.B_b_ptr)[threadIdx.x * params.B_dstate_stride]);
                        }
                        else
                        {
                            int B_b_offset = threadIdx.x * params.B_dstate_stride + t_global * (!kIsComplex ? 1 : 2);
                            B_b_val = float(reinterpret_cast<input_t *>(params.B_b_ptr)[B_b_offset]);
                        }
                    }
                    else
                    {
                        if constexpr (!kIsVariableB)
                        {
                            B_b_val = float(B[threadIdx.x * params.B_dstate_stride]);
                        }
                        else
                        {
                            int B_offset = threadIdx.x * params.B_dstate_stride + t_global * (!kIsComplex ? 1 : 2);
                            B_b_val = float(Bvar[B_offset]);
                        }
                    }
                    
                    // Compute A_bwd @ x_bwd_old for delta gradient
                    float Ax_old_bwd = 0.0f;
                    if (block_idx < params.num_blocks)
                    {
                        for (int local_j = 0; local_j < params.block_size && (block_idx * params.block_size + local_j) < params.dstate; ++local_j)
                        {
                            int global_j = block_idx * params.block_size + local_j;
                            int A_idx = block_idx * params.block_size * params.block_size + local_i * params.block_size + local_j;
                            Ax_old_bwd += A_blocks_bwd_shared[A_idx] * x_bwd_state[global_j];
                        }
                    }
                    for (int r = 0; r < params.low_rank_rank; ++r)
                    {
                        float Vtx_r = 0.0f;
                        for (int j = 0; j < params.dstate; ++j)
                        {
                            Vtx_r += A_V_bwd_shared[j * params.low_rank_rank + r] * x_bwd_state[j];
                        }
                        Ax_old_bwd += A_U_bwd_shared[threadIdx.x * params.low_rank_rank + r] * Vtx_r;
                    }
                    
                    // Method-specific gradient coefficients (same as forward)
                    float delta_grad_coeff = 1.0f;
                    float B_grad_coeff = delta_val;
                    if (params.discretization_method == DISCRETIZATION_FOH)
                    {
                        float delta_sq = delta_val * delta_val;
                        B_grad_coeff = delta_sq * 0.5f;
                    }
                    else if (params.discretization_method == DISCRETIZATION_POLY || 
                             params.discretization_method == DISCRETIZATION_HIGHORDER)
                    {
                        float delta_sq = delta_val * delta_val;
                        B_grad_coeff = delta_val + delta_sq * 0.25f;
                    }
                    
                    // Gradient w.r.t. A_bwd_blocks (only if separate from A_fwd)
                    if (A_blocks_bwd != A_blocks_fwd && block_idx < params.num_blocks)
                    {
                        for (int local_j = 0; local_j < params.block_size && (block_idx * params.block_size + local_j) < params.dstate; ++local_j)
                        {
                            int global_j = block_idx * params.block_size + local_j;
                            int A_idx = block_idx * params.block_size * params.block_size + local_i * params.block_size + local_j;
                            float grad_A_contrib = delta_val * x_bwd_state[global_j] * grad_x_bwd_shared[threadIdx.x];
                            
                            if (params.discretization_method == DISCRETIZATION_FOH && threadIdx.x == global_j)
                            {
                                float delta_cubed = delta_val * delta_val * delta_val;
                                grad_A_contrib += (delta_cubed / 6.0f * B_b_val * u_val) * grad_x_bwd_shared[threadIdx.x];
                            }
                            
                            atomicAdd(&grad_A_bwd_blocks_shared[A_idx], grad_A_contrib);
                        }
                    }
                    
                    // Gradient w.r.t. A_bwd_U and A_bwd_V (only if separate)
                    if (A_blocks_bwd != A_blocks_fwd)
                    {
                        for (int r = 0; r < params.low_rank_rank; ++r)
                        {
                            float Vtx_r = 0.0f;
                            for (int j = 0; j < params.dstate; ++j)
                            {
                                Vtx_r += A_V_bwd_shared[j * params.low_rank_rank + r] * x_bwd_state[j];
                            }
                            atomicAdd(&grad_A_bwd_U_shared[threadIdx.x * params.low_rank_rank + r], 
                                      delta_val * Vtx_r * grad_x_bwd_shared[threadIdx.x]);
                            
                            for (int j = 0; j < params.dstate; ++j)
                            {
                                atomicAdd(&grad_A_bwd_V_shared[j * params.low_rank_rank + r],
                                          delta_val * A_U_bwd_shared[threadIdx.x * params.low_rank_rank + r] * grad_x_bwd_shared[threadIdx.x] * x_bwd_state[j]);
                            }
                        }
                    }
                    
                    // Gradient w.r.t. x_bwd_old: exp(delta*A_bwd)^T @ grad_x
                    float new_grad_x_bwd = grad_x_bwd_shared[threadIdx.x];
                    if (block_idx < params.num_blocks)
                    {
                        for (int local_j = 0; local_j < params.block_size && (block_idx * params.block_size + local_j) < params.dstate; ++local_j)
                        {
                            int global_j = block_idx * params.block_size + local_j;
                            int A_idx = block_idx * params.block_size * params.block_size + local_j * params.block_size + local_i; // Transpose
                            new_grad_x_bwd += delta_val * A_blocks_bwd_shared[A_idx] * grad_x_bwd_shared[global_j];
                        }
                    }
                    for (int r = 0; r < params.low_rank_rank; ++r)
                    {
                        float Utgrad_r = 0.0f;
                        for (int j = 0; j < params.dstate; ++j)
                        {
                            Utgrad_r += A_U_bwd_shared[j * params.low_rank_rank + r] * grad_x_bwd_shared[j];
                        }
                        new_grad_x_bwd += delta_val * A_V_bwd_shared[threadIdx.x * params.low_rank_rank + r] * Utgrad_r;
                    }
                    grad_x_bwd_shared[threadIdx.x] = new_grad_x_bwd;
                }
                __syncthreads();
            }
            
            // Store accumulated gradients for A_fwd and A_bwd to global memory
            // Note: dA_blocks_stride, dA_U_stride, dA_V_stride are in SSMParamsBase
            // For bidirectional, we use the same strides for both directions
            if (params.dA_fwd_blocks_ptr != nullptr && threadIdx.x < total_block_elements)
            {
                float *dA_fwd_blocks = reinterpret_cast<float *>(params.dA_fwd_blocks_ptr) + dim_id * params.dA_blocks_stride;
                atomicAdd(&dA_fwd_blocks[threadIdx.x], grad_A_fwd_blocks_shared[threadIdx.x]);
            }
            if (A_blocks_bwd != A_blocks_fwd && params.dA_bwd_blocks_ptr != nullptr && threadIdx.x < total_block_elements)
            {
                float *dA_bwd_blocks = reinterpret_cast<float *>(params.dA_bwd_blocks_ptr) + dim_id * params.dA_blocks_stride;
                atomicAdd(&dA_bwd_blocks[threadIdx.x], grad_A_bwd_blocks_shared[threadIdx.x]);
            }
            
            if (params.dA_fwd_U_ptr != nullptr && threadIdx.x < U_elements)
            {
                float *dA_fwd_U = reinterpret_cast<float *>(params.dA_fwd_U_ptr) + dim_id * params.dA_U_stride;
                atomicAdd(&dA_fwd_U[threadIdx.x], grad_A_fwd_U_shared[threadIdx.x]);
            }
            if (A_blocks_bwd != A_blocks_fwd && params.dA_bwd_U_ptr != nullptr && threadIdx.x < U_elements)
            {
                float *dA_bwd_U = reinterpret_cast<float *>(params.dA_bwd_U_ptr) + dim_id * params.dA_U_stride;
                atomicAdd(&dA_bwd_U[threadIdx.x], grad_A_bwd_U_shared[threadIdx.x]);
            }
            
            if (params.dA_fwd_V_ptr != nullptr && threadIdx.x < U_elements)
            {
                float *dA_fwd_V = reinterpret_cast<float *>(params.dA_fwd_V_ptr) + dim_id * params.dA_V_stride;
                atomicAdd(&dA_fwd_V[threadIdx.x], grad_A_fwd_V_shared[threadIdx.x]);
            }
            if (A_blocks_bwd != A_blocks_fwd && params.dA_bwd_V_ptr != nullptr && threadIdx.x < U_elements)
            {
                float *dA_bwd_V = reinterpret_cast<float *>(params.dA_bwd_V_ptr) + dim_id * params.dA_V_stride;
                atomicAdd(&dA_bwd_V[threadIdx.x], grad_A_bwd_V_shared[threadIdx.x]);
            }
            __syncthreads();
        }
        return;
    }
    
    // Fallback to standard unidirectional backward if structured A not used
}

// Launch function for bidirectional backward kernel
template <int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_bidirectional_bwd_launch(SSMParamsBidirectional &params, cudaStream_t stream)
{
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&]
    { BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&]
    { BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&]
    { BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&]
    {
        using Ktraits = Selective_Scan_bidirectional_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
        
        constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
        dim3 grid(params.batch, params.dim / kNRows);

        auto kernel = &selective_scan_bidirectional_bwd_kernel<Ktraits>;

        if (kSmemSize >= 48 * 1024) {
#ifndef USE_ROCM
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#endif
        }

        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }); }); }); });
}

template <typename input_t, typename weight_t>
void selective_scan_bidirectional_bwd_cuda(SSMParamsBidirectional &params, cudaStream_t stream)
{
#ifndef USE_ROCM
    if (params.seqlen <= 128)
    {
        selective_scan_bidirectional_bwd_launch<32, 4, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 256)
    {
        selective_scan_bidirectional_bwd_launch<32, 8, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 512)
    {
        selective_scan_bidirectional_bwd_launch<32, 16, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 1024)
    {
        selective_scan_bidirectional_bwd_launch<64, 16, input_t, weight_t>(params, stream);
    }
    else
    {
        selective_scan_bidirectional_bwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
#else
    if (params.seqlen <= 256)
    {
        selective_scan_bidirectional_bwd_launch<64, 4, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 512)
    {
        selective_scan_bidirectional_bwd_launch<64, 8, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 1024)
    {
        selective_scan_bidirectional_bwd_launch<64, 16, input_t, weight_t>(params, stream);
    }
    else
    {
        selective_scan_bidirectional_bwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
#endif
}

// Launch function for bidirectional kernel
template <int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_bidirectional_fwd_launch(SSMParamsBidirectional &params, cudaStream_t stream)
{
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&]
    { BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&]
    { BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&]
    { BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&]
    {
        using Ktraits = Selective_Scan_bidirectional_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
        
        constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
        dim3 grid(params.batch, params.dim / kNRows);

        auto kernel = &selective_scan_bidirectional_fwd_kernel<Ktraits>;

        if (kSmemSize >= 48 * 1024) {
#ifndef USE_ROCM
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#endif
        }

        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }); }); }); });
}

template <typename input_t, typename weight_t>
void selective_scan_bidirectional_fwd_cuda(SSMParamsBidirectional &params, cudaStream_t stream)
{
#ifndef USE_ROCM
    if (params.seqlen <= 128)
    {
        selective_scan_bidirectional_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 256)
    {
        selective_scan_bidirectional_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 512)
    {
        selective_scan_bidirectional_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 1024)
    {
        selective_scan_bidirectional_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    }
    else
    {
        selective_scan_bidirectional_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
#else
    if (params.seqlen <= 256)
    {
        selective_scan_bidirectional_fwd_launch<64, 4, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 512)
    {
        selective_scan_bidirectional_fwd_launch<64, 8, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 1024)
    {
        selective_scan_bidirectional_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    }
    else
    {
        selective_scan_bidirectional_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
#endif
}

#endif // SELECTIVE_SCAN_BIDIRECTIONAL_CUH
