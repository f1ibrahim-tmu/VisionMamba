/******************************************************************************
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h> // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

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

template <int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
          bool kIsVariableB_, bool kIsVariableC_,
          bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits
{
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
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
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
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
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks) void selective_scan_fwd_kernel(SSMParamsBase params)
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

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto &smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage &>(smem_);
    auto &smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage &>(smem_);
    auto &smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage *>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto &smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage &>(smem_);
    auto &smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage *>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride + dim_id * kNRows * params.delta_d_stride;

    // Feature-SST: Structured State Transitions - Block-Diagonal + Low-Rank A
    // SST = Structured State Transitions: A = blockdiag(A_1, ..., A_K) + UV^T
    // ONLY use block-diagonal + low-rank structure, NO full A matrices
    weight_t *A = nullptr;
    float *A_blocks_ptr = nullptr; // Block-diagonal components
    float *A_U_ptr = nullptr; // Low-rank U component
    float *A_V_ptr = nullptr; // Low-rank V component
    
    if (params.use_structured_A)
    {
        // Structured A: A_blocks is (d_inner, num_blocks, block_size, block_size)
        // A_U is (d_inner, d_state, low_rank_rank)
        // A_V is (d_inner, d_state, low_rank_rank)
        A_blocks_ptr = reinterpret_cast<float *>(params.A_blocks_ptr) + dim_id * params.A_block_stride;
        A_U_ptr = reinterpret_cast<float *>(params.A_U_ptr) + dim_id * params.A_U_stride;
        A_V_ptr = reinterpret_cast<float *>(params.A_V_ptr) + dim_id * params.A_V_stride;
    }
    else
    {
        // A is (dim, d_state) - diagonal (original Mamba)
        A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    }

    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

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

    // for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
    //     smem_a[state_idx] = A[state_idx * params.A_dstate_stride];
    //     smem_bc[state_idx] = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride];
    // }

    constexpr int kChunkSize = kNThreads * kNItems;

    // Feature-SST: Structured State Transitions - Block-Diagonal + Low-Rank A
    // SST = Structured State Transitions: A = blockdiag(A_1, ..., A_K) + UV^T
    // ONLY use block-diagonal + low-rank structure, NO full A matrices
    for (int chunk = 0; chunk < params.n_chunks; ++chunk)
    {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            if constexpr (!kDirectIO)
            {
                if (r > 0)
                {
                    __syncthreads();
                }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO)
            {
                __syncthreads();
            }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
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
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx)
        {
            weight_t A_val[kNRows];
            weight_t A_val_orig[kNRows]; // Store original A values for non-ZOH methods
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                // Feature-SST: A is guaranteed to be diagonal here (full A path handled above)
                A_val_orig[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                A_val[r] = A_val_orig[r];
                // For ZOH, multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                // For other methods, we'll use the original A value in the discretization function.
                if (params.discretization_method == DISCRETIZATION_ZOH)
                {
                    constexpr float kLog2e = M_LOG2E;
                    if constexpr (!kIsComplex)
                    {
                        A_val[r] *= kLog2e;
                    }
                    else
                    {
                        A_val[r].real_ *= kLog2e;
                    }
                }
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB)
            {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                                     smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableC)
                {
#pragma unroll
                    for (int r = 0; r < kNRows; ++r)
                    {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC)
            {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                                     smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableB)
                {
#pragma unroll
                    for (int r = 0; r < kNRows; ++r)
                    {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC)
            {
#pragma unroll
                for (int r = 0; r < kNRows; ++r)
                {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if (r > 0)
                {
                    __syncthreads();
                } // Scan could be using the same smem
                scan_t thread_data[kNItems];
#pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    if constexpr (!kIsComplex)
                    {
                        // Get B value (either constant from B matrix or variable B_vals)
                        float B_val_actual;
                        if constexpr (!kIsVariableB)
                        {
                            B_val_actual = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                        }
                        else
                        {
                            B_val_actual = B_vals[i];
                        }
                        float delta_u_val = !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i];
                        // Use original A value for non-ZOH methods
                        weight_t A_val_to_use = (params.discretization_method == DISCRETIZATION_ZOH) ? A_val[r] : A_val_orig[r];
                        thread_data[i] = compute_discretization<weight_t, false>(
                            delta_vals[r][i],
                            A_val_to_use,
                            delta_u_val,
                            B_val_actual,
                            params.discretization_method);
                        if constexpr (!Ktraits::kIsEvenLen)
                        { // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize)
                            {
                                thread_data[i] = make_float2(1.f, 0.f);
            }
        }
    }
    else if (params.use_structured_A)
    {
        // Feature-SST: Optimized path using block-diagonal + low-rank structure
        // A = blockdiag(A_1, ..., A_K) + UV^T
        // We use optimized operations to avoid constructing full matrix
        
        // Load structured components into shared memory
        // A_blocks: (num_blocks, block_size, block_size) for this dim_id
        // A_U: (d_state, low_rank_rank) for this dim_id
        // A_V: (d_state, low_rank_rank) for this dim_id
        __shared__ float A_blocks_shared[16 * 16 * 16]; // Max: 16 blocks * 16x16 each
        __shared__ float A_U_shared[MAX_DSTATE * 16]; // Max: d_state=256, rank=16
        __shared__ float A_V_shared[MAX_DSTATE * 16];
        
        // Load A_blocks cooperatively
        int total_block_elements = params.num_blocks * params.block_size * params.block_size;
        for (int idx = threadIdx.x; idx < total_block_elements; idx += blockDim.x)
        {
            int block_idx = idx / (params.block_size * params.block_size);
            int block_offset = idx % (params.block_size * params.block_size);
            int i = block_offset / params.block_size;
            int j = block_offset % params.block_size;
            if (block_idx < params.num_blocks && i < params.block_size && j < params.block_size)
            {
                // A_blocks is stored as (d_inner, num_blocks, block_size, block_size)
                // Access: A_blocks_ptr[block_idx * (block_size * block_size) + i * block_size + j]
                int A_blocks_idx = block_idx * params.block_size * params.block_size + i * params.block_size + j;
                A_blocks_shared[idx] = A_blocks_ptr[A_blocks_idx];
            }
        }
        
        // Load A_U and A_V cooperatively
        int U_elements = params.dstate * params.low_rank_rank;
        int V_elements = params.dstate * params.low_rank_rank;
        for (int idx = threadIdx.x; idx < U_elements; idx += blockDim.x)
        {
            A_U_shared[idx] = A_U_ptr[idx];
        }
        for (int idx = threadIdx.x; idx < V_elements; idx += blockDim.x)
        {
            A_V_shared[idx] = A_V_ptr[idx];
        }
        __syncthreads();
        
        // Shared state vector
        __shared__ float x_state_shared[MAX_DSTATE];
        __shared__ float x_state_new[MAX_DSTATE];
        
        // Initialize state vector
        if (threadIdx.x < params.dstate)
        {
            x_state_shared[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        // Process chunks
        for (int chunk = 0; chunk < params.n_chunks; ++chunk)
        {
            input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
            __syncthreads();
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if constexpr (!kDirectIO)
                {
                    if (r > 0)
                    {
                        __syncthreads();
                    }
                }
                load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
                if constexpr (!kDirectIO)
                {
                    __syncthreads();
                }
                load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            }
            
            float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
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
                    delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                    out_vals[r][i] = D_val[r] * u_val;
                }
            }
            
            // Feature-SST: No need to store full exp_deltaA matrix!
            // We compute exp(delta * A) @ x directly from structured components
            
            int chunk_start = chunk * kChunkSize;
            int chunk_end = (chunk_start + kChunkSize < params.seqlen) ? (chunk_start + kChunkSize) : params.seqlen;
            int chunk_len = chunk_end - chunk_start;
            
            for (int t_local = 0; t_local < chunk_len; ++t_local)
            {
                int t_global = chunk_start + t_local;
                int thread_id = t_local / kNItems;
                int item_id = t_local % kNItems;
                
                float delta_val = 0.0f;
                float u_val = 0.0f;
                
                if (threadIdx.x == thread_id && item_id < kNItems && t_local < chunk_len)
                {
                    delta_val = delta_vals[0][item_id];
                    u_val = float(u_vals[0][item_id]);
                }
                __shared__ float delta_val_shared;
                __shared__ float u_val_shared;
                if (threadIdx.x == thread_id)
                {
                    delta_val_shared = delta_val;
                    u_val_shared = u_val;
                }
                __syncthreads();
                delta_val = delta_val_shared;
                u_val = u_val_shared;
                
                // Feature-SST: Optimized state transition using block-diagonal + low-rank structure
                // Supports all 6 discretization methods: ZOH, FOH, Bilinear, RK4, Poly, Highorder
                // Computes directly from structured components without forming full matrix
                if (threadIdx.x < params.dstate)
                {
                    float B_val = 0.0f;
                    if constexpr (!kIsVariableB)
                    {
                        B_val = float(B[threadIdx.x * params.B_dstate_stride]);
                    }
                    
                    float x_new_local[MAX_DSTATE];
                    
                    // Use unified discretization function for structured A
                    // This handles all 6 discretization methods
                    structured_discretization<float>(
                        A_blocks_shared,      // Block matrices
                        A_U_shared,          // Low-rank U
                        A_V_shared,          // Low-rank V
                        x_state_shared,      // Input: x_old
                        x_new_local,         // Output: x_new
                        delta_val,           // Time step
                        B_val,               // B value
                        u_val,               // Input u
                        params.dstate,       // State dimension
                        params.block_size,   // Block size
                        params.num_blocks,   // Number of blocks
                        params.low_rank_rank, // Low-rank rank
                        threadIdx.x,         // State index
                        params.discretization_method  // Discretization method
                    );
                    
                    x_state_new[threadIdx.x] = x_new_local[threadIdx.x];
                }
                __syncthreads();
                
                if (threadIdx.x < params.dstate)
                {
                    x_state_shared[threadIdx.x] = x_state_new[threadIdx.x];
                }
                __syncthreads();
                
                __shared__ float y_vals_shared[kChunkSize];
                if (threadIdx.x == 0)
                {
                    float y_val = 0.0f;
                    if constexpr (!kIsVariableC)
                    {
                        for (int s = 0; s < params.dstate; ++s)
                        {
                            y_val += float(C[s * params.C_dstate_stride]) * x_state_shared[s];
                        }
                    }
                    y_vals_shared[t_local] = y_val;
                }
                __syncthreads();
                
                int thread_for_t = t_local / kNItems;
                int item_for_t = t_local % kNItems;
                if (threadIdx.x == thread_for_t && item_for_t < kNItems)
                {
                    out_vals[0][item_for_t] += y_vals_shared[t_local];
                }
            }
            
            u += kChunkSize;
            delta += kChunkSize;
            
            if (threadIdx.x < params.dstate)
            {
                int r = 0;
                if constexpr (!kIsComplex)
                {
                    x[(r * params.n_chunks + chunk) * params.dstate + threadIdx.x] = make_float2(x_state_shared[threadIdx.x], 0.0f);
                }
                else
                {
                    x[(r * params.n_chunks + chunk) * params.dstate + threadIdx.x] = make_float4(x_state_shared[threadIdx.x], 0.0f, 0.0f, 0.0f);
                }
            }
            
            input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
            __syncthreads();
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if constexpr (!kDirectIO)
                {
                    if (r > 0)
                    {
                        __syncthreads();
                    }
                }
                store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
            
            if constexpr (kHasZ)
            {
                input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
                input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
#pragma unroll
                for (int r = 0; r < kNRows; ++r)
                {
                    input_t z_vals[kNItems];
                    __syncthreads();
                    load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
#pragma unroll
                    for (int i = 0; i < kNItems; ++i)
                    {
                        float z_val = z_vals[i];
                        out_vals[r][i] *= z_val / (1 + expf(-z_val));
                    }
                    __syncthreads();
                    store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
                }
            }
        }
        return; // Early return for structured A path
    }
    else
    {
                        // Complex case
                        weight_t B_val_actual;
                        if constexpr (!kIsVariableB)
                        {
                            B_val_actual = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                        }
                        else
                        {
                            B_val_actual = B_vals[i];
                        }
                        weight_t B_delta_u_val = !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i];
                        // Use original A value for non-ZOH methods
                        weight_t A_val_to_use = (params.discretization_method == DISCRETIZATION_ZOH) ? A_val[r] : A_val_orig[r];
                        thread_data[i] = compute_discretization<weight_t, true>(
                            delta_vals[r][i],
                            A_val_to_use,
                            !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i],
                            B_val_actual,
                            params.discretization_method);
                        if constexpr (!Ktraits::kIsEvenLen)
                        { // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize)
                            {
                                thread_data[i] = make_float4(1.f, 0.f, 0.f, 0.f);
                            }
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                if constexpr (!kIsComplex)
                {
                    // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
                }
                else
                {
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float4(1.f, 0.f, 0.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                }
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op);
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0)
                {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
#pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    const weight_t C_val = !kIsVariableC
                                               ? BC_val[r]
                                               : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    if constexpr (!kIsComplex)
                    {
                        out_vals[r][i] += thread_data[i].y * C_val;
                    }
                    else
                    {
                        out_vals[r][i] += (complex_t(thread_data[i].z, thread_data[i].w) * C_val).real_ * 2;
                    }
                }
            }
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            if constexpr (!kDirectIO)
            {
                if (r > 0)
                {
                    __syncthreads();
                }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ)
        {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
#pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar += kChunkSize * (!kIsComplex ? 1 : 2);
    }
}

template <int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream)
{
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&]
                { BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&]
                              { BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&]
                                            { BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&]
                                                          {
                    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
                    
                    constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                    dim3 grid(params.batch, params.dim / kNRows);

                    // Had to change this substantially since potentially the hip 
                    // interface for setting kernel launch attributes is slightly different from 
                    // cuda's. In particualar, it seems to expect a plain const void * pointer.

                    auto kernel = &selective_scan_fwd_kernel<Ktraits>;

                    
                    if (kSmemSize >= 48 * 1024) {
#ifndef USE_ROCM
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            std::cerr << "Warning (selective_scan_fwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
#endif
                    }

                    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); }); }); }); });
}

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream)
{

#ifndef USE_ROCM
    if (params.seqlen <= 128)
    {
        selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 256)
    {
        selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 512)
    {
        selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 1024)
    {
        selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    }
    else
    {
        selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
#else
    if (params.seqlen <= 256)
    {
        selective_scan_fwd_launch<64, 4, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 512)
    {
        selective_scan_fwd_launch<64, 8, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 1024)
    {
        selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    }
    else
    {
        selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
#endif
}
