/******************************************************************************
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMScanParamsBase {
    using index_t = uint32_t;

    int batch, seqlen, n_chunks;
    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t out_batch_stride;

    // Common data pointers.
    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Discretization method enum
enum DiscretizationMethod {
    DISCRETIZATION_ZOH = 0,
    DISCRETIZATION_FOH = 1,
    DISCRETIZATION_BILINEAR = 2,
    DISCRETIZATION_POLY = 3,
    DISCRETIZATION_HIGHORDER = 4,
    DISCRETIZATION_RK4 = 5
};

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;

    bool delta_softplus;
    DiscretizationMethod discretization_method;

    // Feature-SST: Structured State Transitions - Block-Diagonal + Low-Rank A matrix support
    // SST = Structured State Transitions: A = blockdiag(A_1, ..., A_K) + UV^T
    // This allows cross-channel dynamics while maintaining computational efficiency
    bool is_full_A_matrix;  // true if A is (d_inner, d_state, d_state), false if (d_inner, d_state)
    bool use_structured_A;  // true if using block-diagonal + low-rank structure (A_blocks, A_U, A_V)
    int block_size;          // Size of blocks in block-diagonal structure (0 if not block-diagonal)
    int low_rank_rank;       // Rank of low-rank component (0 if not low-rank)
    int num_blocks;          // Number of blocks (d_state / block_size)

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t A_matrix_stride;  // New: stride for accessing full A matrices (dstate * dstate)
    index_t A_block_stride;  // Stride for A_blocks: (d_inner, num_blocks, block_size, block_size)
    index_t A_U_stride;      // Stride for A_U: (d_inner, d_state, low_rank_rank)
    index_t A_V_stride;      // Stride for A_V: (d_inner, d_state, low_rank_rank)
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // Common data pointers.
    void *__restrict__ A_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_z_ptr;
    // Feature-SST: Block-diagonal + low-rank A components
    void *__restrict__ A_blocks_ptr;  // (d_inner, num_blocks, block_size, block_size)
    void *__restrict__ A_U_ptr;       // (d_inner, d_state, low_rank_rank)
    void *__restrict__ A_V_ptr;       // (d_inner, d_state, low_rank_rank)
};

struct SSMParamsBwd: public SSMParamsBase {
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_d_stride;
    index_t dC_dstate_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t dz_batch_stride;
    index_t dz_d_stride;
    index_t ddelta_batch_stride;
    index_t ddelta_d_stride;

    // Feature-SST: Strides for structured A gradients
    index_t dA_blocks_stride;  // For (d_inner, num_blocks, block_size, block_size)
    index_t dA_U_stride;       // For (d_inner, d_state, low_rank_rank)
    index_t dA_V_stride;       // For (d_inner, d_state, low_rank_rank)

    // Common data pointers.
    void *__restrict__ dout_ptr;
    void *__restrict__ dA_ptr;
    void *__restrict__ dB_ptr;
    void *__restrict__ dC_ptr;
    void *__restrict__ dD_ptr;
    void *__restrict__ du_ptr;
    void *__restrict__ dz_ptr;
    void *__restrict__ ddelta_ptr;
    void *__restrict__ ddelta_bias_ptr;

    // Feature-SST: Gradient pointers for structured A components
    void *__restrict__ dA_blocks_ptr;  // Gradient for A_blocks
    void *__restrict__ dA_U_ptr;       // Gradient for A_U
    void *__restrict__ dA_V_ptr;       // Gradient for A_V
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Feature-SST: Bidirectional Mamba support parameters
struct SSMParamsBidirectional: public SSMParamsBase {
    // Backward direction A matrix components
    bool use_structured_A_b;  // Use structured A for backward direction
    int block_size_b;
    int low_rank_rank_b;
    int num_blocks_b;
    
    index_t A_b_d_stride;
    index_t A_b_dstate_stride;
    index_t A_b_block_stride;
    index_t A_b_U_stride;
    index_t A_b_V_stride;
    
    // Backward direction pointers
    void *__restrict__ A_b_ptr;        // Backward A (if diagonal)
    void *__restrict__ A_b_blocks_ptr; // Backward A blocks
    void *__restrict__ A_b_U_ptr;      // Backward A low-rank U
    void *__restrict__ A_b_V_ptr;      // Backward A low-rank V
    void *__restrict__ B_b_ptr;        // Backward B
    void *__restrict__ C_b_ptr;        // Backward C
    
    // Output pointers for bidirectional
    void *__restrict__ out_fwd_ptr;    // Forward pass output
    void *__restrict__ out_bwd_ptr;    // Backward pass output
    void *__restrict__ x_bwd_ptr;      // Backward state storage
    
    // Backward pass gradient pointers (for bidirectional backward)
    void *__restrict__ dout_fwd_ptr;   // Gradient w.r.t. forward output
    void *__restrict__ dout_bwd_ptr;   // Gradient w.r.t. backward output
    void *__restrict__ dA_fwd_blocks_ptr;  // Gradient for forward A_blocks
    void *__restrict__ dA_fwd_U_ptr;       // Gradient for forward A_U
    void *__restrict__ dA_fwd_V_ptr;       // Gradient for forward A_V
    void *__restrict__ dA_bwd_blocks_ptr;  // Gradient for backward A_blocks (if separate)
    void *__restrict__ dA_bwd_U_ptr;       // Gradient for backward A_U (if separate)
    void *__restrict__ dA_bwd_V_ptr;       // Gradient for backward A_V (if separate)
    
    // Backward pass gradient strides (from SSMParamsBwd)
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dA_blocks_stride;  // For (d_inner, num_blocks, block_size, block_size)
    index_t dA_U_stride;       // For (d_inner, d_state, low_rank_rank)
    index_t dA_V_stride;       // For (d_inner, d_state, low_rank_rank)
    
    // Combination method
    bool concat_bidirectional;  // true = concatenate, false = add
};
