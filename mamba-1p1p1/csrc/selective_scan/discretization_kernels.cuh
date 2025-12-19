/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Modified to support multiple discretization methods
 ******************************************************************************/

#pragma once

#include "selective_scan_common.h"
#include "selective_scan.h"

// Helper function to compute A_d and B_d for different discretization methods
// Returns (A_d, B_d * u) as float2 for real or float4 for complex
template <typename weight_t, bool kIsComplex>
__device__ __forceinline__ auto compute_discretization(
    float delta_val,
    weight_t A_val,
    float delta_u_val,
    weight_t B_val,
    DiscretizationMethod method)
{
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;

    if constexpr (!kIsComplex)
    {
        float A_d, B_d_u;

        switch (method)
        {
        case DISCRETIZATION_ZOH:
        {
            // ZOH: A_d = exp(A*delta), B_d = delta * B
            constexpr float kLog2e = M_LOG2E;
            A_d = exp2f(delta_val * A_val * kLog2e);
            B_d_u = delta_u_val; // Already multiplied by delta
            break;
        }

        case DISCRETIZATION_FOH:
        {
            // FOH (Correct Formula): A_d = exp(A*delta), B_d = A^(-2) * (exp(A*delta) - I - A*delta) * B
            // Using Taylor series expansion to avoid division:
            // (exp(A*Δ) - 1 - A*Δ) / A^2 = Δ²/2! + A*Δ³/3! + A²*Δ⁴/4! + A³*Δ⁵/5! + ...
            // So: B_d/B = (Δ²/2 + A*Δ³/6 + A²*Δ⁴/24 + A³*Δ⁵/120)
            //
            // For this kernel, B is factored out and applied at output (as BC_val) for non-variable B.
            // So we compute: B_d_u = coeff * u (without B, like ZOH computes delta*u)
            // 
            // For non-variable B: delta_u_val = delta * u, so u = delta_u_val / delta
            //   Return: coeff * u = coeff * delta_u_val / delta
            // For variable B: delta_u_val = B * delta * u
            //   The B is already included, so we need: coeff * B * u = coeff * delta_u_val / delta
            // Both cases: B_d_u = coeff * delta_u_val / delta
            
            constexpr float kLog2e = M_LOG2E;
            float A_val_scaled = A_val * kLog2e;
            A_d = exp2f(delta_val * A_val_scaled);
            
            // Compute powers of delta
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            float delta_5th = delta_4th * delta_val;
            
            // Compute A powers
            float A_sq = A_val * A_val;
            float A_cubed = A_sq * A_val;
            
            // B_d/B coefficient (without B, which is handled at output)
            float coeff = delta_sq / 2.0f +
                          A_val * delta_cubed / 6.0f +
                          A_sq * delta_4th / 24.0f +
                          A_cubed * delta_5th / 120.0f;
            
            // Handle edge case when delta is very small
            if (fabsf(delta_val) < 1e-8f) {
                // Limit as delta -> 0: coeff -> 0, so use first-order term
                B_d_u = delta_sq / 2.0f * delta_u_val / delta_val;
            } else {
                // B_d_u = coeff * u (where u is extracted from delta_u_val)
                B_d_u = coeff * delta_u_val / delta_val;
            }
            break;
        }

        case DISCRETIZATION_POLY:
        {
            // POLY: A_d = exp(A*delta), B_d = delta*B + delta^2*A*B/2 + delta^3*A^2*B/6
            constexpr float kLog2e = M_LOG2E;
            float A_val_scaled = A_val * kLog2e;
            A_d = exp2f(delta_val * A_val_scaled);
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float A_B = A_val * B_val;
            float A2_B = A_val * A_B;
            B_d_u = delta_u_val +
                    delta_sq * A_B / 2.0f +
                    delta_cubed * A2_B / 6.0f;
            break;
        }

        case DISCRETIZATION_HIGHORDER:
        {
            // HIGHORDER: Similar to FOH but with higher-order terms
            constexpr float kLog2e = M_LOG2E;
            float A_val_scaled = A_val * kLog2e;
            A_d = exp2f(delta_val * A_val_scaled);
            float delta_sq = delta_val * delta_val;
            float A_B = A_val * B_val;
            if (fabsf(A_val) > 1e-8f)
            {
                B_d_u = delta_u_val + delta_sq * A_B / 2.0f;
            }
            else
            {
                B_d_u = delta_u_val;
            }
            break;
        }

        case DISCRETIZATION_RK4:
        {
            // RK4: A_d = exp(A*delta), B_d uses RK4 coefficients
            constexpr float kLog2e = M_LOG2E;
            float A_val_scaled = A_val * kLog2e;
            A_d = exp2f(delta_val * A_val_scaled);
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            float A_B = A_val * B_val;
            float A2_B = A_val * A_B;
            float A3_B = A_val * A2_B;
            B_d_u = delta_u_val +
                    delta_sq * A_B / 2.0f +
                    delta_cubed * A2_B / 6.0f +
                    delta_4th * A3_B / 24.0f;
            break;
        }

        case DISCRETIZATION_BILINEAR:
        default:
        {
            // BILINEAR: For scalar case, A_d = (1 - A*delta/2) / (1 + A*delta/2)
            // B_d = delta * B / (1 + A*delta/2)
            // This is a simplified version for scalar A
            float half_delta_A = delta_val * A_val * 0.5f;
            float denom = 1.0f + half_delta_A;
            A_d = (1.0f - half_delta_A) / denom;
            B_d_u = delta_u_val / denom;
            break;
        }
        }

        return make_float2(A_d, B_d_u);
    }
    else
    {
        // Complex case - similar logic but with complex arithmetic
        complex_t A_d, B_d_u_complex;

        switch (method)
        {
        case DISCRETIZATION_ZOH:
        {
            constexpr float kLog2e = M_LOG2E;
            A_d = cexp2f(complex_t(delta_val * A_val.real_ * kLog2e, delta_val * A_val.imag_ * kLog2e));
            B_d_u_complex = complex_t(delta_u_val, 0.0f) * B_val;
            break;
        }

        case DISCRETIZATION_FOH:
        {
            // FOH for complex A using Taylor series expansion
            // B_d/B = (Δ²/2 + A*Δ³/6 + A²*Δ⁴/24 + A³*Δ⁵/120)
            constexpr float kLog2e = M_LOG2E;
            complex_t delta_A = complex_t(delta_val * A_val.real_, delta_val * A_val.imag_);
            A_d = cexp2f(complex_t(delta_A.real_ * kLog2e, delta_A.imag_ * kLog2e));
            
            // Compute powers of delta
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            float delta_5th = delta_4th * delta_val;
            
            // Compute A powers (complex multiplication)
            complex_t A_sq = A_val * A_val;
            complex_t A_cubed = A_sq * A_val;
            
            // B_d/B coefficient
            complex_t coeff = complex_t(delta_sq / 2.0f, 0.0f) +
                              A_val * complex_t(delta_cubed / 6.0f, 0.0f) +
                              A_sq * complex_t(delta_4th / 24.0f, 0.0f) +
                              A_cubed * complex_t(delta_5th / 120.0f, 0.0f);
            
            // B_d_u = coeff * u (B handled at output)
            B_d_u_complex = coeff * complex_t(delta_u_val / delta_val, 0.0f);
            break;
        }

        case DISCRETIZATION_POLY:
        case DISCRETIZATION_HIGHORDER:
        case DISCRETIZATION_RK4:
        case DISCRETIZATION_BILINEAR:
        default:
        {
            // For complex, fall back to ZOH for now
            // Full implementation would require complex matrix operations
            constexpr float kLog2e = M_LOG2E;
            A_d = cexp2f(complex_t(delta_val * A_val.real_ * kLog2e, delta_val * A_val.imag_ * kLog2e));
            B_d_u_complex = complex_t(delta_u_val, 0.0f) * B_val;
            break;
        }
        }

        return make_float4(A_d.real_, A_d.imag_, B_d_u_complex.real_, B_d_u_complex.imag_);
    }
}
