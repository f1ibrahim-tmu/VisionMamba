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
            // As delta -> 0, coeff -> 0, so B_d_u -> 0
            // Use safe computation to avoid division by zero
            if (fabsf(delta_val) < 1e-8f) {
                // Limit as delta -> 0: B_d_u -> 0
                B_d_u = 0.0f;  // Safe: FOH term vanishes as delta -> 0
            } else {
                // B_d_u = coeff * u (where u is extracted from delta_u_val)
                B_d_u = coeff * delta_u_val / delta_val;
            }
            break;
        }

        case DISCRETIZATION_POLY:
        {
            // POLY (Polynomial Interpolation): 
            // Correct formula: B̄ = A⁻¹(exp(AΔ)-I)B + ½A⁻²(exp(AΔ)-I-AΔ)B
            // Using Taylor expansion to avoid division:
            // A⁻¹(exp(AΔ)-I) = Δ + AΔ²/2 + A²Δ³/6 + ...
            // ½A⁻²(exp(AΔ)-I-AΔ) = ½(Δ²/2 + AΔ³/6 + A²Δ⁴/24 + ...) = Δ²/4 + AΔ³/12 + ...
            // Combined: B̄ = ΔB + (A/2 + 1/4)Δ²B + (A²/6 + A/12)Δ³B + ...
            constexpr float kLog2e = M_LOG2E;
            float A_val_scaled = A_val * kLog2e;
            A_d = exp2f(delta_val * A_val_scaled);
            
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            
            // ZOH Taylor terms: Δ + AΔ²/2 + A²Δ³/6 + A³Δ⁴/24
            // + ½ FOH Taylor terms: Δ²/4 + AΔ³/12 + A²Δ⁴/48
            // Combined coefficients (factoring out B which is handled at output):
            // Δ term: 1
            // Δ² term: A/2 + 1/4
            // Δ³ term: A²/6 + A/12
            // Δ⁴ term: A³/24 + A²/48
            
            float A_sq = A_val * A_val;
            float A_cubed = A_sq * A_val;
            
            // For non-variable B: delta_u_val = delta * u
            // We want: (Δ + (A/2 + 1/4)Δ² + (A²/6 + A/12)Δ³ + ...) * B * u
            // = delta_u_val + (A/2 + 1/4) * Δ² * B * u + ...
            // Since B is factored out at output, we compute coeff * u where u = delta_u_val / delta / B
            // Actually for consistency with other methods, let's compute directly
            
            float coeff_delta = 1.0f;  // coefficient for Δ term
            float coeff_delta2 = A_val / 2.0f + 0.25f;  // coefficient for Δ² term
            float coeff_delta3 = A_sq / 6.0f + A_val / 12.0f;  // coefficient for Δ³ term
            float coeff_delta4 = A_cubed / 24.0f + A_sq / 48.0f;  // coefficient for Δ⁴ term
            
            // Simplified: extract u = delta_u_val / delta_val (for non-variable B case)
            // For variable B case, delta_u_val = B * delta * u, so delta_u_val / delta_val = B * u
            // Guard against division by zero
            if (fabsf(delta_val) > 1e-8f) {
                float u_factor = delta_u_val / delta_val;  // This is either 'u' or 'B*u'
                B_d_u = delta_u_val +  // Δ * (B * u) or Δ * u
                        delta_sq * coeff_delta2 * u_factor +
                        delta_cubed * coeff_delta3 * u_factor +
                        delta_4th * coeff_delta4 * u_factor;
            } else {
                // As delta -> 0, higher-order terms vanish, keep only first-order term
                B_d_u = delta_u_val;
            }
            break;
        }

        case DISCRETIZATION_HIGHORDER:
        {
            // HIGHER-ORDER HOLD (n=2, Quadratic)
            // Generalized formula: B̄ = Σ(i=0 to n) A^(-(i+1)) * [exp(AΔ) - Σ(k=0 to i)(AΔ)^k/k!] / i! * B
            // For n=2: B̄ = ZOH_B + FOH_B + (1/2!)×[A⁻³(exp(AΔ) - I - AΔ - (AΔ)²/2)]B
            // 
            // Using Taylor expansion:
            // n=0 (ZOH): A⁻¹(exp(AΔ)-I) = Δ + AΔ²/2 + A²Δ³/6 + A³Δ⁴/24 + ...
            // n=1 (FOH): A⁻²(exp(AΔ)-I-AΔ) = Δ²/2 + AΔ³/6 + A²Δ⁴/24 + ...
            // n=2: A⁻³(exp(AΔ)-I-AΔ-(AΔ)²/2)/2 = (Δ³/6 + AΔ⁴/24 + ...)/2 = Δ³/12 + AΔ⁴/48 + ...
            //
            // Combined (n=2): Δ + (A/2)Δ² + (A²/6 + 1/2)Δ² + (A²/6 + A/12)Δ³ + (Δ³/12)
            //               = Δ + (A/2 + 1/2)Δ² + (A²/6 + A/6 + 1/12)Δ³ + ...
            
            constexpr float kLog2e = M_LOG2E;
            float A_val_scaled = A_val * kLog2e;
            A_d = exp2f(delta_val * A_val_scaled);
            
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            float delta_5th = delta_4th * delta_val;
            
            float A_sq = A_val * A_val;
            float A_cubed = A_sq * A_val;
            
            // Coefficients for Higher-Order Hold (n=2):
            // Δ term: 1 (from ZOH)
            // Δ² term: A/2 (from ZOH) + 1/2 (from FOH) = A/2 + 1/2
            // Δ³ term: A²/6 (from ZOH) + A/6 (from FOH) + 1/12 (from n=2) = A²/6 + A/6 + 1/12
            // Δ⁴ term: A³/24 (from ZOH) + A²/24 (from FOH) + A/48 (from n=2) = A³/24 + A²/24 + A/48
            // Δ⁵ term: A⁴/120 + A³/120 + A²/240
            
            float coeff_delta2 = A_val / 2.0f + 0.5f;
            float coeff_delta3 = A_sq / 6.0f + A_val / 6.0f + 1.0f / 12.0f;
            float coeff_delta4 = A_cubed / 24.0f + A_sq / 24.0f + A_val / 48.0f;
            
            if (fabsf(delta_val) > 1e-8f) {
                float u_factor = delta_u_val / delta_val;
                B_d_u = delta_u_val +
                        delta_sq * coeff_delta2 * u_factor +
                        delta_cubed * coeff_delta3 * u_factor +
                        delta_4th * coeff_delta4 * u_factor;
            } else {
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
            // BILINEAR (Tustin Transform) - Correct stability-preserving formula:
            // Ā = (I - ΔA/2)⁻¹(I + ΔA/2)
            // B̄ = (I - ΔA/2)⁻¹ΔB
            //
            // For scalar case:
            // A_d = (1 + A*delta/2) / (1 - A*delta/2)
            // B_d = delta * B / (1 - A*delta/2)
            //
            // This ensures stability: for A < 0 (stable continuous), |A_d| < 1 (stable discrete)
            float half_delta_A = delta_val * A_val * 0.5f;
            float denom = 1.0f - half_delta_A;  // (I - ΔA/2) for scalar
            
            // Handle edge case when denominator is near zero
            if (fabsf(denom) < 1e-8f) {
                // Fall back to ZOH approximation
                constexpr float kLog2e = M_LOG2E;
                A_d = exp2f(delta_val * A_val * kLog2e);
                B_d_u = delta_u_val;
            } else {
                A_d = (1.0f + half_delta_A) / denom;
                B_d_u = delta_u_val / denom;
            }
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
            // delta_u_val is already complex_t in complex case, so use it directly
            B_d_u_complex = delta_u_val * B_val;
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
            // delta_u_val is complex, extract u by dividing by delta
            // Handle edge case when delta is very small to avoid division by zero
            if (fabsf(delta_val) < 1e-8f) {
                // Limit as delta -> 0: B_d_u -> 0
                B_d_u_complex = complex_t(0.0f, 0.0f);  // Safe: FOH term vanishes as delta -> 0
            } else {
                complex_t u_factor = delta_u_val * complex_t(1.0f / delta_val, 0.0f);
                B_d_u_complex = coeff * u_factor;
            }
            break;
        }

        case DISCRETIZATION_POLY:
        {
            // POLY for complex A using Taylor series
            constexpr float kLog2e = M_LOG2E;
            complex_t delta_A = complex_t(delta_val * A_val.real_, delta_val * A_val.imag_);
            A_d = cexp2f(complex_t(delta_A.real_ * kLog2e, delta_A.imag_ * kLog2e));
            
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            
            complex_t A_sq = A_val * A_val;
            complex_t A_cubed = A_sq * A_val;
            
            // Coefficients: Δ² term = A/2 + 1/4, Δ³ term = A²/6 + A/12
            complex_t coeff_delta2 = A_val * complex_t(0.5f, 0.0f) + complex_t(0.25f, 0.0f);
            complex_t coeff_delta3 = A_sq * complex_t(1.0f/6.0f, 0.0f) + A_val * complex_t(1.0f/12.0f, 0.0f);
            complex_t coeff_delta4 = A_cubed * complex_t(1.0f/24.0f, 0.0f) + A_sq * complex_t(1.0f/48.0f, 0.0f);
            
            // delta_u_val is complex, extract u by dividing by delta
            // Guard against division by zero
            if (fabsf(delta_val) > 1e-8f) {
                complex_t u_factor = delta_u_val * complex_t(1.0f / delta_val, 0.0f);
                B_d_u_complex = delta_u_val +
                                coeff_delta2 * complex_t(delta_sq, 0.0f) * u_factor +
                                coeff_delta3 * complex_t(delta_cubed, 0.0f) * u_factor +
                                coeff_delta4 * complex_t(delta_4th, 0.0f) * u_factor;
            } else {
                // As delta -> 0, higher-order terms vanish, keep only first-order term
                B_d_u_complex = delta_u_val;
            }
            break;
        }
        
        case DISCRETIZATION_HIGHORDER:
        {
            // HIGHER-ORDER for complex A (n=2)
            constexpr float kLog2e = M_LOG2E;
            complex_t delta_A = complex_t(delta_val * A_val.real_, delta_val * A_val.imag_);
            A_d = cexp2f(complex_t(delta_A.real_ * kLog2e, delta_A.imag_ * kLog2e));
            
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            
            complex_t A_sq = A_val * A_val;
            complex_t A_cubed = A_sq * A_val;
            
            complex_t coeff_delta2 = A_val * complex_t(0.5f, 0.0f) + complex_t(0.5f, 0.0f);
            complex_t coeff_delta3 = A_sq * complex_t(1.0f/6.0f, 0.0f) + A_val * complex_t(1.0f/6.0f, 0.0f) + complex_t(1.0f/12.0f, 0.0f);
            complex_t coeff_delta4 = A_cubed * complex_t(1.0f/24.0f, 0.0f) + A_sq * complex_t(1.0f/24.0f, 0.0f) + A_val * complex_t(1.0f/48.0f, 0.0f);
            
            // delta_u_val is complex, extract u by dividing by delta
            // Guard against division by zero
            if (fabsf(delta_val) > 1e-8f) {
                complex_t u_factor = delta_u_val * complex_t(1.0f / delta_val, 0.0f);
                B_d_u_complex = delta_u_val +
                                coeff_delta2 * complex_t(delta_sq, 0.0f) * u_factor +
                                coeff_delta3 * complex_t(delta_cubed, 0.0f) * u_factor +
                                coeff_delta4 * complex_t(delta_4th, 0.0f) * u_factor;
            } else {
                // As delta -> 0, higher-order terms vanish, keep only first-order term
                B_d_u_complex = delta_u_val;
            }
            break;
        }
        
        case DISCRETIZATION_BILINEAR:
        {
            // BILINEAR for complex A
            // Ā = (I - ΔA/2)⁻¹(I + ΔA/2)
            complex_t half_delta_A = A_val * complex_t(delta_val * 0.5f, 0.0f);
            complex_t one = complex_t(1.0f, 0.0f);
            complex_t numer = one + half_delta_A;  // (1 + ΔA/2)
            complex_t denom = one - half_delta_A;  // (1 - ΔA/2)
            
            // Complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
            float denom_mag_sq = denom.real_ * denom.real_ + denom.imag_ * denom.imag_;
            if (denom_mag_sq > 1e-16f) {
                complex_t denom_inv = complex_t(denom.real_ / denom_mag_sq, -denom.imag_ / denom_mag_sq);
                A_d = numer * denom_inv;
                // delta_u_val is already complex_t in complex case
                B_d_u_complex = delta_u_val * denom_inv;
            } else {
                // Fall back to ZOH
                constexpr float kLog2e = M_LOG2E;
                A_d = cexp2f(complex_t(delta_val * A_val.real_ * kLog2e, delta_val * A_val.imag_ * kLog2e));
                // delta_u_val is already complex_t in complex case
                B_d_u_complex = delta_u_val * B_val;
            }
            break;
        }
        
        case DISCRETIZATION_RK4:
        default:
        {
            // For RK4 complex, use Taylor expansion like real case
            constexpr float kLog2e = M_LOG2E;
            complex_t delta_A = complex_t(delta_val * A_val.real_, delta_val * A_val.imag_);
            A_d = cexp2f(complex_t(delta_A.real_ * kLog2e, delta_A.imag_ * kLog2e));
            
            float delta_sq = delta_val * delta_val;
            float delta_cubed = delta_sq * delta_val;
            float delta_4th = delta_cubed * delta_val;
            
            complex_t A_sq = A_val * A_val;
            complex_t A_cubed = A_sq * A_val;
            
            // delta_u_val is complex, extract u by dividing by delta
            // Guard against division by zero
            if (fabsf(delta_val) > 1e-8f) {
                complex_t u_factor = delta_u_val * complex_t(1.0f / delta_val, 0.0f);
                B_d_u_complex = delta_u_val +
                                A_val * complex_t(delta_sq / 2.0f, 0.0f) * u_factor +
                                A_sq * complex_t(delta_cubed / 6.0f, 0.0f) * u_factor +
                                A_cubed * complex_t(delta_4th / 24.0f, 0.0f) * u_factor;
            } else {
                // As delta -> 0, higher-order terms vanish, keep only first-order term
                B_d_u_complex = delta_u_val;
            }
            break;
        }
        }

        return make_float4(A_d.real_, A_d.imag_, B_d_u_complex.real_, B_d_u_complex.imag_);
    }
}
