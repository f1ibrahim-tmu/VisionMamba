# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
# Compatibility: custom_bwd/custom_fwd moved from torch.cuda.amp to torch.amp in PyTorch 2.1.0
try:
    from torch.amp import custom_bwd, custom_fwd
    _has_device_type_arg = True
except ImportError:
    from torch.cuda.amp import custom_bwd, custom_fwd
    _has_device_type_arg = False

# Compatibility wrappers for device_type argument (only supported in PyTorch 2.1.0+)
# In older PyTorch, custom_fwd/custom_bwd are decorators themselves, not decorator factories
if _has_device_type_arg:
    # PyTorch >= 2.1.0: custom_fwd/custom_bwd are decorator factories that accept device_type
    def _custom_fwd(*args, **kwargs):
        return custom_fwd(*args, **kwargs)
    
    def _custom_bwd(*args, **kwargs):
        return custom_bwd(*args, **kwargs)
else:
    # PyTorch < 2.1.0: custom_fwd/custom_bwd are decorators themselves
    # We need to return them directly, ignoring any arguments
    def _custom_fwd(*args, **kwargs):
        # Ignore all arguments and return the decorator directly
        return custom_fwd
    
    def _custom_bwd(*args, **kwargs):
        # Ignore all arguments and return the decorator directly
        return custom_bwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

import selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, discretization_method="zoh", use_cuda_kernel=None):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        
        # Determine which implementation to use
        # use_cuda_kernel: True = force CUDA, False = force Python, None = auto (try CUDA first)
        force_cuda = use_cuda_kernel is True
        force_python = use_cuda_kernel is False
        auto_select = use_cuda_kernel is None
        
        # Polynomial Interpolation requires bidirectional (non-causal) scan,
        # which is only implemented in the Python reference. Force Python path.
        if discretization_method == "poly":
            force_python = True
            force_cuda = False
            auto_select = False
        
        # Map discretization method string to enum value
        disc_method_map = {
            "zoh": 0,
            "foh": 1,
            "bilinear": 2,
            "poly": 3,
            "highorder": 4,
            "rk4": 5
        }
        disc_method_enum = disc_method_map.get(discretization_method, 0)
        
        # Try CUDA kernel if requested (force or auto-select)
        if (force_cuda or auto_select) and selective_scan_cuda is not None:
            try:
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, disc_method_enum)
                ctx.delta_softplus = delta_softplus
                ctx.has_z = z is not None
                ctx.discretization_method = discretization_method
                ctx.use_cuda_kernel = True
                last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
                if not ctx.has_z:
                    ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
                    return out if not return_last_state else (out, last_state)
                else:
                    ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
                    out_z = rest[0]
                    return out_z if not return_last_state else (out_z, last_state)
            except Exception as e:
                # Fall back to reference implementation if CUDA kernel fails
                if force_cuda:
                    # If CUDA was forced but failed, raise the error
                    raise RuntimeError(f"CUDA kernel failed for {discretization_method} (use_cuda_kernel=True): {e}")
                # Otherwise, fall through to Python reference
                if auto_select:
                    # Only print warning in auto-select mode
                    import warnings
                    warnings.warn(f"CUDA kernel failed for {discretization_method}, falling back to Python reference: {e}", UserWarning)
        
        # Use Python reference implementation (either forced or as fallback)
        result = selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, 
                                  return_last_state, discretization_method)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.discretization_method = discretization_method
        ctx.use_cuda_kernel = False
        if not return_last_state:
            return result
        else:
            out, last_state = result
            return out, last_state
        if not return_last_state:
            return result
        else:
            out, last_state = result
            return out, last_state

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        # If using a non-ZOH method and we need to implement backward, this would be handled here
        if hasattr(ctx, 'discretization_method') and ctx.discretization_method != "zoh":
            # For now, this is not implemented, so we'll use the default backward
            pass
            
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None)  # Return None for discretization_method


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False, discretization_method="zoh", use_cuda_kernel=None):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    
    Args:
        discretization_method: str, one of ["zoh", "foh", "poly", "bilinear", "highorder", "rk4"]
            zoh: Zero Order Hold (default in original implementation)
            foh: First Order Hold
            poly: Polynomial interpolation
            bilinear: Bilinear (Tustin) Transform
            highorder: Higher-order hold
            rk4: Runge-Kutta 4th order
        use_cuda_kernel: bool or None
            If True: Force use of CUDA kernel (if available)
            If False: Force use of Python reference implementation
            If None: Auto-select (try CUDA first, fallback to Python)
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state, discretization_method, use_cuda_kernel)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False, discretization_method="zoh"):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    discretization_method: str, one of ["zoh", "foh", "poly", "bilinear", "highorder"]
        zoh: Zero Order Hold (default in original implementation)
        foh: First Order Hold
        poly: Polynomial interpolation
        bilinear: Bilinear (Tustin) Transform
        highorder: Higher-order hold

    out: r(B D L)
    last_state: r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    
    # Discretization methods
    if discretization_method == "zoh":
        # Zero Order Hold (original implementation)
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    elif discretization_method == "foh":
        # First Order Hold (Correct Formula)
        # For FOH: A_d = exp(A*delta), B_d = A^(-2) * (exp(A*delta) - I - A*delta) * B
        # Using Taylor series expansion to avoid division:
        # (exp(A*Δ) - 1 - A*Δ) / A^2 = Δ²/2! + A*Δ³/3! + A²*Δ⁴/4! + A³*Δ⁵/5! + ...
        # So: B_d = (Δ²/2 + A*Δ³/6 + A²*Δ⁴/24 + A³*Δ⁵/120) * B
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        
        # Compute powers of delta (bdl shape)
        delta_sq = delta ** 2
        delta_cubed = delta ** 3
        delta_4th = delta ** 4
        delta_5th = delta ** 5
        
        if not is_variable_B:
            # B_d * u = (Δ²/2 * B + A*Δ³/6 * B + A²*Δ⁴/24 * B + A³*Δ⁵/120 * B) * u
            # Compute coefficient: Δ²/2 + A*Δ³/6 + A²*Δ⁴/24 + A³*Δ⁵/120 for each (b,d,l,n)
            # Then multiply by B and u
            
            # Expand delta powers to (B, D, L, 1) for broadcasting with A (D, N)
            delta_sq_exp = delta_sq.unsqueeze(-1)  # (B, D, L, 1)
            delta_cubed_exp = delta_cubed.unsqueeze(-1)
            delta_4th_exp = delta_4th.unsqueeze(-1)
            delta_5th_exp = delta_5th.unsqueeze(-1)
            
            A_exp = A.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)
            A_sq = A ** 2
            A_cubed = A ** 3
            A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
            A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
            B_exp = B.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)
            
            # coeff * B = (Δ²/2 + A*Δ³/6 + A²*Δ⁴/24 + A³*Δ⁵/120) * B
            deltaB = (delta_sq_exp / 2.0 * B_exp +
                      delta_cubed_exp / 6.0 * A_exp * B_exp +
                      delta_4th_exp / 24.0 * A_sq_exp * B_exp +
                      delta_5th_exp / 120.0 * A_cubed_exp * B_exp)  # (B, D, L, N)
            
            deltaB_u = deltaB * u.unsqueeze(-1)  # (B, D, L, N)
        else:
            if B.dim() == 3:
                # B is (B, N, L)
                delta_sq_exp = delta_sq.unsqueeze(-1)  # (B, D, L, 1)
                delta_cubed_exp = delta_cubed.unsqueeze(-1)
                delta_4th_exp = delta_4th.unsqueeze(-1)
                delta_5th_exp = delta_5th.unsqueeze(-1)
                
                A_exp = A.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)
                A_sq = A ** 2
                A_cubed = A ** 3
                A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
                A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
                
                # B: (B, N, L) -> (B, 1, L, N) for broadcasting
                B_exp = B.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, L, N)
                
                deltaB = (delta_sq_exp / 2.0 * B_exp +
                          delta_cubed_exp / 6.0 * A_exp * B_exp +
                          delta_4th_exp / 24.0 * A_sq_exp * B_exp +
                          delta_5th_exp / 120.0 * A_cubed_exp * B_exp)  # (B, D, L, N)
                
                deltaB_u = deltaB * u.unsqueeze(-1)  # (B, D, L, N)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                
                delta_sq_exp = delta_sq.unsqueeze(-1)
                delta_cubed_exp = delta_cubed.unsqueeze(-1)
                delta_4th_exp = delta_4th.unsqueeze(-1)
                delta_5th_exp = delta_5th.unsqueeze(-1)
                
                A_exp = A.unsqueeze(0).unsqueeze(2)
                A_sq = A ** 2
                A_cubed = A ** 3
                A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
                A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
                
                # B: (B, D, N, L) -> (B, D, L, N) for proper broadcasting
                B_exp = B.permute(0, 1, 3, 2)  # (B, D, L, N)
                
                deltaB = (delta_sq_exp / 2.0 * B_exp +
                          delta_cubed_exp / 6.0 * A_exp * B_exp +
                          delta_4th_exp / 24.0 * A_sq_exp * B_exp +
                          delta_5th_exp / 120.0 * A_cubed_exp * B_exp)
                
                deltaB_u = deltaB * u.unsqueeze(-1)
    
    elif discretization_method == "bilinear":
        # Bilinear (Tustin) Transform - Correct stability-preserving formula:
        # Ā = (I - ΔA/2)⁻¹(I + ΔA/2)
        # B̄ = (I - ΔA/2)⁻¹ΔB
        # This ensures stability: left half-plane maps to inside unit circle
        
        # Compute half_delta_A as a vector: (batch, dim, seqlen, dstate)
        half_delta_A_vec = torch.einsum('bdl,dn->bdln', delta, A) * 0.5
        
        # Convert to diagonal matrix: (batch, dim, seqlen, dstate, dstate)
        # Each (b, d, l) gets a diagonal matrix with half_delta_A_vec[b, d, l, :] on the diagonal
        half_delta_A = torch.diag_embed(half_delta_A_vec, dim1=-2, dim2=-1)
        
        # Create identity matrix with correct shape: (1, 1, 1, dstate, dstate) for broadcasting
        I = torch.eye(dstate, device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # I now has shape (1, 1, 1, dstate, dstate) which will broadcast to (batch, dim, seqlen, dstate, dstate)
        
        I_plus_half_delta_A = I + half_delta_A   # (I + ΔA/2)
        I_minus_half_delta_A = I - half_delta_A  # (I - ΔA/2)
        
        # Compute (I - A*delta/2)^-1 using batch matrix inverse (CORRECTED: invert I - ΔA/2)
        I_minus_half_delta_A_reshaped = rearrange(I_minus_half_delta_A, 'b d l n1 n2 -> (b d l) n1 n2')
        I_minus_half_delta_A_inv_reshaped = torch.inverse(I_minus_half_delta_A_reshaped)
        I_minus_half_delta_A_inv = rearrange(I_minus_half_delta_A_inv_reshaped, '(b d l) n1 n2 -> b d l n1 n2', 
                                          b=batch, d=dim, l=delta.size(2))
        
        # A_d = (I - A*delta/2)^-1 * (I + A*delta/2) (CORRECTED order)
        deltaA = torch.matmul(I_minus_half_delta_A_inv, I_plus_half_delta_A)
        
        # Compute B_d = (I - ΔA/2)⁻¹ΔB
        # For bilinear: B_d = (I - ΔA/2)⁻¹ * (delta * B) where delta is scalar and B is vector
        if not is_variable_B:
            # B is (dim, dstate)
            # For each (b, d, l), we have delta[b, d, l] (scalar) and B[d, :] (vector of shape dstate)
            # delta * B should be (batch, dim, seqlen, dstate, 1)
            # Expand B: (dim, dstate) -> (1, dim, 1, dstate, 1)
            B_expanded = B.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, dim, 1, dstate, 1)
            # Expand delta: (batch, dim, seqlen) -> (batch, dim, seqlen, 1, 1)
            delta_expanded = delta.unsqueeze(-1).unsqueeze(-1)  # (batch, dim, seqlen, 1, 1)
            # delta * B: (batch, dim, seqlen, 1, 1) * (1, dim, 1, dstate, 1) -> (batch, dim, seqlen, dstate, 1)
            # But we need to broadcast properly - B should be (1, dim, 1, dstate, 1) and delta (batch, dim, seqlen, 1, 1)
            # Actually, we need to match dimensions: B[d, :] for each d, so B should be (1, dim, 1, dstate, 1)
            delta_B = delta_expanded * B_expanded  # (batch, dim, seqlen, dstate, 1)
            # Now multiply by inverse: (I - ΔA/2)⁻¹ * (delta * B)
            deltaB = torch.matmul(I_minus_half_delta_A_inv, delta_B)  # (batch, dim, seqlen, dstate, 1)
            # Multiply by u: u has shape (batch, dim, seqlen), expand to (batch, dim, seqlen, 1)
            deltaB_u = deltaB.squeeze(-1) * u.unsqueeze(-1)  # (batch, dim, seqlen, dstate)
        else:
            # Handle variable B case 
            if B.dim() == 3:
                # B is (batch, dstate, seqlen) -> transpose to (batch, seqlen, dstate)
                # Need to expand to (batch, dim, seqlen, dstate, 1)
                B_transposed = B.permute(0, 2, 1)  # (batch, seqlen, dstate)
                B_expanded = B_transposed.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seqlen, dstate, 1)
                # Expand to match dim dimension
                B_expanded = repeat(B_expanded, 'b 1 l n 1 -> b d l n 1', d=dim)
                delta_expanded = delta.unsqueeze(-1).unsqueeze(-1)  # (batch, dim, seqlen, 1, 1)
                delta_B = delta_expanded * B_expanded  # (batch, dim, seqlen, dstate, 1)
                deltaB = torch.matmul(I_minus_half_delta_A_inv, delta_B)
                deltaB_u = deltaB.squeeze(-1) * u.unsqueeze(-1)  # (batch, dim, seqlen, dstate)
            else:
                # B is (batch, n_groups, dstate, seqlen)
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                # B is now (batch, dim, dstate, seqlen) -> permute to (batch, dim, seqlen, dstate)
                B_permuted = B.permute(0, 1, 3, 2)  # (batch, dim, seqlen, dstate)
                B_expanded = B_permuted.unsqueeze(-1)  # (batch, dim, seqlen, dstate, 1)
                delta_expanded = delta.unsqueeze(-1).unsqueeze(-1)  # (batch, dim, seqlen, 1, 1)
                delta_B = delta_expanded * B_expanded  # (batch, dim, seqlen, dstate, 1)
                deltaB = torch.matmul(I_minus_half_delta_A_inv, delta_B)
                deltaB_u = deltaB.squeeze(-1) * u.unsqueeze(-1)  # (batch, dim, seqlen, dstate)
    
    elif discretization_method == "poly":
        # Polynomial Interpolation (Non-Causal, Bidirectional):
        # B̄ = A⁻¹(exp(AΔ)-I)B + ½A⁻²(exp(AΔ)-I-AΔ)B
        # Using Taylor expansion to avoid division:
        # ZOH term: A⁻¹(exp(AΔ)-I) = Δ + AΔ²/2 + A²Δ³/6 + A³Δ⁴/24
        # ½FOH term: ½A⁻²(exp(AΔ)-I-AΔ) = Δ²/4 + AΔ³/12 + A²Δ⁴/48
        # Combined: B̄ = (Δ + (A/2 + 1/4)Δ² + (A²/6 + A/12)Δ³ + (A³/24 + A²/48)Δ⁴) * B
        # 
        # NOTE: Polynomial Interpolation is NON-CAUSAL - it uses bidirectional scan
        # to access both past and future information, creating smooth interpolation
        # between points (like bicubic interpolation in image resizing)
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        
        # Compute powers of delta
        delta_sq = delta ** 2
        delta_cubed = delta ** 3
        delta_4th = delta ** 4
        
        if not is_variable_B:
            # Expand delta powers for broadcasting
            delta_exp = delta.unsqueeze(-1)  # (B, D, L, 1)
            delta_sq_exp = delta_sq.unsqueeze(-1)
            delta_cubed_exp = delta_cubed.unsqueeze(-1)
            delta_4th_exp = delta_4th.unsqueeze(-1)
            
            A_exp = A.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)
            A_sq = A ** 2
            A_cubed = A ** 3
            A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
            A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
            B_exp = B.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)
            
            # Coefficients for polynomial: Δ + (A/2 + 1/4)Δ² + (A²/6 + A/12)Δ³ + (A³/24 + A²/48)Δ⁴
            term1 = delta_exp * B_exp  # Δ * B
            term2 = delta_sq_exp * (A_exp / 2.0 + 0.25) * B_exp  # (A/2 + 1/4)Δ² * B
            term3 = delta_cubed_exp * (A_sq_exp / 6.0 + A_exp / 12.0) * B_exp  # (A²/6 + A/12)Δ³ * B
            term4 = delta_4th_exp * (A_cubed_exp / 24.0 + A_sq_exp / 48.0) * B_exp  # (A³/24 + A²/48)Δ⁴ * B
            
            deltaB = term1 + term2 + term3 + term4  # (B, D, L, N)
            deltaB_u = deltaB * u.unsqueeze(-1)
        else:
            # Handle the variable B case
            if B.dim() == 3:
                # B is (B, N, L)
                delta_exp = delta.unsqueeze(-1)
                delta_sq_exp = delta_sq.unsqueeze(-1)
                delta_cubed_exp = delta_cubed.unsqueeze(-1)
                delta_4th_exp = delta_4th.unsqueeze(-1)
                
                A_exp = A.unsqueeze(0).unsqueeze(2)
                A_sq = A ** 2
                A_cubed = A ** 3
                A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
                A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
                
                # B: (B, N, L) -> (B, 1, L, N)
                B_exp = B.unsqueeze(1).permute(0, 1, 3, 2)
                
                term1 = delta_exp * B_exp
                term2 = delta_sq_exp * (A_exp / 2.0 + 0.25) * B_exp
                term3 = delta_cubed_exp * (A_sq_exp / 6.0 + A_exp / 12.0) * B_exp
                term4 = delta_4th_exp * (A_cubed_exp / 24.0 + A_sq_exp / 48.0) * B_exp
                
                deltaB = term1 + term2 + term3 + term4
                deltaB_u = deltaB * u.unsqueeze(-1)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                
                delta_exp = delta.unsqueeze(-1)
                delta_sq_exp = delta_sq.unsqueeze(-1)
                delta_cubed_exp = delta_cubed.unsqueeze(-1)
                delta_4th_exp = delta_4th.unsqueeze(-1)
                
                A_exp = A.unsqueeze(0).unsqueeze(2)
                A_sq = A ** 2
                A_cubed = A ** 3
                A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
                A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
                
                B_exp = B.permute(0, 1, 3, 2)
                
                term1 = delta_exp * B_exp
                term2 = delta_sq_exp * (A_exp / 2.0 + 0.25) * B_exp
                term3 = delta_cubed_exp * (A_sq_exp / 6.0 + A_exp / 12.0) * B_exp
                term4 = delta_4th_exp * (A_cubed_exp / 24.0 + A_sq_exp / 48.0) * B_exp
                
                deltaB = term1 + term2 + term3 + term4
                deltaB_u = deltaB * u.unsqueeze(-1)
    
    elif discretization_method == "highorder":
        # Higher-Order Hold (n=2, Quadratic) - CAUSAL Method:
        # B̄ = Σ(i=0 to n) A^(-(i+1)) * [exp(AΔ) - Σ(k=0 to i)(AΔ)^k/k!] / i! * B
        # For n=2: Combines ZOH (n=0) + FOH (n=1) + Quadratic (n=2) terms
        #
        # Using Taylor expansion:
        # n=0 (ZOH): Δ + AΔ²/2 + A²Δ³/6 + A³Δ⁴/24
        # n=1 (FOH): Δ²/2 + AΔ³/6 + A²Δ⁴/24
        # n=2: Δ³/12 + AΔ⁴/48
        #
        # Combined coefficients:
        # Δ term: 1
        # Δ² term: A/2 + 1/2
        # Δ³ term: A²/6 + A/6 + 1/12
        # Δ⁴ term: A³/24 + A²/24 + A/48
        #
        # NOTE: HOH is CAUSAL - delta (Δ) is applied at the INPUT/SAMPLING stage.
        # It only uses past information to project forward, like "shooting in the dark"
        # based on momentum from previous points. This can cause overshoot when the
        # signal changes direction suddenly.
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        
        # Compute powers of delta
        delta_sq = delta ** 2
        delta_cubed = delta ** 3
        delta_4th = delta ** 4
        
        if not is_variable_B:
            # Expand delta powers for broadcasting
            delta_exp = delta.unsqueeze(-1)  # (B, D, L, 1)
            delta_sq_exp = delta_sq.unsqueeze(-1)
            delta_cubed_exp = delta_cubed.unsqueeze(-1)
            delta_4th_exp = delta_4th.unsqueeze(-1)
            
            A_exp = A.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)
            A_sq = A ** 2
            A_cubed = A ** 3
            A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
            A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
            B_exp = B.unsqueeze(0).unsqueeze(2)  # (1, D, 1, N)
            
            # Coefficients for higher-order (n=2):
            # Δ + (A/2 + 1/2)Δ² + (A²/6 + A/6 + 1/12)Δ³ + (A³/24 + A²/24 + A/48)Δ⁴
            term1 = delta_exp * B_exp  # Δ * B
            term2 = delta_sq_exp * (A_exp / 2.0 + 0.5) * B_exp  # (A/2 + 1/2)Δ² * B
            term3 = delta_cubed_exp * (A_sq_exp / 6.0 + A_exp / 6.0 + 1.0/12.0) * B_exp
            term4 = delta_4th_exp * (A_cubed_exp / 24.0 + A_sq_exp / 24.0 + A_exp / 48.0) * B_exp
            
            deltaB = term1 + term2 + term3 + term4  # (B, D, L, N)
            deltaB_u = deltaB * u.unsqueeze(-1)
        else:
            if B.dim() == 3:
                # B is (B, N, L)
                delta_exp = delta.unsqueeze(-1)
                delta_sq_exp = delta_sq.unsqueeze(-1)
                delta_cubed_exp = delta_cubed.unsqueeze(-1)
                delta_4th_exp = delta_4th.unsqueeze(-1)
                
                A_exp = A.unsqueeze(0).unsqueeze(2)
                A_sq = A ** 2
                A_cubed = A ** 3
                A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
                A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
                
                B_exp = B.unsqueeze(1).permute(0, 1, 3, 2)
                
                term1 = delta_exp * B_exp
                term2 = delta_sq_exp * (A_exp / 2.0 + 0.5) * B_exp
                term3 = delta_cubed_exp * (A_sq_exp / 6.0 + A_exp / 6.0 + 1.0/12.0) * B_exp
                term4 = delta_4th_exp * (A_cubed_exp / 24.0 + A_sq_exp / 24.0 + A_exp / 48.0) * B_exp
                
                deltaB = term1 + term2 + term3 + term4
                deltaB_u = deltaB * u.unsqueeze(-1)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                
                delta_exp = delta.unsqueeze(-1)
                delta_sq_exp = delta_sq.unsqueeze(-1)
                delta_cubed_exp = delta_cubed.unsqueeze(-1)
                delta_4th_exp = delta_4th.unsqueeze(-1)
                
                A_exp = A.unsqueeze(0).unsqueeze(2)
                A_sq = A ** 2
                A_cubed = A ** 3
                A_sq_exp = A_sq.unsqueeze(0).unsqueeze(2)
                A_cubed_exp = A_cubed.unsqueeze(0).unsqueeze(2)
                
                B_exp = B.permute(0, 1, 3, 2)
                
                term1 = delta_exp * B_exp
                term2 = delta_sq_exp * (A_exp / 2.0 + 0.5) * B_exp
                term3 = delta_cubed_exp * (A_sq_exp / 6.0 + A_exp / 6.0 + 1.0/12.0) * B_exp
                term4 = delta_4th_exp * (A_cubed_exp / 24.0 + A_sq_exp / 24.0 + A_exp / 48.0) * B_exp
                
                deltaB = term1 + term2 + term3 + term4
                deltaB_u = deltaB * u.unsqueeze(-1)
    elif discretization_method == "rk4":
        # Runge-Kutta 4th order discretization
        # For RK4: A_d = exp(A*delta), B_d = (A^-1)*(A_d - I)*B
        # with additional terms for higher accuracy
        
        # Compute A^2 and A^3 for RK4
        A_squared = torch.einsum('dn,dm->dnm', A, A)
        A_cubed = torch.einsum('dnm,dk->dnmk', A_squared, A)
        
        # Compute delta powers
        delta_sq = (delta ** 2).unsqueeze(-1).unsqueeze(-1)
        delta_cubed = (delta ** 3).unsqueeze(-1).unsqueeze(-1)
        
        # Compute A_d using RK4
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        
        if not is_variable_B:
            # Compute B_d using RK4 coefficients
            AB = torch.einsum('dn,dm->dnm', A, B)
            A2B = torch.einsum('dnm,dm->dn', A_squared, B)
            A3B = torch.einsum('dnmk,dk->dn', A_cubed, B)
            
            # RK4 coefficients for B_d
            k1 = delta.unsqueeze(-1) * B.unsqueeze(0).unsqueeze(0)
            k2 = delta_sq * AB.unsqueeze(0).unsqueeze(0) / 2.0
            k3 = delta_cubed * A2B.unsqueeze(0).unsqueeze(0) / 6.0
            k4 = (delta ** 4).unsqueeze(-1).unsqueeze(-1) * A3B.unsqueeze(0).unsqueeze(0) / 24.0
            
            deltaB = k1 + k2 + k3 + k4
            deltaB_u = torch.einsum('bdln,bdl->bdln', deltaB, u)
        else:
            # Handle variable B case
            if B.dim() == 3:
                # B is (batch, dstate, seqlen) = (B, N, L)
                # Need to compute RK4 coefficients for each (b, d, l)
                # For variable B, we need to handle per-sequence B values
                
                # Reshape B to (batch, 1, dstate, seqlen) for broadcasting with dim
                B_expanded = B.unsqueeze(1)  # (batch, 1, dstate, seqlen)
                # Expand to match dim dimension: (batch, dim, dstate, seqlen)
                B_expanded = repeat(B_expanded, 'b 1 n l -> b d n l', d=dim)
                
                # Compute AB, A2B, A3B for variable B
                # A is (dim, dstate), B_expanded is (batch, dim, dstate, seqlen)
                AB = torch.einsum('dn,bdnl->bdln', A, B_expanded)  # (batch, dim, seqlen, dstate)
                A2B = torch.einsum('dnm,bdml->bdln', A_squared, B_expanded)  # (batch, dim, seqlen, dstate)
                A3B = torch.einsum('dnmk,bdkl->bdln', A_cubed, B_expanded)  # (batch, dim, seqlen, dstate)
                
                # RK4 coefficients: k1, k2, k3, k4
                # delta is (batch, dim, seqlen)
                k1 = delta.unsqueeze(-1) * B_expanded.permute(0, 1, 3, 2)  # (batch, dim, seqlen, dstate)
                k2 = (delta ** 2).unsqueeze(-1) * AB / 2.0  # (batch, dim, seqlen, dstate)
                k3 = (delta ** 3).unsqueeze(-1) * A2B / 6.0  # (batch, dim, seqlen, dstate)
                k4 = (delta ** 4).unsqueeze(-1) * A3B / 24.0  # (batch, dim, seqlen, dstate)
                
                deltaB = k1 + k2 + k3 + k4  # (batch, dim, seqlen, dstate)
                deltaB_u = deltaB * u.unsqueeze(-1)  # (batch, dim, seqlen, dstate)
            else:
                # B is (batch, n_groups, dstate, seqlen)
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                # B is now (batch, dim, dstate, seqlen)
                B_expanded = B.permute(0, 1, 3, 2)  # (batch, dim, seqlen, dstate)
                
                # Compute AB, A2B, A3B
                AB = torch.einsum('dn,bdnl->bdln', A, B)  # (batch, dim, seqlen, dstate)
                A2B = torch.einsum('dnm,bdml->bdln', A_squared, B)  # (batch, dim, seqlen, dstate)
                A3B = torch.einsum('dnmk,bdkl->bdln', A_cubed, B)  # (batch, dim, seqlen, dstate)
                
                # RK4 coefficients
                k1 = delta.unsqueeze(-1) * B_expanded  # (batch, dim, seqlen, dstate)
                k2 = (delta ** 2).unsqueeze(-1) * AB / 2.0  # (batch, dim, seqlen, dstate)
                k3 = (delta ** 3).unsqueeze(-1) * A2B / 6.0  # (batch, dim, seqlen, dstate)
                k4 = (delta ** 4).unsqueeze(-1) * A3B / 24.0  # (batch, dim, seqlen, dstate)
                
                deltaB = k1 + k2 + k3 + k4  # (batch, dim, seqlen, dstate)
                deltaB_u = deltaB * u.unsqueeze(-1)  # (batch, dim, seqlen, dstate)
    else:
        raise ValueError(f"Unknown discretization method: {discretization_method}")
    
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    
    # Check if deltaA is a matrix (bilinear) or vector (other methods)
    is_matrix_deltaA = deltaA.dim() == 5  # (batch, dim, seqlen, dstate, dstate)
    
    # Polynomial Interpolation uses bidirectional (non-causal) scan
    if discretization_method == "poly":
        # Forward pass: scan from left to right
        x_f = A.new_zeros((batch, dim, dstate))
        ys_f = []
        for i in range(u.shape[2]):
            if is_matrix_deltaA:
                x_f = torch.matmul(deltaA[:, :, i], x_f.unsqueeze(-1)).squeeze(-1) + deltaB_u[:, :, i]
            else:
                x_f = deltaA[:, :, i] * x_f + deltaB_u[:, :, i]
            if not is_variable_C:
                y_f = torch.einsum('bdn,dn->bd', x_f, C)
            else:
                if C.dim() == 3:
                    y_f = torch.einsum('bdn,bn->bd', x_f, C[:, :, i])
                else:
                    y_f = torch.einsum('bdn,bdn->bd', x_f, C[:, :, :, i])
            if y_f.is_complex():
                y_f = y_f.real * 2
            ys_f.append(y_f)
        
        # Backward pass: scan from right to left (flip inputs)
        x_b = A.new_zeros((batch, dim, dstate))
        ys_b = []
        deltaA_b = deltaA.flip([2])  # Flip along sequence dimension
        deltaB_u_b = deltaB_u.flip([2])
        u_b = u.flip([2])
        # Handle variable C: flip along sequence dimension if it's variable
        if is_variable_C:
            if C.dim() == 3:
                C_b = C.flip([2])  # (B, N, L) -> flip L
            elif C.dim() == 4:
                C_b = C.flip([3])  # (B, G, N, L) -> flip L
            else:
                C_b = C
        else:
            C_b = C
        
        for i in range(u_b.shape[2]):
            if is_matrix_deltaA:
                x_b = torch.matmul(deltaA_b[:, :, i], x_b.unsqueeze(-1)).squeeze(-1) + deltaB_u_b[:, :, i]
            else:
                x_b = deltaA_b[:, :, i] * x_b + deltaB_u_b[:, :, i]
            if not is_variable_C:
                y_b = torch.einsum('bdn,dn->bd', x_b, C_b)
            else:
                if C_b.dim() == 3:
                    y_b = torch.einsum('bdn,bn->bd', x_b, C_b[:, :, i])
                else:
                    y_b = torch.einsum('bdn,bdn->bd', x_b, C_b[:, :, :, i])
            if y_b.is_complex():
                y_b = y_b.real * 2
            ys_b.append(y_b)
        
        # Combine forward and backward passes (average for smooth interpolation)
        y_f = torch.stack(ys_f, dim=2)  # (batch dim L)
        y_b = torch.stack(ys_b, dim=2).flip([2])  # (batch dim L), flip back to original order
        y = (y_f + y_b) / 2.0  # Average forward and backward for non-causal smooth interpolation
        
        last_state = x_f  # Use forward state as last state
        
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        return out if not return_last_state else (out, last_state)
    
    # Causal scan for all other methods (including HOH)
    ys = []  # Initialize output list for causal methods
    for i in range(u.shape[2]):
        if is_matrix_deltaA:
            # Bilinear: deltaA is a matrix, use matrix-vector multiplication
            # deltaA[:, :, i] has shape (batch, dim, dstate, dstate)
            # x has shape (batch, dim, dstate)
            # Need to do: deltaA[:, :, i] @ x.unsqueeze(-1) -> (batch, dim, dstate, 1) -> squeeze to (batch, dim, dstate)
            x = torch.matmul(deltaA[:, :, i], x.unsqueeze(-1)).squeeze(-1) + deltaB_u[:, :, i]
        else:
            # Other methods: deltaA is a vector, use element-wise multiplication
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFnNoOutProj(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type='cuda')
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, discretization_method="zoh"):
        """
             xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, None, None, True)
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        # discretization_method enum: 0=zoh, 1=foh, 2=bilinear, 3=poly, 4=highorder, 5=rk4
        # Map discretization method string to enum value
        disc_method_map = {
            "zoh": 0,
            "foh": 1,
            "bilinear": 2,
            "poly": 3,
            "highorder": 4,
            "rk4": 5
        }
        disc_method_enum = disc_method_map.get(discretization_method, 0)
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus, disc_method_enum
        )
        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        # return rearrange(out_z, "b d l -> b l d")
        return out_z

    @staticmethod
    @_custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, None, None, True)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None, None, None)
    

class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type='cuda')
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @_custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None, None)


class BiMambaInnerFn(torch.autograd.Function):

    @staticmethod
    @_custom_fwd(device_type='cuda')
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, A_b, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
             xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias,None, None, None, True)
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out_f, scan_intermediates_f, out_z_f = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        assert not A_b.is_complex(), "A should not be complex!!"
        out_b, scan_intermediates_b, out_z_b = selective_scan_cuda.fwd(
            conv1d_out.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, delta_softplus,
        )

        out_z = out_z_f + out_z_b.flip([-1])

        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, A_b, B, C, D, delta_bias, scan_intermediates_f, scan_intermediates_b, out_f, out_b)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @_custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, A_b, B, C, D, delta_bias, scan_intermediates_f, scan_intermediates_b, out_f, out_b) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z_f = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates_f, out_f, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        # flip one
        dz_b = torch.empty_like(dz)
        dconv1d_out_f_b, ddelta_f_b, dA_b, dB_f_b, dC_f_b, dD_b, ddelta_bias_b, dz_b, out_z_b = selective_scan_cuda.bwd(
            conv1d_out.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, dout_y.flip([-1]), scan_intermediates_b, out_b, dz_b,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )

        dconv1d_out = dconv1d_out + dconv1d_out_f_b.flip([-1])
        ddelta = ddelta + ddelta_f_b.flip([-1])
        dB = dB + dB_f_b.flip([-1])
        dC = dC + dC_f_b.flip([-1])
        dD = dD + dD_b
        ddelta_bias = ddelta_bias + ddelta_bias_b
        dz = dz + dz_b.flip([-1])
        out_z = out_z_f + out_z_b.flip([-1])
        
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dA_b, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None, None)

def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)

def bimamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, A_b, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return BiMambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, A_b, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)


def mamba_inner_fn_no_out_proj(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=None, discretization_method="zoh"
):
    import os
    if checkpoint_lvl is None:
        checkpoint_lvl = int(os.environ.get("MAMBA_CHECKPOINT_LVL", "1"))
    return MambaInnerFnNoOutProj.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, discretization_method)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


def bimamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, A_b, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, "silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    y_b = selective_scan_fn(x.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, delta_softplus=True)
    y = y + y_b.flip([-1])
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
