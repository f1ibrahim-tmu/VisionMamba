#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to verify that different discretization methods produce different outputs.
This helps identify if the CUDA kernel switch is working correctly.
"""

import torch
import sys
import os

# Add the mamba path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mamba-1p1p1'))

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("Successfully imported selective_scan_fn")
except ImportError as e:
    print(f"Failed to import selective_scan_fn: {e}")
    sys.exit(1)

def verify_math_divergence():
    """Verify that ZOH and Bilinear produce different outputs"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"\n{'='*60}")
    print("DISCRETIZATION METHOD VERIFICATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    if device == "cpu":
        print("Warning: Running on CPU. CUDA kernel won't be used.")
        print("   This test is designed for CUDA to verify the kernel switch.")
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Minimal test dimensions
    B, D, L, N = 1, 64, 128, 16
    u = torch.randn(B, D, L, device=device, dtype=dtype)
    delta = torch.randn(B, D, L, device=device, dtype=dtype).exp()
    A = -torch.rand(D, N, device=device, dtype=torch.float32)
    B_param = torch.randn(B, N, L, device=device, dtype=dtype)
    C_param = torch.randn(B, N, L, device=device, dtype=dtype)
    D_param = torch.randn(D, device=device, dtype=torch.float32)

    print(f"\nInput shapes:")
    print(f"  u: {u.shape}, dtype: {u.dtype}")
    print(f"  delta: {delta.shape}, dtype: {delta.dtype}")
    print(f"  A: {A.shape}, dtype: {A.dtype}")
    print(f"  B: {B_param.shape}, dtype: {B_param.dtype}")
    print(f"  C: {C_param.shape}, dtype: {C_param.dtype}")
    
    # Test 1: ZOH
    print(f"\n{'='*60}")
    print("Running Selective Scan with Method: ZOH")
    print(f"{'='*60}")
    try:
        out_zoh, _ = selective_scan_fn(
            u, delta, A, B_param, C_param, D_param, 
            discretization_method="zoh",
            use_cuda_kernel=True  # Force CUDA kernel
        )
        print(f"ZOH completed successfully")
        print(f"  Output shape: {out_zoh.shape}, dtype: {out_zoh.dtype}")
        print(f"  Output mean: {out_zoh.mean().item():.6f}")
        print(f"  Output std: {out_zoh.std().item():.6f}")
        print(f"  Output min: {out_zoh.min().item():.6f}, max: {out_zoh.max().item():.6f}")
    except Exception as e:
        print(f"✗ ZOH failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Bilinear
    print(f"\n{'='*60}")
    print("Running Selective Scan with Method: Bilinear")
    print(f"{'='*60}")
    try:
        out_bilinear, _ = selective_scan_fn(
            u, delta, A, B_param, C_param, D_param,
            discretization_method="bilinear",
            use_cuda_kernel=True  # Force CUDA kernel
        )
        print(f"Bilinear completed successfully")
        print(f"  Output shape: {out_bilinear.shape}, dtype: {out_bilinear.dtype}")
        print(f"  Output mean: {out_bilinear.mean().item():.6f}")
        print(f"  Output std: {out_bilinear.std().item():.6f}")
        print(f"  Output min: {out_bilinear.min().item():.6f}, max: {out_bilinear.max().item():.6f}")
    except Exception as e:
        print(f"✗ Bilinear failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Compare outputs
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Calculate differences
    diff = (out_zoh - out_bilinear).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    min_diff = diff.min().item()
    
    print(f"Mean Absolute Difference: {mean_diff:.10f}")
    print(f"Max Absolute Difference: {max_diff:.10f}")
    print(f"Min Absolute Difference: {min_diff:.10f}")
    
    # Check if outputs are identical (within numerical precision)
    # For bfloat16, we expect some numerical differences even if methods are different
    # But if they're EXACTLY identical, that's suspicious
    are_identical = torch.allclose(out_zoh, out_bilinear, atol=1e-5, rtol=1e-5)
    
    if mean_diff == 0.0 and max_diff == 0.0:
        print(f"\nCRITICAL: Outputs are EXACTLY identical!")
        print(f"   This strongly suggests the CUDA kernel switch is NOT working.")
        print(f"   Both methods are using the same code path.")
    elif are_identical:
        print(f"\nWARNING: Outputs are numerically identical (within tolerance)")
        print(f"   Mean difference: {mean_diff:.2e}")
        print(f"   This suggests the discretization methods may not be implemented correctly.")
    elif mean_diff < 1e-6:
        print(f"\nWARNING: Differences are very small")
        print(f"   Mean difference: {mean_diff:.2e}")
        print(f"   This might indicate:")
        print(f"   1. The methods are too similar for these inputs")
        print(f"   2. Numerical precision issues")
        print(f"   3. The CUDA kernel switch might not be fully working")
    else:
        print(f"\nSUCCESS: Different discretization methods produce different outputs!")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   The CUDA kernel switch appears to be working correctly.")
    
    # Additional diagnostic: Check individual elements
    print(f"\n{'='*60}")
    print("DETAILED DIAGNOSTICS")
    print(f"{'='*60}")
    
    # Sample a few elements to compare
    sample_indices = [(0, 0, 0), (0, 0, 10), (0, 10, 20), (0, 20, 50)]
    print(f"\nSample element comparisons:")
    for b, d, l in sample_indices:
        zoh_val = out_zoh[b, d, l].item()
        bil_val = out_bilinear[b, d, l].item()
        diff_val = abs(zoh_val - bil_val)
        print(f"  Element [{b}, {d}, {l}]: ZOH={zoh_val:.6f}, Bilinear={bil_val:.6f}, Diff={diff_val:.6f}")
    
    # Check if there's a pattern (all zeros, all same, etc.)
    num_identical = (diff < 1e-7).sum().item()
    total_elements = diff.numel()
    pct_identical = 100.0 * num_identical / total_elements
    print(f"\nElements identical (diff < 1e-7): {num_identical}/{total_elements} ({pct_identical:.2f}%)")
    
    return mean_diff, max_diff

if __name__ == "__main__":
    try:
        mean_diff, max_diff = verify_math_divergence()
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if mean_diff == 0.0:
            print("1. Rebuild the CUDA extension:")
            print("   cd mamba-1p1p1")
            print("   pip install -e . --force-reinstall --no-cache-dir")
            print("\n2. Check that discretization_kernels.cuh is being included")
            print("\n3. Verify the enum values match between Python and C++:")
            print("   Python: zoh=0, bilinear=2")
            print("   C++: DISCRETIZATION_ZOH=0, DISCRETIZATION_BILINEAR=2")
            print("\n4. Try using use_cuda_kernel=False to test Python reference implementation")
        elif mean_diff < 1e-6:
            print("1. Try with different input values (larger delta, different A values)")
            print("2. Check if numerical precision is masking differences")
            print("3. Verify the discretization formulas are correctly implemented")
        else:
            print("The discretization methods are working correctly!")
            
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

