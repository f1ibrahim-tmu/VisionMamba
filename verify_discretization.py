#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive diagnostic script to verify that all discretization methods produce different outputs.
This helps identify if the CUDA kernel switch is working correctly for all methods.
"""

import torch
import sys
import os
from collections import OrderedDict

# Add the mamba path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mamba-1p1p1'))

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("Successfully imported selective_scan_fn")
except ImportError as e:
    print(f"Failed to import selective_scan_fn: {e}")
    sys.exit(1)

# All available discretization methods
ALL_METHODS = OrderedDict([
    ("zoh", "Zero Order Hold"),
    ("foh", "First Order Hold"),
    ("bilinear", "Bilinear (Tustin)"),
    ("poly", "Polynomial Interpolation"),
    ("highorder", "Higher-Order Hold"),
    ("rk4", "Runge-Kutta 4th Order"),
])

def verify_all_methods():
    """Verify that all discretization methods produce different outputs"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE DISCRETIZATION METHOD VERIFICATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Determine which kernel to use
    if device == "cpu":
        print("Warning: Running on CPU. CUDA kernel won't be used.")
        print("   Using Python reference implementation to test discretization methods.")
        print("   Note: This will test the Python implementation, not the CUDA kernel switch.")
        use_cuda = False  # Force Python reference on CPU
    else:
        print("Using CUDA device. Will test CUDA kernel implementation.")
        use_cuda = True  # Force CUDA kernel on GPU
    
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
    
    # Test all methods
    print(f"\n{'='*70}")
    print("TESTING ALL DISCRETIZATION METHODS")
    print(f"{'='*70}")
    
    outputs = {}
    method_stats = {}
    
    for method_key, method_name in ALL_METHODS.items():
        print(f"\n{'-'*70}")
        print(f"Method: {method_name} ({method_key.upper()})")
        print(f"Implementation: {'CUDA Kernel' if use_cuda else 'Python Reference'}")
        print(f"{'-'*70}")
        
        try:
            out, _ = selective_scan_fn(
                u, delta, A, B_param, C_param, D_param,
                discretization_method=method_key,
                use_cuda_kernel=use_cuda
            )
            
            outputs[method_key] = out
            method_stats[method_key] = {
                'mean': out.mean().item(),
                'std': out.std().item(),
                'min': out.min().item(),
                'max': out.max().item(),
                'shape': out.shape,
                'dtype': str(out.dtype)
            }
            
            print(f"  Status: SUCCESS")
            print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
            print(f"  Output mean: {method_stats[method_key]['mean']:.6f}")
            print(f"  Output std: {method_stats[method_key]['std']:.6f}")
            print(f"  Output range: [{method_stats[method_key]['min']:.6f}, {method_stats[method_key]['max']:.6f}]")
            
        except Exception as e:
            print(f"  Status: FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            outputs[method_key] = None
    
    # Check if all methods succeeded
    failed_methods = [k for k, v in outputs.items() if v is None]
    if failed_methods:
        print(f"\n{'='*70}")
        print(f"ERROR: {len(failed_methods)} method(s) failed: {', '.join(failed_methods)}")
        print(f"{'='*70}")
        return None
    
    # Pairwise comparison
    print(f"\n{'='*70}")
    print("PAIRWISE COMPARISON MATRIX")
    print(f"{'='*70}")
    
    method_keys = list(ALL_METHODS.keys())
    n_methods = len(method_keys)
    
    # Create comparison matrix
    comparison_matrix = {}
    differences_summary = []
    
    print(f"\n{'Method 1':<15} {'Method 2':<15} {'Mean Diff':<15} {'Max Diff':<15} {'Status':<20}")
    print(f"{'-'*80}")
    
    for i, method1 in enumerate(method_keys):
        for j, method2 in enumerate(method_keys):
            if i >= j:  # Only compare upper triangle (avoid duplicates)
                continue
            
            out1 = outputs[method1]
            out2 = outputs[method2]
            
            diff = (out1 - out2).abs()
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            min_diff = diff.min().item()
            
            # Determine status
            if mean_diff == 0.0 and max_diff == 0.0:
                status = "IDENTICAL (BUG!)"
            elif torch.allclose(out1, out2, atol=1e-5, rtol=1e-5):
                status = "Numerically Identical"
            elif mean_diff < 1e-6:
                status = "Very Similar"
            else:
                status = "Different (OK)"
            
            comparison_matrix[(method1, method2)] = {
                'mean_diff': mean_diff,
                'max_diff': max_diff,
                'min_diff': min_diff,
                'status': status
            }
            
            method1_short = method1.upper()[:12]
            method2_short = method2.upper()[:12]
            
            print(f"{method1_short:<15} {method2_short:<15} {mean_diff:<15.10f} {max_diff:<15.10f} {status:<20}")
            
            differences_summary.append({
                'method1': method1,
                'method2': method2,
                'mean_diff': mean_diff,
                'max_diff': max_diff,
                'status': status
            })
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    identical_pairs = [d for d in differences_summary if d['status'] == "IDENTICAL (BUG!)"]
    numerically_identical = [d for d in differences_summary if d['status'] == "Numerically Identical"]
    very_similar = [d for d in differences_summary if d['status'] == "Very Similar"]
    different_pairs = [d for d in differences_summary if d['status'] == "Different (OK)"]
    
    print(f"\nTotal method pairs compared: {len(differences_summary)}")
    print(f"  Different (OK): {len(different_pairs)}")
    print(f"  Very Similar: {len(very_similar)}")
    print(f"  Numerically Identical: {len(numerically_identical)}")
    print(f"  IDENTICAL (BUG!): {len(identical_pairs)}")
    
    if identical_pairs:
        print(f"\n{'='*70}")
        print("CRITICAL: Found identical method pairs!")
        print(f"{'='*70}")
        for pair in identical_pairs:
            print(f"  {pair['method1'].upper()} vs {pair['method2'].upper()}: EXACTLY IDENTICAL")
            print(f"    This suggests the CUDA kernel switch is NOT working for these methods.")
    
    if numerically_identical:
        print(f"\n{'='*70}")
        print("WARNING: Found numerically identical method pairs!")
        print(f"{'='*70}")
        for pair in numerically_identical:
            print(f"  {pair['method1'].upper()} vs {pair['method2'].upper()}: Numerically identical")
    
    # Detailed diagnostics for a few sample pairs
    print(f"\n{'='*70}")
    print("DETAILED SAMPLE COMPARISONS")
    print(f"{'='*70}")
    
    # Show detailed comparison for ZOH vs Bilinear (original test case)
    if 'zoh' in outputs and 'bilinear' in outputs:
        print(f"\nDetailed: ZOH vs Bilinear")
        print(f"{'-'*70}")
        out_zoh = outputs['zoh']
        out_bilinear = outputs['bilinear']
        diff = (out_zoh - out_bilinear).abs()
        
        sample_indices = [(0, 0, 0), (0, 0, 10), (0, 10, 20), (0, 20, 50)]
        print(f"\nSample element comparisons:")
        for b, d, l in sample_indices:
            zoh_val = out_zoh[b, d, l].item()
            bil_val = out_bilinear[b, d, l].item()
            diff_val = abs(zoh_val - bil_val)
            print(f"  Element [{b}, {d}, {l}]: ZOH={zoh_val:.6f}, Bilinear={bil_val:.6f}, Diff={diff_val:.6f}")
        
        num_identical = (diff < 1e-7).sum().item()
        total_elements = diff.numel()
        pct_identical = 100.0 * num_identical / total_elements
        print(f"\nElements identical (diff < 1e-7): {num_identical}/{total_elements} ({pct_identical:.2f}%)")
    
    # Method statistics table
    print(f"\n{'='*70}")
    print("METHOD OUTPUT STATISTICS")
    print(f"{'='*70}")
    print(f"\n{'Method':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print(f"{'-'*80}")
    for method_key, stats in method_stats.items():
        method_name = ALL_METHODS[method_key]
        print(f"{method_name[:18]:<20} {stats['mean']:<15.6f} {stats['std']:<15.6f} "
              f"{stats['min']:<15.6f} {stats['max']:<15.6f}")
    
    return {
        'outputs': outputs,
        'comparison_matrix': comparison_matrix,
        'differences_summary': differences_summary,
        'method_stats': method_stats,
        'identical_pairs': identical_pairs,
        'different_pairs': different_pairs
    }

if __name__ == "__main__":
    try:
        result = verify_all_methods()
        if result is None:
            print("\nTest failed - could not complete verification")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}")
        
        if result['identical_pairs']:
            print("\n1. CRITICAL: Rebuild the CUDA extension:")
            print("   cd mamba-1p1p1")
            print("   pip install -e . --force-reinstall --no-cache-dir")
            print("\n2. Check that discretization_kernels.cuh is being included")
            print("\n3. Verify the enum values match between Python and C++:")
            print("   Python: zoh=0, foh=1, bilinear=2, poly=3, highorder=4, rk4=5")
            print("   C++: DISCRETIZATION_ZOH=0, DISCRETIZATION_FOH=1, etc.")
            print("\n4. Check the switch statement in discretization_kernels.cuh")
            print("   Ensure all cases have proper break statements")
        elif len(result['different_pairs']) == len(result['differences_summary']):
            print("\nSUCCESS: All discretization methods produce different outputs!")
            print("The CUDA kernel switch appears to be working correctly for all methods.")
        else:
            print("\nPARTIAL SUCCESS: Some methods produce different outputs.")
            print("Review the comparison matrix above for details.")
            print("\nIf some methods are too similar:")
            print("1. Try with different input values (larger delta, different A values)")
            print("2. Check if numerical precision is masking differences")
            print("3. Verify the discretization formulas are correctly implemented")
            
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except TypeError as e:
        # Handle case where verify_all_methods returns None
        print(f"\nTest could not complete: {e}")
        sys.exit(1)
