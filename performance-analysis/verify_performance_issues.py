#!/usr/bin/env python3
"""
Comprehensive verification script to check for performance issues that could explain
throughput differences between discretization methods.

Checks:
1. Implementation/path differences (CUDA vs Python fallback)
2. Different code paths or preprocessing
3. Threading/BLAS/OpenMP differences
4. GPU synchronization issues
5. Caching/warm-up issues
6. Memory access patterns
"""

import torch
import time
import os
import sys
import numpy as np
from typing import Dict, List, Tuple
import argparse

def check_1_identical_inputs(batch: int = 8, dim: int = 512, seqlen: int = 1024, dstate: int = 16):
    """Verify identical inputs: same batch_size, image shapes, dtype, data loader, and seed."""
    print("="*60)
    print("CHECK 1: Verifying Identical Inputs")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Create identical test data
    u = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    delta = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32) * 0.1
    A = torch.randn(dim, dstate, device=device, dtype=torch.float32) * 0.1
    B = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    C = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    
    print(f"✓ Input shapes verified:")
    print(f"  u: {u.shape}, dtype: {u.dtype}, device: {u.device}")
    print(f"  delta: {delta.shape}, dtype: {delta.dtype}, device: {delta.device}")
    print(f"  A: {A.shape}, dtype: {A.dtype}, device: {A.device}")
    print(f"  B: {B.shape}, dtype: {B.dtype}, device: {B.device}")
    print(f"  C: {C.shape}, dtype: {C.dtype}, device: {C.device}")
    
    return u, delta, A, B, C


def check_2_gpu_synchronization():
    """Ensure correct GPU timing: call torch.cuda.synchronize() before/after timed regions."""
    print("\n" + "="*60)
    print("CHECK 2: GPU Synchronization")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping GPU sync check")
        return
    
    device = torch.device("cuda")
    
    # Test with and without synchronization
    x = torch.randn(1000, 1000, device=device)
    
    # Without sync (may be inaccurate)
    start = time.perf_counter()
    y = x @ x
    end = time.perf_counter()
    time_no_sync = (end - start) * 1000
    
    # With sync (accurate)
    torch.cuda.synchronize()
    start = time.perf_counter()
    y = x @ x
    torch.cuda.synchronize()
    end = time.perf_counter()
    time_with_sync = (end - start) * 1000
    
    print(f"✓ Timing test:")
    print(f"  Without sync: {time_no_sync:.4f} ms (may be inaccurate)")
    print(f"  With sync: {time_with_sync:.4f} ms (accurate)")
    print(f"  Difference: {abs(time_no_sync - time_with_sync):.4f} ms")
    
    if abs(time_no_sync - time_with_sync) > 0.1:
        print("  ⚠ WARNING: Significant difference - ensure sync is used in benchmarks")


def check_3_threading_blas():
    """Compare OMP/MKL threads to see large swings: OMP_NUM_THREADS=1 vs 8."""
    print("\n" + "="*60)
    print("CHECK 3: Threading/BLAS Configuration")
    print("="*60)
    
    print(f"Current environment variables:")
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'Not set')
    mkl_threads = os.environ.get('MKL_NUM_THREADS', 'Not set')
    print(f"  OMP_NUM_THREADS: {omp_threads}")
    print(f"  MKL_NUM_THREADS: {mkl_threads}")
    
    # Check PyTorch threading
    print(f"\nPyTorch configuration:")
    print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
    
    if torch.cuda.is_available():
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"  torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    
    print(f"\n⚠ Recommendation: Ensure OMP_NUM_THREADS is consistent across all runs")


def check_4_code_paths():
    """Verify which code paths are being used (CUDA kernel vs Python fallback)."""
    print("\n" + "="*60)
    print("CHECK 4: Code Path Verification")
    print("="*60)
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
        import selective_scan_cuda
        cuda_available = selective_scan_cuda is not None
    except ImportError as e:
        print(f"⚠ Could not import selective_scan modules: {e}")
        return
    
    print(f"CUDA kernel available: {cuda_available}")
    
    if not cuda_available:
        print("⚠ WARNING: CUDA kernel not available - all methods will use Python fallback")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠ CUDA not available, cannot test CUDA kernels")
        return
    
    # Create test data
    u, delta, A, B, C = check_1_identical_inputs()
    
    methods = ["zoh", "foh", "bilinear", "poly", "highorder", "rk4"]
    code_paths = {}
    
    print(f"\nTesting code paths for each method:")
    for method in methods:
        try:
            # Try to call and see if it uses CUDA or falls back
            # We can't directly detect this, but we can check if it raises an error
            result = selective_scan_fn(u, delta, A, B, C, discretization_method=method)
            code_paths[method] = "Success (likely CUDA or fallback)"
            print(f"  {method.upper()}: ✓ Success")
        except Exception as e:
            code_paths[method] = f"Error: {e}"
            print(f"  {method.upper()}: ✗ Error: {e}")
    
    print(f"\n⚠ To verify actual code path, check selective_scan_interface.py:")
    print(f"  - CUDA kernel is used if selective_scan_cuda.fwd() succeeds")
    print(f"  - Python fallback is used if CUDA kernel fails or is None")


def check_5_warmup_caching():
    """Check for caching/warm-up issues (JIT/AVX/CPU freq scaling) or I/O bottlenecks."""
    print("\n" + "="*60)
    print("CHECK 5: Warm-up and Caching")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError:
        print("⚠ Could not import selective_scan_fn")
        return
    
    u, delta, A, B, C = check_1_identical_inputs()
    
    # Test warm-up effect
    print("Testing warm-up effect (first few iterations may be slower):")
    times = []
    
    for i in range(20):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = selective_scan_fn(u, delta, A, B, C, discretization_method="zoh")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
        if i < 5 or i % 5 == 0:
            print(f"  Iteration {i+1}: {times[-1]:.4f} ms")
    
    warmup_avg = np.mean(times[:5])
    steady_avg = np.mean(times[10:])
    
    print(f"\n  First 5 iterations (warmup): {warmup_avg:.4f} ms avg")
    print(f"  Last 10 iterations (steady): {steady_avg:.4f} ms avg")
    
    if warmup_avg > steady_avg * 1.5:
        print(f"  ⚠ WARNING: Significant warm-up effect ({warmup_avg/steady_avg:.2f}x slower)")
        print(f"     Recommendation: Skip first 5-10 iterations in benchmarks")


def check_6_micro_benchmark():
    """Compare single-kernel micro-benchmarks to find where time is spent."""
    print("\n" + "="*60)
    print("CHECK 6: Micro-benchmark Individual Operations")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠ CUDA not available, skipping micro-benchmark")
        return
    
    u, delta, A, B, C = check_1_identical_inputs()
    n_trials = 100
    
    operations = {
        "exp_operation": lambda: torch.exp(torch.einsum('bdl,dn->bdln', delta, A)),
        "matrix_inverse": lambda: torch.inverse(
            torch.eye(A.shape[1], device=device).unsqueeze(0).unsqueeze(0) + 
            torch.einsum('bdl,dn->bdln', delta, A) * 0.5
        ),
        "einsum_delta_B_u": lambda: torch.einsum('bdl,dn,bdl->bdln', delta, B, u),
        "matrix_multiply": lambda: torch.matmul(
            torch.randn(100, 100, device=device),
            torch.randn(100, 100, device=device)
        ),
    }
    
    print("Profiling individual operations:")
    results = {}
    
    for op_name, op_func in operations.items():
        # Warmup
        for _ in range(10):
            _ = op_func()
        
        torch.cuda.synchronize()
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = op_func()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        results[op_name] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
        
        print(f"  {op_name:20s}: {results[op_name]['mean']:8.4f} ± {results[op_name]['std']:.4f} ms")
    
    # Compare exp vs inverse (key difference between ZOH and BILINEAR)
    exp_time = results['exp_operation']['mean']
    inv_time = results['matrix_inverse']['mean']
    ratio = inv_time / exp_time if exp_time > 0 else float('inf')
    
    print(f"\n  Key comparison:")
    print(f"    exp operation: {exp_time:.4f} ms")
    print(f"    matrix inverse: {inv_time:.4f} ms")
    print(f"    Ratio (inv/exp): {ratio:.2f}x")
    
    if ratio > 10:
        print(f"    ⚠ WARNING: Matrix inverse is {ratio:.2f}x slower than exp")
        print(f"       This could explain BILINEAR being slower if using Python fallback")


def check_7_full_method_comparison():
    """Full comparison of all discretization methods with proper synchronization."""
    print("\n" + "="*60)
    print("CHECK 7: Full Method Comparison (with proper sync)")
    print("="*60)
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError:
        print("⚠ Could not import selective_scan_fn")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠ CUDA not available, skipping full comparison")
        return
    
    u, delta, A, B, C = check_1_identical_inputs()
    methods = ["zoh", "foh", "bilinear", "poly", "highorder", "rk4"]
    n_trials = 50
    n_warmup = 10
    
    print(f"Profiling all methods with {n_warmup} warmup + {n_trials} trials:")
    
    results = {}
    for method in methods:
        # Warmup
        for _ in range(n_warmup):
            try:
                _ = selective_scan_fn(u, delta, A, B, C, discretization_method=method)
            except Exception as e:
                print(f"  {method.upper()}: ✗ Failed during warmup: {e}")
                continue
        
        torch.cuda.synchronize()
        
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                _ = selective_scan_fn(u, delta, A, B, C, discretization_method=method)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            except Exception as e:
                print(f"  {method.upper()}: ✗ Error: {e}")
                break
        
        if times:
            results[method] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'throughput': (u.shape[0] * n_trials) / (np.sum(times) / 1000)
            }
            print(f"  {method.upper():12s}: {results[method]['mean']:8.4f} ± {results[method]['std']:.4f} ms "
                  f"({results[method]['throughput']:.2f} img/s)")
    
    # Compare to ZOH baseline
    if 'zoh' in results:
        zoh_time = results['zoh']['mean']
        print(f"\n  Comparison to ZOH baseline:")
        for method, stats in results.items():
            if method != 'zoh':
                speedup = zoh_time / stats['mean']
                print(f"    {method.upper():12s}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than ZOH")


def main():
    parser = argparse.ArgumentParser(description='Verify performance issues')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--dim', type=int, default=512, help='Dimension')
    parser.add_argument('--seqlen', type=int, default=1024, help='Sequence length')
    parser.add_argument('--dstate', type=int, default=16, help='State dimension')
    parser.add_argument('--skip-checks', nargs='+', type=int, default=[], 
                       help='Skip specific checks (1-7)')
    
    args = parser.parse_args()
    
    checks = [
        (1, check_1_identical_inputs, [args.batch, args.dim, args.seqlen, args.dstate]),
        (2, check_2_gpu_synchronization, []),
        (3, check_3_threading_blas, []),
        (4, check_4_code_paths, []),
        (5, check_5_warmup_caching, []),
        (6, check_6_micro_benchmark, []),
        (7, check_7_full_method_comparison, []),
    ]
    
    print("\n" + "="*60)
    print("PERFORMANCE VERIFICATION CHECKS")
    print("="*60)
    
    for check_num, check_func, check_args in checks:
        if check_num not in args.skip_checks:
            try:
                check_func(*check_args)
            except Exception as e:
                print(f"\n✗ Check {check_num} failed with error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print("\nRecommendations:")
    print("1. Ensure all runs use identical inputs (same seed, shapes, dtype)")
    print("2. Always use torch.cuda.synchronize() before/after timing")
    print("3. Set consistent OMP_NUM_THREADS across all runs")
    print("4. Skip warmup iterations (first 5-10) in benchmarks")
    print("5. Verify CUDA kernels are being used (not Python fallback)")
    print("6. Profile individual operations to find bottlenecks")


if __name__ == "__main__":
    main()
