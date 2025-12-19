#!/usr/bin/env python3
"""
Profiling script to measure performance differences between discretization methods.
This script helps identify bottlenecks in ZOH (CUDA kernel) vs other methods (Python reference).
"""

import torch
import time
import numpy as np
from typing import Dict, List
import argparse

def profile_exp2f_vs_inverse(batch: int = 8, dim: int = 512, seqlen: int = 1024, dstate: int = 16, n_trials: int = 100):
    """Profile exp2f operations (ZOH) vs matrix inverse operations (BILINEAR)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    
    results = {}
    
    # Create test data
    delta = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    A = torch.randn(dim, dstate, device=device, dtype=torch.float32) * 0.1  # Small values for stability
    B = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    u = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        _ = torch.inverse(torch.eye(dstate, device=device).unsqueeze(0).unsqueeze(0) + 
                          torch.einsum('bdl,dn->bdln', delta, A) * 0.5)
    
    torch.cuda.synchronize()
    
    # Profile ZOH: exp2f equivalent (torch.exp)
    times_exp = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_exp.append((end - start) * 1000)  # Convert to ms
    
    results['exp_ops'] = {
        'mean_ms': np.mean(times_exp),
        'std_ms': np.std(times_exp),
        'min_ms': np.min(times_exp),
        'max_ms': np.max(times_exp)
    }
    
    # Profile BILINEAR: matrix inverse
    # For bilinear, we need (I + A*delta/2)^-1
    half_delta_A = torch.einsum('bdl,dn->bdln', delta, A) * 0.5
    I = torch.eye(dstate, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    I_plus_half_delta_A = I + half_delta_A.unsqueeze(-1)
    
    times_inverse = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        I_plus_half_delta_A_reshaped = I_plus_half_delta_A.reshape(-1, dstate, dstate)
        I_plus_half_delta_A_inv_reshaped = torch.inverse(I_plus_half_delta_A_reshaped)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_inverse.append((end - start) * 1000)
    
    results['inverse_ops'] = {
        'mean_ms': np.mean(times_inverse),
        'std_ms': np.std(times_inverse),
        'min_ms': np.min(times_inverse),
        'max_ms': np.max(times_inverse)
    }
    
    # Profile full ZOH discretization (CUDA kernel path simulation)
    times_zoh_full = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        # Simulate scan operation
        x = torch.zeros(batch, dim, dstate, device=device)
        for i in range(seqlen):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_zoh_full.append((end - start) * 1000)
    
    results['zoh_full'] = {
        'mean_ms': np.mean(times_zoh_full),
        'std_ms': np.std(times_zoh_full),
        'min_ms': np.min(times_zoh_full),
        'max_ms': np.max(times_zoh_full)
    }
    
    # Profile full BILINEAR discretization (Python reference path)
    times_bilinear_full = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        half_delta_A = torch.einsum('bdl,dn->bdln', delta, A) * 0.5
        I = torch.eye(dstate, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        I_plus_half_delta_A = I + half_delta_A
        I_minus_half_delta_A = I - half_delta_A
        
        I_plus_half_delta_A_reshaped = I_plus_half_delta_A.reshape(-1, dstate, dstate)
        I_plus_half_delta_A_inv_reshaped = torch.inverse(I_plus_half_delta_A_reshaped)
        I_plus_half_delta_A_inv = I_plus_half_delta_A_inv_reshaped.reshape(batch, dim, seqlen, dstate, dstate)
        
        deltaA = torch.matmul(I_plus_half_delta_A_inv, I_minus_half_delta_A)
        delta_expanded = delta.unsqueeze(-1).unsqueeze(-1)
        B_expanded = B.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        deltaB = torch.matmul(I_plus_half_delta_A_inv, delta_expanded * B_expanded)
        deltaB_u = deltaB * u.unsqueeze(-1).unsqueeze(-1)
        
        # Simulate scan operation
        x = torch.zeros(batch, dim, dstate, device=device)
        for i in range(seqlen):
            x = torch.einsum('bdn,bdnn->bdn', x, deltaA[:, :, i]) + deltaB_u[:, :, i].squeeze(-1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_bilinear_full.append((end - start) * 1000)
    
    results['bilinear_full'] = {
        'mean_ms': np.mean(times_bilinear_full),
        'std_ms': np.std(times_bilinear_full),
        'min_ms': np.min(times_bilinear_full),
        'max_ms': np.max(times_bilinear_full)
    }
    
    return results


def profile_selective_scan_methods(batch: int = 8, dim: int = 512, seqlen: int = 1024, dstate: int = 16, n_trials: int = 50):
    """Profile the actual selective_scan_fn with different discretization methods."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError:
        print("Warning: Could not import selective_scan_fn. Make sure mamba-ssm is installed.")
        return {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling selective_scan_fn on device: {device}")
    
    results = {}
    methods = ["zoh", "bilinear", "foh", "poly", "highorder", "rk4"]
    
    # Create test data
    u = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    delta = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32) * 0.1
    A = torch.randn(dim, dstate, device=device, dtype=torch.float32) * 0.1
    B = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    C = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    
    for method in methods:
        print(f"  Profiling {method.upper()}...")
        # Warmup
        for _ in range(5):
            try:
                _ = selective_scan_fn(u, delta, A, B, C, discretization_method=method)
            except Exception as e:
                print(f"    Warning: {method} failed during warmup: {e}")
                break
        
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
                print(f"    Error in {method}: {e}")
                break
        
        if times:
            results[method] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'throughput_imgs_per_sec': (batch * n_trials) / (np.sum(times) / 1000)
            }
    
    return results


def print_results(results: Dict, title: str):
    """Print profiling results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    for key, stats in results.items():
        print(f"\n{key.upper()}:")
        print(f"  Mean:   {stats['mean_ms']:.4f} ms")
        print(f"  Std:    {stats['std_ms']:.4f} ms")
        print(f"  Min:    {stats['min_ms']:.4f} ms")
        print(f"  Max:    {stats['max_ms']:.4f} ms")
        if 'throughput_imgs_per_sec' in stats:
            print(f"  Throughput: {stats['throughput_imgs_per_sec']:.2f} img/s")
    
    # Compare if we have both zoh and bilinear
    if 'zoh' in results and 'bilinear' in results:
        speedup = results['zoh']['mean_ms'] / results['bilinear']['mean_ms']
        print(f"\n{'='*60}")
        print(f"BILINEAR vs ZOH Speedup: {speedup:.2f}x")
        if speedup > 1:
            print(f"  BILINEAR is {speedup:.2f}x FASTER than ZOH")
        else:
            print(f"  ZOH is {1/speedup:.2f}x FASTER than BILINEAR")


def main():
    parser = argparse.ArgumentParser(description='Profile discretization methods')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--dim', type=int, default=512, help='Dimension')
    parser.add_argument('--seqlen', type=int, default=1024, help='Sequence length')
    parser.add_argument('--dstate', type=int, default=16, help='State dimension')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--profile_ops', action='store_true', help='Profile individual operations')
    parser.add_argument('--profile_scan', action='store_true', help='Profile selective_scan_fn')
    
    args = parser.parse_args()
    
    if not args.profile_ops and not args.profile_scan:
        # Default: profile both
        args.profile_ops = True
        args.profile_scan = True
    
    if args.profile_ops:
        print("Profiling individual operations (exp vs inverse)...")
        op_results = profile_exp2f_vs_inverse(
            batch=args.batch,
            dim=args.dim,
            seqlen=args.seqlen,
            dstate=args.dstate,
            n_trials=args.n_trials
        )
        print_results(op_results, "Individual Operations Profiling")
    
    if args.profile_scan:
        print("\nProfiling selective_scan_fn with different methods...")
        scan_results = profile_selective_scan_methods(
            batch=args.batch,
            dim=args.dim,
            seqlen=args.seqlen,
            dstate=args.dstate,
            n_trials=args.n_trials
        )
        print_results(scan_results, "Selective Scan Methods Profiling")


if __name__ == "__main__":
    main()
