#!/usr/bin/env python3
"""
Extract and compute statistics (Mean ± Std) from multiple seed runs.

Usage:
    python extract_results.py [--method METHOD] [--seeds 0,1,2,3,4]
    
Example:
    python extract_results.py --method zoh
    python extract_results.py --method all
"""

import json
import numpy as np
import argparse
import glob
from pathlib import Path

# Default seeds
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Method names mapping
METHODS = {
    'zoh': 'vim_tiny_zoh',
    'foh': 'vim_tiny_foh',
    'bilinear': 'vim_tiny_bilinear',
    'poly': 'vim_tiny_poly',
    'highorder': 'vim_tiny_highorder',
    'rk4': 'vim_tiny_rk4'
}

def extract_max_accuracy(log_file):
    """Extract maximum accuracy from log file."""
    max_acc = 0.0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'test_acc1' in data:
                        max_acc = max(max_acc, data['test_acc1'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: {log_file} not found")
        return None
    return max_acc if max_acc > 0 else None

def compute_statistics(method_name, seeds, base_dir="./output/classification_logs"):
    """Compute Mean ± Std for a given method across multiple seeds."""
    accuracies = []
    
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    
    for seed in seeds:
        log_file = f"{base_dir}/{method_name}_seed{seed}/log.txt"
        acc = extract_max_accuracy(log_file)
        
        if acc is not None:
            accuracies.append(acc)
            print(f"  Seed {seed}: {acc:.2f}%")
        else:
            print(f"  Seed {seed}: Not found or incomplete")
    
    if len(accuracies) == 0:
        print(f"  No valid results found for {method_name}")
        return None, None
    
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    
    print(f"\n  Results: {len(accuracies)}/{len(seeds)} seeds completed")
    print(f"  Mean ± Std: {mean:.2f} ± {std:.2f}%")
    print(f"  Min: {min(accuracies):.2f}%, Max: {max(accuracies):.2f}%")
    
    return mean, std

def main():
    parser = argparse.ArgumentParser(description='Extract results from multiple seed runs')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all'] + list(METHODS.keys()),
                       help='Method to extract results for (default: all)')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, DEFAULT_SEEDS)),
                       help='Comma-separated list of seeds (default: 0,1,2,3,4)')
    parser.add_argument('--base-dir', type=str, default='./output/classification_logs',
                       help='Base directory for output logs (default: ./output/classification_logs)')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Determine which methods to process
    if args.method == 'all':
        methods_to_process = METHODS.items()
    else:
        methods_to_process = [(args.method, METHODS[args.method])]
    
    # Process each method
    results = {}
    for method_key, method_name in methods_to_process:
        mean, std = compute_statistics(method_name, seeds, args.base_dir)
        if mean is not None:
            results[method_key] = {'mean': mean, 'std': std}
    
    # Print summary table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print(f"{'='*60}")
        print(f"{'Method':<15} {'Mean ± Std':<20} {'N seeds':<10}")
        print(f"{'-'*60}")
        for method_key, stats in results.items():
            print(f"{method_key:<15} {stats['mean']:.2f} ± {stats['std']:.2f}%{'':<5} {len(seeds)}")
        print(f"{'='*60}")
    
    # Save results to file
    output_file = f"{args.base_dir}/results_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()

