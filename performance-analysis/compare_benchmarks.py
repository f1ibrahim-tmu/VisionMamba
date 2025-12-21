#!/usr/bin/env python3
"""
Compare benchmark results across different discretization methods.
Usage:
    python performance-analysis/compare_benchmarks.py [--output-base ./output]
"""

import json
import os
import sys
from pathlib import Path
import argparse

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Results will be printed as text only.")


def parse_benchmark_results(results_dir):
    """Parse all benchmark JSONs in a directory"""
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        return results
    
    for json_file in results_path.glob('benchmark_*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}")
    
    return results


def compare_methods(output_base='./output', batch_size=1):
    """Compare benchmarks across different methods"""
    methods = ['zoh', 'foh', 'bilinear', 'poly', 'highorder', 'rk4']
    
    comparison = []
    
    for method in methods:
        results_dir = f"{output_base}/vim_tiny_{method}/benchmark_results"
        
        if os.path.exists(results_dir):
            results = parse_benchmark_results(results_dir)
            
            if results:
                # Filter by batch size if specified
                filtered_results = [r for r in results if r.get('batch_size') == batch_size]
                if not filtered_results:
                    # If no exact match, use latest result
                    filtered_results = results
                
                latest = filtered_results[-1]  # Get latest benchmark
                
                model_info = latest.get('model_info', {})
                latency = latest.get('latency', {})
                flops = latest.get('flops')
                
                if latency:
                    throughput = 1000 / latency.get('avg_ms', 1.0)
                else:
                    throughput = None
                
                comparison.append({
                    'Method': method.upper(),
                    'Params (M)': model_info.get('total_params', 0) / 1e6 if model_info else 0,
                    'FLOPs (B)': flops if flops else None,
                    'Latency (ms)': latency.get('avg_ms') if latency else None,
                    'Latency Min (ms)': latency.get('min_ms') if latency else None,
                    'Latency Max (ms)': latency.get('max_ms') if latency else None,
                    'Throughput (img/s)': throughput,
                    'Batch Size': latest.get('batch_size', batch_size),
                    'Model Size (MB)': model_info.get('model_size_mb') if model_info else None,
                })
            else:
                print(f"⚠ No benchmark results found for {method}")
        else:
            print(f"⚠ Results directory not found: {results_dir}")
    
    if not comparison:
        print("No benchmark results found to compare.")
        return None
    
    # Print comparison
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON ACROSS METHODS")
    print("="*100)
    
    if HAS_PANDAS:
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        print("="*100)
        
        # Save to CSV
        csv_path = './benchmark_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to {csv_path}")
        
        # Print summary statistics
        if len(comparison) > 1:
            print("\n" + "="*100)
            print("SUMMARY STATISTICS")
            print("="*100)
            
            latency_col = 'Latency (ms)'
            if latency_col in df.columns and df[latency_col].notna().any():
                fastest = df.loc[df[latency_col].idxmin()]
                slowest = df.loc[df[latency_col].idxmax()]
                
                print(f"\nFastest Method: {fastest['Method']} ({fastest[latency_col]:.4f} ms)")
                print(f"Slowest Method: {slowest['Method']} ({slowest[latency_col]:.4f} ms)")
                
                if fastest[latency_col] > 0:
                    speedup = slowest[latency_col] / fastest[latency_col]
                    print(f"Speedup: {speedup:.2f}x faster")
            
            if 'FLOPs (B)' in df.columns and df['FLOPs (B)'].notna().any():
                print(f"\nFLOPs Range: {df['FLOPs (B)'].min():.2f} B - {df['FLOPs (B)'].max():.2f} B")
            
            if 'Throughput (img/s)' in df.columns and df['Throughput (img/s)'].notna().any():
                max_throughput = df['Throughput (img/s)'].max()
                print(f"Max Throughput: {max_throughput:.2f} images/second")
    else:
        # Print without pandas
        print("\nMethod | Params (M) | FLOPs (B) | Latency (ms) | Throughput (img/s) | Batch Size")
        print("-" * 100)
        for row in comparison:
            params = f"{row['Params (M)']:.2f}" if row['Params (M)'] else "N/A"
            flops = f"{row['FLOPs (B)']:.2f}" if row['FLOPs (B)'] else "N/A"
            latency = f"{row['Latency (ms)']:.4f}" if row['Latency (ms)'] else "N/A"
            throughput = f"{row['Throughput (img/s)']:.2f}" if row['Throughput (img/s)'] else "N/A"
            batch = row['Batch Size']
            
            print(f"{row['Method']:6} | {params:10} | {flops:9} | {latency:12} | {throughput:17} | {batch:10}")
        
        # Save to JSON
        json_path = './benchmark_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Comparison saved to {json_path}")
    
    print("="*100)
    
    return comparison


def compare_batch_sizes(output_base='./output', method='zoh'):
    """Compare performance across different batch sizes for a single method"""
    results_dir = f"{output_base}/vim_tiny_{method}/benchmark_results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None
    
    results = parse_benchmark_results(results_dir)
    
    if not results:
        print(f"No benchmark results found in {results_dir}")
        return None
    
    # Group by batch size
    batch_comparison = []
    batch_sizes = sorted(set(r.get('batch_size', 1) for r in results))
    
    for bs in batch_sizes:
        batch_results = [r for r in results if r.get('batch_size') == bs]
        if batch_results:
            latest = batch_results[-1]
            latency = latest.get('latency', {})
            
            batch_comparison.append({
                'Batch Size': bs,
                'Latency (ms/img)': latency.get('avg_ms') if latency else None,
                'Throughput (img/s)': 1000 / latency.get('avg_ms', 1.0) if latency and latency.get('avg_ms') else None,
            })
    
    print(f"\n{'='*60}")
    print(f"BATCH SIZE COMPARISON: {method.upper()}")
    print(f"{'='*60}")
    
    if HAS_PANDAS:
        df = pd.DataFrame(batch_comparison)
        print(df.to_string(index=False))
    else:
        for row in batch_comparison:
            print(f"Batch {row['Batch Size']:3d}: {row['Latency (ms/img)']:.4f} ms/img, {row['Throughput (img/s)']:.2f} img/s")
    
    print("="*60)
    
    return batch_comparison


def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results across methods')
    parser.add_argument('--output-base', default='./output', type=str,
                        help='Base directory containing output folders')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size to compare (default: 1)')
    parser.add_argument('--compare-batch-sizes', type=str, default=None,
                        help='Compare batch sizes for a specific method (e.g., zoh)')
    
    args = parser.parse_args()
    
    if args.compare_batch_sizes:
        compare_batch_sizes(args.output_base, args.compare_batch_sizes)
    else:
        compare_methods(args.output_base, args.batch_size)


if __name__ == '__main__':
    main()

