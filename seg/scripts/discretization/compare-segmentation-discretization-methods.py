#!/usr/bin/env python3
"""
This script evaluates and compares Vision Mamba segmentation models trained with different discretization methods.
It loads each model, evaluates it on the ADE20K validation set, and compares their performance.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

# Add the seg directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

def get_args_parser():
    parser = argparse.ArgumentParser('Compare Vision Mamba Segmentation Discretization Methods', add_help=False)
    parser.add_argument('--work-dirs', default='work_dirs', type=str, help='Path to work directories')
    parser.add_argument('--output', default='./segmentation_discretization_comparison', type=str, help='Output directory for results')
    parser.add_argument('--config-dir', default='configs/vim/upernet', type=str, help='Path to config files')
    return parser

def load_model_results(work_dir, method_name):
    """Load results from a trained model's work directory"""
    results = {}
    
    # Look for log files and checkpoints
    log_files = list(Path(work_dir).glob("*.log"))
    checkpoint_files = list(Path(work_dir).glob("*.pth"))
    
    if log_files:
        # Parse log file for metrics
        log_file = log_files[0]
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Extract final mIoU and other metrics from the last few lines
                for line in reversed(lines[-50:]):  # Check last 50 lines
                    if 'mIoU' in line or 'mAcc' in line or 'aAcc' in line:
                        # Parse metrics from log line
                        parts = line.strip().split()
                        for i, part in enumerate(parts):
                            if 'mIoU' in part and i + 1 < len(parts):
                                results['mIoU'] = float(parts[i + 1].rstrip(','))
                            elif 'mAcc' in part and i + 1 < len(parts):
                                results['mAcc'] = float(parts[i + 1].rstrip(','))
                            elif 'aAcc' in part and i + 1 < len(parts):
                                results['aAcc'] = float(parts[i + 1].rstrip(','))
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
    
    # Add method name
    results['method'] = method_name
    results['work_dir'] = str(work_dir)
    
    return results

def compare_segmentation_methods(args):
    """Compare all segmentation discretization methods"""
    
    # Define the methods and their expected work directories
    methods = {
        'ZOH (Default)': 'vimseg-t-zoh',
        'First Order Hold (FOH)': 'vimseg-t-foh', 
        'Bilinear (Tustin)': 'vimseg-t-bilinear',
        'Polynomial Interpolation': 'vimseg-t-poly',
        'Higher-Order Hold': 'vimseg-t-highorder',
        'Runge-Kutta 4th Order (RK4)': 'vimseg-t-rk4'
    }
    
    results = []
    
    print("Loading results from trained models...")
    for method_name, work_dir_name in methods.items():
        work_dir = Path(args.work_dirs) / work_dir_name
        if work_dir.exists():
            print(f"Loading results for {method_name}...")
            result = load_model_results(work_dir, method_name)
            results.append(result)
        else:
            print(f"Warning: Work directory {work_dir} not found for {method_name}")
    
    if not results:
        print("No results found! Make sure the models have been trained.")
        return
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to CSV
    csv_path = output_dir / 'segmentation_discretization_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create comparison plots
    if 'mIoU' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Plot mIoU comparison
        plt.subplot(2, 2, 1)
        plt.bar(df['method'], df['mIoU'])
        plt.title('mIoU Comparison Across Discretization Methods')
        plt.ylabel('mIoU (%)')
        plt.xticks(rotation=45, ha='right')
        
        # Plot mAcc comparison if available
        if 'mAcc' in df.columns:
            plt.subplot(2, 2, 2)
            plt.bar(df['method'], df['mAcc'])
            plt.title('mAcc Comparison Across Discretization Methods')
            plt.ylabel('mAcc (%)')
            plt.xticks(rotation=45, ha='right')
        
        # Plot aAcc comparison if available
        if 'aAcc' in df.columns:
            plt.subplot(2, 2, 3)
            plt.bar(df['method'], df['aAcc'])
            plt.title('aAcc Comparison Across Discretization Methods')
            plt.ylabel('aAcc (%)')
            plt.xticks(rotation=45, ha='right')
        
        # Combined metrics plot
        plt.subplot(2, 2, 4)
        x = np.arange(len(df))
        width = 0.25
        
        if 'mIoU' in df.columns:
            plt.bar(x - width, df['mIoU'], width, label='mIoU')
        if 'mAcc' in df.columns:
            plt.bar(x, df['mAcc'], width, label='mAcc')
        if 'aAcc' in df.columns:
            plt.bar(x + width, df['aAcc'], width, label='aAcc')
        
        plt.title('All Metrics Comparison')
        plt.ylabel('Score (%)')
        plt.xticks(x, df['method'], rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plot_path = output_dir / 'segmentation_discretization_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {plot_path}")
        plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("SEGMENTATION DISCRETIZATION METHODS COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    
    # Find best method
    if 'mIoU' in df.columns:
        best_method = df.loc[df['mIoU'].idxmax()]
        print(f"\nBest performing method: {best_method['method']}")
        print(f"Best mIoU: {best_method['mIoU']:.2f}%")
    
    # Save detailed results as JSON
    json_path = output_dir / 'segmentation_discretization_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compare Vision Mamba Segmentation Discretization Methods', parents=[get_args_parser()])
    args = parser.parse_args()
    compare_segmentation_methods(args)
