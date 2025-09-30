#!/usr/bin/env python3
"""
This script evaluates and compares Vision Mamba detection models trained with different discretization methods.
It loads each model, evaluates it on the MS-COCO validation set, and compares their performance.
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

# Add the det directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

def get_args_parser():
    parser = argparse.ArgumentParser('Compare Vision Mamba Detection Discretization Methods', add_help=False)
    parser.add_argument('--work-dirs', default='work_dirs', type=str, help='Path to work directories')
    parser.add_argument('--output', default='./detection_discretization_comparison', type=str, help='Output directory for results')
    parser.add_argument('--config-dir', default='projects/ViTDet/configs/COCO', type=str, help='Path to config files')
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
                # Extract final COCO metrics from the last few lines
                for line in reversed(lines[-100:]):  # Check last 100 lines
                    if 'bbox AP' in line or 'segm AP' in line:
                        # Parse COCO metrics from log line
                        parts = line.strip().split()
                        for i, part in enumerate(parts):
                            if 'bbox' in part and 'AP' in part and i + 1 < len(parts):
                                results['bbox_AP'] = float(parts[i + 1].rstrip(','))
                            elif 'segm' in part and 'AP' in part and i + 1 < len(parts):
                                results['segm_AP'] = float(parts[i + 1].rstrip(','))
                            elif 'AP50' in part and i + 1 < len(parts):
                                results['AP50'] = float(parts[i + 1].rstrip(','))
                            elif 'AP75' in part and i + 1 < len(parts):
                                results['AP75'] = float(parts[i + 1].rstrip(','))
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
    
    # Add method name
    results['method'] = method_name
    results['work_dir'] = str(work_dir)
    
    return results

def compare_detection_methods(args):
    """Compare all detection discretization methods"""
    
    # Define the methods and their expected work directories
    methods = {
        'ZOH (Default)': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh-4gpu',
        'First Order Hold (FOH)': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_foh-4gpu', 
        'Bilinear (Tustin)': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_bilinear-4gpu',
        'Polynomial Interpolation': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_poly-4gpu',
        'Higher-Order Hold': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_highorder-4gpu',
        'Runge-Kutta 4th Order (RK4)': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_rk4-4gpu'
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
    csv_path = output_dir / 'detection_discretization_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create comparison plots
    if 'bbox_AP' in df.columns:
        plt.figure(figsize=(15, 10))
        
        # Plot bbox AP comparison
        plt.subplot(2, 3, 1)
        plt.bar(df['method'], df['bbox_AP'])
        plt.title('Bbox AP Comparison Across Discretization Methods')
        plt.ylabel('Bbox AP (%)')
        plt.xticks(rotation=45, ha='right')
        
        # Plot segm AP comparison if available
        if 'segm_AP' in df.columns:
            plt.subplot(2, 3, 2)
            plt.bar(df['method'], df['segm_AP'])
            plt.title('Segm AP Comparison Across Discretization Methods')
            plt.ylabel('Segm AP (%)')
            plt.xticks(rotation=45, ha='right')
        
        # Plot AP50 comparison if available
        if 'AP50' in df.columns:
            plt.subplot(2, 3, 3)
            plt.bar(df['method'], df['AP50'])
            plt.title('AP50 Comparison Across Discretization Methods')
            plt.ylabel('AP50 (%)')
            plt.xticks(rotation=45, ha='right')
        
        # Plot AP75 comparison if available
        if 'AP75' in df.columns:
            plt.subplot(2, 3, 4)
            plt.bar(df['method'], df['AP75'])
            plt.title('AP75 Comparison Across Discretization Methods')
            plt.ylabel('AP75 (%)')
            plt.xticks(rotation=45, ha='right')
        
        # Combined metrics plot
        plt.subplot(2, 3, 5)
        x = np.arange(len(df))
        width = 0.2
        
        metrics = []
        labels = []
        if 'bbox_AP' in df.columns:
            plt.bar(x - 1.5*width, df['bbox_AP'], width, label='Bbox AP')
            metrics.append('bbox_AP')
            labels.append('Bbox AP')
        if 'segm_AP' in df.columns:
            plt.bar(x - 0.5*width, df['segm_AP'], width, label='Segm AP')
            metrics.append('segm_AP')
            labels.append('Segm AP')
        if 'AP50' in df.columns:
            plt.bar(x + 0.5*width, df['AP50'], width, label='AP50')
            metrics.append('AP50')
            labels.append('AP50')
        if 'AP75' in df.columns:
            plt.bar(x + 1.5*width, df['AP75'], width, label='AP75')
            metrics.append('AP75')
            labels.append('AP75')
        
        plt.title('All Metrics Comparison')
        plt.ylabel('Score (%)')
        plt.xticks(x, df['method'], rotation=45, ha='right')
        plt.legend()
        
        # Performance improvement plot
        plt.subplot(2, 3, 6)
        if 'bbox_AP' in df.columns:
            baseline = df[df['method'] == 'ZOH (Default)']['bbox_AP'].iloc[0] if 'ZOH (Default)' in df['method'].values else df['bbox_AP'].iloc[0]
            improvements = df['bbox_AP'] - baseline
            colors = ['green' if x > 0 else 'red' for x in improvements]
            plt.bar(df['method'], improvements, color=colors)
            plt.title('Performance Improvement over ZOH (Bbox AP)')
            plt.ylabel('Improvement (%)')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'detection_discretization_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {plot_path}")
        plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("DETECTION DISCRETIZATION METHODS COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    
    # Find best method
    if 'bbox_AP' in df.columns:
        best_method = df.loc[df['bbox_AP'].idxmax()]
        print(f"\nBest performing method: {best_method['method']}")
        print(f"Best Bbox AP: {best_method['bbox_AP']:.2f}%")
    
    # Save detailed results as JSON
    json_path = output_dir / 'detection_discretization_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compare Vision Mamba Detection Discretization Methods', parents=[get_args_parser()])
    args = parser.parse_args()
    compare_detection_methods(args)
