#!/usr/bin/env python3
"""
Master comparison script for all Vision Mamba discretization experiments.
This script compares results across all tasks: classification, segmentation, and detection.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def get_args_parser():
    parser = argparse.ArgumentParser('Compare All Vision Mamba Discretization Methods', add_help=False)
    parser.add_argument('--vim-output', default='vim/output', type=str, help='Path to VIM classification output')
    parser.add_argument('--seg-work-dirs', default='seg/work_dirs', type=str, help='Path to segmentation work directories')
    parser.add_argument('--det-work-dirs', default='det/work_dirs', type=str, help='Path to detection work directories')
    parser.add_argument('--output', default='./all_discretization_comparison', type=str, help='Output directory for results')
    return parser

def load_classification_results(vim_output_dir):
    """Load classification results from VIM output directory"""
    results = []
    methods = ['zoh', 'foh', 'bilinear', 'poly', 'highorder', 'rk4']
    
    for method in methods:
        method_dir = Path(vim_output_dir) / f'vim_tiny_{method}'
        if method_dir.exists():
            # Look for log files or checkpoint files
            log_files = list(method_dir.glob("*.txt"))
            if log_files:
                try:
                    with open(log_files[0], 'r') as f:
                        lines = f.readlines()
                        # Extract final accuracy from the last few lines
                        for line in reversed(lines[-20:]):
                            if 'Accuracy' in line and 'test' in line:
                                parts = line.strip().split()
                                for i, part in enumerate(parts):
                                    if 'acc1' in part and i + 1 < len(parts):
                                        acc = float(parts[i + 1].rstrip('%'))
                                        results.append({
                                            'task': 'Classification',
                                            'method': method.upper(),
                                            'metric': 'Top-1 Accuracy',
                                            'value': acc
                                        })
                                        break
                except Exception as e:
                    print(f"Error loading classification results for {method}: {e}")
    
    return results

def load_segmentation_results(seg_work_dirs):
    """Load segmentation results from work directories"""
    results = []
    methods = {
        'zoh': 'vimseg-t-zoh',
        'foh': 'vimseg-t-foh',
        'bilinear': 'vimseg-t-bilinear',
        'poly': 'vimseg-t-poly',
        'highorder': 'vimseg-t-highorder',
        'rk4': 'vimseg-t-rk4'
    }
    
    for method, work_dir_name in methods.items():
        work_dir = Path(seg_work_dirs) / work_dir_name
        if work_dir.exists():
            log_files = list(work_dir.glob("*.log"))
            if log_files:
                try:
                    with open(log_files[0], 'r') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-50:]):
                            if 'mIoU' in line:
                                parts = line.strip().split()
                                for i, part in enumerate(parts):
                                    if 'mIoU' in part and i + 1 < len(parts):
                                        miou = float(parts[i + 1].rstrip('%'))
                                        results.append({
                                            'task': 'Segmentation',
                                            'method': method.upper(),
                                            'metric': 'mIoU',
                                            'value': miou
                                        })
                                        break
                except Exception as e:
                    print(f"Error loading segmentation results for {method}: {e}")
    
    return results

def load_detection_results(det_work_dirs):
    """Load detection results from work directories"""
    results = []
    methods = {
        'zoh': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh-4gpu',
        'foh': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_foh-4gpu',
        'bilinear': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_bilinear-4gpu',
        'poly': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_poly-4gpu',
        'highorder': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_highorder-4gpu',
        'rk4': 'cascade_mask_rcnn_vimdet_t_100ep_adj1_rk4-4gpu'
    }
    
    for method, work_dir_name in methods.items():
        work_dir = Path(det_work_dirs) / work_dir_name
        if work_dir.exists():
            log_files = list(work_dir.glob("*.log"))
            if log_files:
                try:
                    with open(log_files[0], 'r') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-100:]):
                            if 'bbox AP' in line:
                                parts = line.strip().split()
                                for i, part in enumerate(parts):
                                    if 'bbox' in part and 'AP' in part and i + 1 < len(parts):
                                        ap = float(parts[i + 1].rstrip('%'))
                                        results.append({
                                            'task': 'Detection',
                                            'method': method.upper(),
                                            'metric': 'Bbox AP',
                                            'value': ap
                                        })
                                        break
                except Exception as e:
                    print(f"Error loading detection results for {method}: {e}")
    
    return results

def compare_all_methods(args):
    """Compare all discretization methods across all tasks"""
    
    print("Loading results from all tasks...")
    
    # Load results from all tasks
    classification_results = load_classification_results(args.vim_output)
    segmentation_results = load_segmentation_results(args.seg_work_dirs)
    detection_results = load_detection_results(args.det_work_dirs)
    
    # Combine all results
    all_results = classification_results + segmentation_results + detection_results
    
    if not all_results:
        print("No results found! Make sure the models have been trained.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to CSV
    csv_path = output_dir / 'all_discretization_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Performance by task
    task_metrics = df.groupby(['task', 'method'])['value'].mean().unstack(fill_value=0)
    task_metrics.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Performance by Task and Method')
    axes[0, 0].set_ylabel('Performance Score')
    axes[0, 0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Method comparison across tasks
    method_performance = df.groupby('method')['value'].mean().sort_values(ascending=False)
    method_performance.plot(kind='bar', ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title('Average Performance by Method (All Tasks)')
    axes[0, 1].set_ylabel('Average Performance Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Task-specific performance
    for i, task in enumerate(df['task'].unique()):
        task_data = df[df['task'] == task]
        if i == 0:
            ax = axes[1, 0]
        else:
            ax = axes[1, 1]
        
        method_scores = task_data.groupby('method')['value'].mean().sort_values(ascending=False)
        method_scores.plot(kind='bar', ax=ax, color=plt.cm.Set3(i))
        ax.set_title(f'{task} Performance by Method')
        ax.set_ylabel('Performance Score')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = output_dir / 'all_discretization_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE DISCRETIZATION METHODS COMPARISON")
    print("="*80)
    
    # Summary by task
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        print(f"\n{task.upper()} RESULTS:")
        print("-" * 40)
        task_summary = task_data.groupby('method')['value'].agg(['mean', 'std']).round(2)
        print(task_summary)
        
        # Best method for this task
        best_method = task_data.loc[task_data['value'].idxmax()]
        print(f"Best method: {best_method['method']} ({best_method['value']:.2f})")
    
    # Overall best method
    overall_best = df.loc[df['value'].idxmax()]
    print(f"\nOVERALL BEST PERFORMANCE:")
    print(f"Method: {overall_best['method']}")
    print(f"Task: {overall_best['task']}")
    print(f"Score: {overall_best['value']:.2f}")
    
    # Save detailed results as JSON
    json_path = output_dir / 'all_discretization_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to {json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compare All Vision Mamba Discretization Methods', parents=[get_args_parser()])
    args = parser.parse_args()
    compare_all_methods(args)
