#!/usr/bin/env python3
"""
Compare Vision Mamba detection runs across discretization methods.

Reads metrics.json (Detectron2 line-delimited JSON) from each run's output dir and
extracts the last COCO evaluation (bbox/AP, segm/AP, AP50, AP75). Supports Rorqual
and Fir output layouts (--layout rorqual|fir, --work-dirs pointing to the base dir
that contains vim_tiny_rorqual_vimdet_* or vim_tiny_fir_vimdet_* subdirs).

Example:
  python compare-detection-discretization-methods.py --work-dirs output/detection_logs --layout rorqual
  python compare-detection-discretization-methods.py --work-dirs /path/to/rorqual_detection_logs --layout rorqual
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add the det directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Subdir names used by CC-Rorqual and CC-Fir detection scripts (output/detection_logs/<subdir>)
LAYOUT_SUBDIRS = {
    "rorqual": {
        "ZOH (Default)": "vim_tiny_rorqual_vimdet_zoh",
        "First Order Hold (FOH)": "vim_tiny_rorqual_vimdet_foh",
        "Bilinear (Tustin)": "vim_tiny_rorqual_vimdet_bilinear",
        "Polynomial Interpolation": "vim_tiny_rorqual_vimdet_poly",
        "Higher-Order Hold": "vim_tiny_rorqual_vimdet_highorder",
        "Runge-Kutta 4th Order (RK4)": "vim_tiny_rorqual_vimdet_rk4",
    },
    "fir": {
        "ZOH (Default)": "vim_tiny_fir_vimdet_zoh",
        "First Order Hold (FOH)": "vim_tiny_fir_vimdet_foh",
        "Bilinear (Tustin)": "vim_tiny_fir_vimdet_bilinear",
        "Polynomial Interpolation": "vim_tiny_fir_vimdet_poly",
        "Higher-Order Hold": "vim_tiny_fir_vimdet_highorder",
        "Runge-Kutta 4th Order (RK4)": "vim_tiny_fir_vimdet_rk4",
    },
}


def get_args_parser():
    parser = argparse.ArgumentParser('Compare Vision Mamba Detection Discretization Methods', add_help=False)
    parser.add_argument(
        '--work-dirs',
        default='output/detection_logs',
        type=str,
        help='Base path to detection run dirs (e.g. output/detection_logs or /path/to/rorqual_detection_logs)',
    )
    parser.add_argument(
        '--layout',
        default='rorqual',
        choices=['rorqual', 'fir'],
        type=str,
        help='Output dir layout: rorqual (vim_tiny_rorqual_vimdet_*) or fir (vim_tiny_fir_vimdet_*)',
    )
    parser.add_argument('--output', default='./detection_discretization_comparison', type=str, help='Output directory for results')
    parser.add_argument('--config-dir', default='projects/ViTDet/configs/COCO', type=str, help='Path to config files')
    return parser


def load_model_results_from_metrics_json(work_dir, method_name):
    """
    Load COCO metrics from a run's metrics.json (Detectron2 line-delimited JSON).
    Uses the last evaluation line (contains bbox/AP, segm/AP) as the final result.
    """
    results = {"method": method_name, "work_dir": str(work_dir)}
    metrics_file = Path(work_dir) / "metrics.json"
    if not metrics_file.exists():
        return results
    last_eval = None
    try:
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if not line or '"bbox/AP":' not in line:
                    continue
                try:
                    obj = json.loads(line)
                    if "bbox/AP" in obj:
                        last_eval = obj
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return results
    if last_eval:
        results["bbox_AP"] = last_eval["bbox/AP"]
        results["segm_AP"] = last_eval.get("segm/AP")
        results["AP50"] = last_eval.get("bbox/AP50")
        results["AP75"] = last_eval.get("bbox/AP75")
        results["iteration"] = last_eval.get("iteration")
    return results

def compare_detection_methods(args):
    """Compare all detection discretization methods using metrics.json from each run."""
    methods = LAYOUT_SUBDIRS[args.layout]
    results = []

    print(f"Layout: {args.layout}, work-dirs: {args.work_dirs}")
    print("Loading results from metrics.json...")
    for method_name, work_dir_name in methods.items():
        work_dir = Path(args.work_dirs) / work_dir_name
        if work_dir.exists():
            print(f"  Loading {method_name} from {work_dir}...")
            result = load_model_results_from_metrics_json(work_dir, method_name)
            results.append(result)
        else:
            print(f"  Warning: Work directory {work_dir} not found for {method_name}")
    
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
            zoh_row = df[df['method'] == 'ZOH (Default)']
            if len(zoh_row) and pd.notna(zoh_row['bbox_AP'].iloc[0]):
                baseline = zoh_row['bbox_AP'].iloc[0]
            else:
                baseline = df['bbox_AP'].dropna().iloc[0] if df['bbox_AP'].notna().any() else 0.0
            improvements = df['bbox_AP'].fillna(0) - baseline
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
        plt.close()
    
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
