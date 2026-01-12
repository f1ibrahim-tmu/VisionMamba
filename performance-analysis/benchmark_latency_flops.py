#!/usr/bin/env python
"""
Benchmark script to measure inference latency (ms/image) and FLOPs for Vision Mamba models.
Usage:
    python performance-analysis/benchmark_latency_flops.py \
        --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
        --checkpoint ./output/classification_logs/vim_tiny_zoh/best_checkpoint.pth \
        --batch-size 1 \
        --input-size 224 \
        --num-samples 1000 \
        --output-dir ./output/classification_logs/vim_tiny_zoh/benchmark_results
"""

import argparse
import time
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import vim modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Also add vim directory to path so rope can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vim'))

import torch
import torch.backends.cudnn as cudnn

# Try to import fvcore first, then fall back to thop
FVCORE_AVAILABLE = False
THOP_AVAILABLE = False
FlopCountAnalyzer = None
thop_profile = None

try:
    from fvcore.nn import FlopCountAnalyzer
    from fvcore.common.timer import Timer
    # Verify FlopCountAnalyzer is actually available (some fvcore versions don't have it)
    if FlopCountAnalyzer is not None:
        FVCORE_AVAILABLE = True
        print("✓ Using fvcore for FLOPs computation")
    else:
        raise ImportError("FlopCountAnalyzer not available in this fvcore version")
except (ImportError, AttributeError) as e:
    # Try thop as fallback
    try:
        from thop import profile, clever_format
        thop_profile = profile
        THOP_AVAILABLE = True
        print("✓ Using thop for FLOPs computation (fvcore not available or incompatible)")
    except ImportError:
        print("⚠ Neither fvcore nor thop available. FLOPs computation will be disabled.")
        print("   Install one of: pip install fvcore  or  pip install thop")

# Import vim modules
import vim.models_mamba as models_mamba
import vim.utils as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Vision Mamba Benchmark', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
                        type=str, help='Model name')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--input-size', default=224, type=int, help='Input image size')
    parser.add_argument('--num-classes', type=int, default=None, 
                        help='Number of classes (auto-detected from checkpoint if not specified)')
    
    # Benchmark parameters
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--num-samples', default=1000, type=int, help='Number of samples to benchmark')
    parser.add_argument('--warmup-iters', default=10, type=int, help='Number of warmup iterations')
    parser.add_argument('--num-repeats', default=5, type=int, help='Number of timing repeats')
    
    # Output parameters
    parser.add_argument('--output-dir', default='./benchmark_results', type=str, help='Output directory for results')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')
    parser.add_argument('--seed', default=0, type=int)
    
    return parser


def load_model(args):
    """Load model from checkpoint"""
    print(f"Creating model: {args.model}")
    
    if not hasattr(models_mamba, args.model):
        raise ValueError(f"Model {args.model} not found in models_mamba. Available models: {[m for m in dir(models_mamba) if m.startswith('vim_')]}")
    
    # Load checkpoint first to detect num_classes if needed
    num_classes = args.num_classes
    state_dict = None
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from distributed training)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Auto-detect num_classes from checkpoint if not specified
        if num_classes is None and 'head.weight' in state_dict:
            num_classes = state_dict['head.weight'].shape[0]
            print(f"✓ Auto-detected num_classes={num_classes} from checkpoint")
        elif num_classes is None and 'head.bias' in state_dict:
            num_classes = state_dict['head.bias'].shape[0]
            print(f"✓ Auto-detected num_classes={num_classes} from checkpoint")
    
    # Create model with detected/specified num_classes
    if num_classes is not None:
        print(f"Creating model with num_classes={num_classes}")
        model = models_mamba.__dict__[args.model](pretrained=False, num_classes=num_classes, img_size=args.input_size)
    else:
        model = models_mamba.__dict__[args.model](pretrained=False, img_size=args.input_size)
    
    # Load checkpoint if available
    if state_dict is not None:
        # Filter out incompatible keys (size mismatches)
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                if model_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_keys.append(f"{k} (shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape})")
            else:
                skipped_keys.append(f"{k} (not in model)")
        
        if skipped_keys:
            print(f"⚠ Skipping {len(skipped_keys)} incompatible keys:")
            for key in skipped_keys[:5]:  # Show first 5
                print(f"   - {key}")
            if len(skipped_keys) > 5:
                print(f"   ... and {len(skipped_keys) - 5} more")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✓ Checkpoint loaded successfully ({len(filtered_state_dict)}/{len(state_dict)} keys matched)")
    else:
        print(f"⚠ No checkpoint found, using randomly initialized weights")
    
    model = model.to(args.device)
    model.eval()
    return model


def estimate_discretization_flops(model, discretization_method, batch_size=1, input_size=224):
    """
    Estimate additional FLOPs for discretization methods that profilers miss.
    
    The profiler sees the same architecture regardless of discretization method,
    but different methods have different computational costs in the discretization step.
    
    Returns: Additional FLOPs in billions
    """
    try:
        # Extract model dimensions
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # Get discretization method from first layer
            first_layer = model.layers[0]
            if hasattr(first_layer, 'mixer') and hasattr(first_layer.mixer, 'discretization_method'):
                discretization_method = first_layer.mixer.discretization_method
            elif hasattr(first_layer, 'mixer') and hasattr(first_layer.mixer, 'ssm') and hasattr(first_layer.mixer.ssm, 'discretization_method'):
                discretization_method = first_layer.mixer.ssm.discretization_method
        
        # Get model dimensions
        depth = len(model.layers) if hasattr(model, 'layers') else 24
        embed_dim = model.embed_dim if hasattr(model, 'embed_dim') else 192
        d_state = 16  # Default, could extract from model if available
        num_patches = model.patch_embed.num_patches if hasattr(model, 'patch_embed') else (input_size // 16) ** 2
        
        # Sequence length (including cls token if present)
        seq_len = num_patches
        if hasattr(model, 'if_cls_token') and model.if_cls_token:
            seq_len += 1
        
        # Base discretization FLOPs per layer (ZOH baseline)
        # For each layer: delta (B, D, L) × A (D, N) -> (B, D, L, N)
        # einsum('bdl,dn->bdln') = B * D * L * N operations
        base_discretization_flops_per_layer = batch_size * embed_dim * seq_len * d_state
        
        # Additional FLOPs multipliers for each method (relative to ZOH)
        # These are rough estimates based on the mathematical operations
        method_multipliers = {
            'zoh': 1.0,           # Baseline: exp + einsum
            'foh': 1.5,           # exp + power operations (delta^2, delta^3, etc.)
            'bilinear': 2.5,      # Matrix inversions (I + A*delta/2)^-1
            'poly': 2.0,          # Multiple power operations and matrix multiplications
            'highorder': 3.0,     # Higher-order Taylor series terms
            'rk4': 4.0,           # 4 function evaluations per step
        }
        
        multiplier = method_multipliers.get(discretization_method.lower(), 1.0)
        
        # Calculate total discretization FLOPs
        # Additional FLOPs = (multiplier - 1) * base_flops (since base is already counted)
        additional_flops_per_layer = (multiplier - 1.0) * base_discretization_flops_per_layer
        total_additional_flops = additional_flops_per_layer * depth
        
        # Convert to billions
        additional_flops_giga = total_additional_flops / 1e9
        
        return additional_flops_giga, discretization_method
        
    except Exception as e:
        # If we can't extract, return 0
        return 0.0, 'unknown'


def compute_flops(model, args):
    """Compute FLOPs for the model using fvcore or thop, with manual discretization FLOPs"""
    if not FVCORE_AVAILABLE and not THOP_AVAILABLE:
        print("⚠ No FLOPs computation library available, skipping FLOPs computation")
        return None
    
    try:
        print("\nComputing FLOPs...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size, device=args.device)
        
        base_flops_giga = None
        profiler_name = None
        
        # Use fvcore if available
        if FVCORE_AVAILABLE and FlopCountAnalyzer is not None:
            flops_analyzer = FlopCountAnalyzer(model, dummy_input)
            total_flops = flops_analyzer.total()
            base_flops_giga = total_flops / 1e9
            profiler_name = "fvcore"
        
        # Fall back to thop
        elif THOP_AVAILABLE and thop_profile is not None:
            flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)
            base_flops_giga = flops / 1e9
            profiler_name = "thop"
        else:
            print("⚠ FLOPs computation libraries not properly initialized")
            return None
        
        # Extract discretization method and estimate additional FLOPs
        discretization_method = None
        if hasattr(model, 'layers') and len(model.layers) > 0:
            first_layer = model.layers[0]
            if hasattr(first_layer, 'mixer'):
                if hasattr(first_layer.mixer, 'discretization_method'):
                    discretization_method = first_layer.mixer.discretization_method
                elif hasattr(first_layer.mixer, 'ssm') and hasattr(first_layer.mixer.ssm, 'discretization_method'):
                    discretization_method = first_layer.mixer.ssm.discretization_method
        
        # If we can't find it, try to infer from model name
        if discretization_method is None:
            model_name_lower = args.model.lower()
            if 'rk4' in model_name_lower:
                discretization_method = 'rk4'
            elif 'foh' in model_name_lower:
                discretization_method = 'foh'
            elif 'bilinear' in model_name_lower:
                discretization_method = 'bilinear'
            elif 'poly' in model_name_lower:
                discretization_method = 'poly'
            elif 'highorder' in model_name_lower:
                discretization_method = 'highorder'
            else:
                discretization_method = 'zoh'
        
        # Estimate additional discretization FLOPs
        additional_flops_giga, detected_method = estimate_discretization_flops(
            model, discretization_method, batch_size=1, input_size=args.input_size
        )
        
        # Total FLOPs = base (from profiler) + additional (from discretization)
        total_flops_giga = base_flops_giga + additional_flops_giga
        
        print(f"✓ Base FLOPs ({profiler_name}): {base_flops_giga:.4f} B (Billion)")
        if additional_flops_giga > 0:
            print(f"✓ Additional discretization FLOPs ({detected_method.upper()}): {additional_flops_giga:.4f} B (Billion)")
            print(f"✓ Total FLOPs (with discretization): {total_flops_giga:.4f} B (Billion)")
        else:
            print(f"✓ Total FLOPs: {total_flops_giga:.4f} B (Billion)")
        print(f"✓ FLOPs per image: {total_flops_giga:.4f} B")
        
        # Return dictionary with detailed FLOPs breakdown
        return {
            'total': total_flops_giga,
            'base': base_flops_giga,
            'additional_discretization': additional_flops_giga,
            'discretization_method': detected_method,
            'profiler': profiler_name
        }
            
    except Exception as e:
        print(f"✗ Error computing FLOPs: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_latency(model, args):
    """Benchmark inference latency"""
    print(f"\nBenchmarking latency with batch_size={args.batch_size}, {args.num_samples} samples...")
    
    # Create dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=args.device)
    
    # Warmup
    print(f"Warming up for {args.warmup_iters} iterations...")
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            _ = model(dummy_input)
    
    if args.device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {args.num_repeats} repeats with {args.num_samples // args.num_repeats} samples each...")
    latencies = []
    
    with torch.no_grad():
        for repeat in range(args.num_repeats):
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Run inference on num_samples images
            num_iters = args.num_samples // args.batch_size
            for _ in range(num_iters):
                _ = model(dummy_input)
            
            if args.device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            latency_per_image_ms = (elapsed / args.num_samples) * 1000
            latencies.append(latency_per_image_ms)
            
            print(f" Repeat {repeat + 1}/{args.num_repeats}: {latency_per_image_ms:.4f} ms/image")
    
    # Statistics
    latencies = sorted(latencies)
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    median_latency = latencies[len(latencies) // 2]
    
    print(f"\n✓ Latency Statistics:")
    print(f" - Min: {min_latency:.4f} ms/image")
    print(f" - Max: {max_latency:.4f} ms/image")
    print(f" - Avg: {avg_latency:.4f} ms/image")
    print(f" - Median: {median_latency:.4f} ms/image")
    
    return {
        'min_ms': min_latency,
        'max_ms': max_latency,
        'avg_ms': avg_latency,
        'median_ms': median_latency,
    }


def get_model_info(model):
    """Get model size and parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    model_size_mb = (total_params * 4) / (1024 * 1024) # Assuming float32
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
    }


def extract_method_name(output_dir):
    """Extract method name from output directory path"""
    # Try to extract from path like: .../vim_tiny_zoh/benchmark_results
    path_parts = os.path.normpath(output_dir).split(os.sep)
    for part in reversed(path_parts):
        if 'vim_tiny_' in part:
            method = part.replace('vim_tiny_', '')
            return method
    # Fallback: try to extract from any part containing method names
    methods = ['zoh', 'foh', 'bilinear', 'poly', 'highorder', 'rk4']
    for part in path_parts:
        for method in methods:
            if method in part.lower():
                return method
    return 'unknown'


def save_results(results, output_dir, method=None, batch_size=None):
    """Save benchmark results to JSON with descriptive filename"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract method name if not provided
    if method is None:
        method = extract_method_name(output_dir)
    
    # Get batch_size from results if not provided
    if batch_size is None:
        batch_size = results.get('batch_size', 'unknown')
    
    # Create results filename: benchmark_method_batchsize_timestamp.json
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'benchmark_{method}_bs{batch_size}_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    return results_file


def print_summary(results):
    """Print summary of benchmark results"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if 'model_info' in results:
        info = results['model_info']
        print(f"\nModel Information:")
        print(f" - Total Parameters: {info['total_params']:,}")
        print(f" - Trainable Parameters: {info['trainable_params']:,}")
        print(f" - Model Size: {info['model_size_mb']:.2f} MB")
    
    if 'flops' in results and results['flops']:
        print(f"\nComputational Complexity:")
        if isinstance(results['flops'], dict):
            # Detailed FLOPs breakdown
            flops_info = results['flops']
            print(f" - Base FLOPs ({flops_info.get('profiler', 'profiler')}): {flops_info.get('base', 0):.4f} B")
            if flops_info.get('additional_discretization', 0) > 0:
                print(f" - Additional Discretization FLOPs ({flops_info.get('discretization_method', 'unknown').upper()}): {flops_info.get('additional_discretization', 0):.4f} B")
            print(f" - Total FLOPs: {flops_info.get('total', 0):.4f} B (Billion)")
        else:
            # Legacy format (just a number)
            print(f" - FLOPs: {results['flops']:.2f} B (Billion)")
    
    if 'latency' in results:
        lat = results['latency']
        print(f"\nInference Latency:")
        print(f" - Average: {lat['avg_ms']:.4f} ms/image")
        print(f" - Median: {lat['median_ms']:.4f} ms/image")
        print(f" - Min: {lat['min_ms']:.4f} ms/image")
        print(f" - Max: {lat['max_ms']:.4f} ms/image")
    
    # Calculate throughput
    if 'latency' in results:
        throughput_img_s = 1000 / results['latency']['avg_ms']
        print(f"\nThroughput:")
        print(f" - {throughput_img_s:.2f} images/second")
    
    print("="*60)


def main(args):
    print(f"\n{'='*60}")
    print("Vision Mamba Latency & FLOPs Benchmark")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load model
    model = load_model(args)
    
    # Get model info
    model_info = get_model_info(model)
    
    # Compute FLOPs
    flops = compute_flops(model, args)
    
    # Benchmark latency
    latency = benchmark_latency(model, args)
    
    # Compile results
    results = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'batch_size': args.batch_size,
        'input_size': args.input_size,
        'num_samples': args.num_samples,
        'warmup_iters': args.warmup_iters,
        'num_repeats': args.num_repeats,
        'device': args.device,
        'model_info': model_info,
        'flops': flops,
        'latency': latency,
    }
    
    # Print summary
    print_summary(results)
    
    # Save results with method name and batch size in filename
    method = extract_method_name(args.output_dir)
    save_results(results, args.output_dir, method=method, batch_size=args.batch_size)
    
    return results


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    main(args)

