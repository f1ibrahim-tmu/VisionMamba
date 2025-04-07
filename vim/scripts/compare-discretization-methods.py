#!/usr/bin/env python3
"""
This script evaluates and compares Vision Mamba models trained with different discretization methods.
It loads each model, evaluates it on the validation set, and compares their performance.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import Vision Mamba models
sys.path.append(str(Path(__file__).parent.parent))
from models_mamba import (
    vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,
    vim_tiny_patch16_224_bimambav2_foh,
    vim_tiny_patch16_224_bimambav2_bilinear,
    vim_tiny_patch16_224_bimambav2_poly,
    vim_tiny_patch16_224_bimambav2_highorder,
)

# Model configurations for each discretization method
MODEL_CONFIGS = {
    "ZOH (Default)": {
        "model_fn": vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,
        "checkpoint_path": "output/vim_tiny_zoh/checkpoint_best.pth",
    },
    "First Order Hold (FOH)": {
        "model_fn": vim_tiny_patch16_224_bimambav2_foh,
        "checkpoint_path": "output/vim_tiny_foh/checkpoint_best.pth",
    },
    "Bilinear (Tustin)": {
        "model_fn": vim_tiny_patch16_224_bimambav2_bilinear,
        "checkpoint_path": "output/vim_tiny_bilinear/checkpoint_best.pth",
    },
    "Polynomial Interpolation": {
        "model_fn": vim_tiny_patch16_224_bimambav2_poly,
        "checkpoint_path": "output/vim_tiny_poly/checkpoint_best.pth",
    },
    "Higher-Order Hold": {
        "model_fn": vim_tiny_patch16_224_bimambav2_highorder,
        "checkpoint_path": "output/vim_tiny_highorder/checkpoint_best.pth",
    },
}

def get_args_parser():
    parser = argparse.ArgumentParser(description='Vision Mamba Discretization Methods Comparison', add_help=False)
    parser.add_argument('--data-path', default='/Volumes/X10 Pro/datasets/imagenet-1k/validation', type=str, help='validation dataset path')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size for evaluation')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--output', default='./comparison_results', type=str, help='output directory for results')
    return parser

def build_val_dataset(args):
    # Define transforms for validation
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset directly from the validation directory
    val_dataset = datasets.ImageFolder(args.data_path, val_transforms)
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    return val_loader, len(val_dataset.classes)

def evaluate_model(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def plot_results(results, output_dir):
    # Prepare data for plotting
    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in methods]
    losses = [results[m]["loss"] for m in methods]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.bar(methods, accuracies, color='skyblue')
    plt.xlabel('Discretization Method')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison Across Discretization Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    
    # Plot loss comparison
    plt.figure(figsize=(12, 6))
    plt.bar(methods, losses, color='salmon')
    plt.xlabel('Discretization Method')
    plt.ylabel('Validation Loss')
    plt.title('Loss Comparison Across Discretization Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
    
    # Save results as CSV
    with open(os.path.join(output_dir, 'results.csv'), 'w') as f:
        f.write('Method,Accuracy,Loss\n')
        for method in methods:
            f.write(f'{method},{results[method]["accuracy"]:.2f},{results[method]["loss"]:.4f}\n')

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build validation dataset
    val_loader, num_classes = build_val_dataset(args)
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    # Dictionary to store results
    results = {}
    
    # Evaluate each model
    for method_name, config in MODEL_CONFIGS.items():
        print(f"\nEvaluating {method_name}...")
        
        # Check if checkpoint exists
        checkpoint_path = config["checkpoint_path"]
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...")
            continue
        
        # Load model
        model = config["model_fn"](num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(device)
        
        # Evaluate model
        loss, accuracy = evaluate_model(model, val_loader, device)
        print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Store results
        results[method_name] = {
            "loss": loss,
            "accuracy": accuracy
        }
    
    # Plot and save results
    if results:
        print("\nGenerating comparison plots...")
        plot_results(results, args.output)
        print(f"Results saved to {args.output}")
    else:
        print("\nNo results to display. Make sure models are trained and checkpoints exist.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vision Mamba Discretization Methods Comparison', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 