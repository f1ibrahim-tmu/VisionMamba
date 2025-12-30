#!/bin/bash
# Quick setup script for Weights & Biases
# Usage: ./setup_wandb.sh

echo "=========================================="
echo "Weights & Biases Setup"
echo "=========================================="
echo ""

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
else
    echo "✓ wandb is already installed"
fi

# Login to W&B
echo ""
echo "Logging in to Weights & Biases..."
echo "Using API key: 964b6fd194cced2fc3fabd18754dd54218145929"
wandb login 964b6fd194cced2fc3fabd18754dd54218145929

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now use W&B in your training scripts by adding:"
echo "  --use-wandb"
echo ""
echo "Example:"
echo "  python vim/main.py --use-wandb --wandb-project visionmamba ..."
echo ""
echo "For more information, see WANDB_SETUP.md"
echo ""

