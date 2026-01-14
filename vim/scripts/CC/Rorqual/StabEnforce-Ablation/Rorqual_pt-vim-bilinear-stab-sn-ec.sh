#!/bin/bash

# Feature-StabEnforce: Spectral Normalization + Eigenvalue Clamping - Bilinear discretization
# conda activate conda_visionmamba
# Get the project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Accept seed as first argument, default to 0
SEED=${1:-0}

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT"

# Generate unique port based on SLURM job ID (if available) or use process ID
# Port range: 29500-29999 (500 ports available)
if [ -n "$SLURM_JOB_ID" ]; then
    MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 500))
else
    # Fallback: use process ID if not in SLURM environment
    MASTER_PORT=$((29500 + $$ % 500))
fi
export MASTER_PORT

echo "Using MASTER_PORT=$MASTER_PORT for job ${SLURM_JOB_ID:-$$}"
echo "Running Spectral Normalization + Eigenvalue Clamping with Bilinear discretization, seed $SEED"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port $MASTER_PORT \
    ./main.py \
    --model vim_tiny_patch16_224_bimambav2_bilinear \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --num_workers 2 \
    --input-size 32 \
    --data-set CIFAR \
    --data-path /home/f7ibrahi/links/scratch/dataset/cifar100 \
    --seed $SEED \
    --use-spectral-normalization \
    --use-eigenvalue-clamping \
    --stability-epsilon 0.01 \
    --stability-penalty-weight 0.1 \
    --output_dir /home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_bilinear_stab_sn_ec_seed${SEED} \
    --resume /home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_bilinear_stab_sn_ec_seed${SEED}/checkpoint.pth
