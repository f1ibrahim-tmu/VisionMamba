#!/bin/bash
#SBATCH --job-name=vimseg-bilinear
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vimseg-bilinear-%j.out
#SBATCH --error=logs/vimseg-bilinear-%j.err
#SBATCH --gres=gpu:4

# Load modules on Compute Canada
module load python/3.10
module load cuda/11.3

# Activate your virtual environment
source vim_seg_env/bin/activate

# Navigate to project directory
cd /project/6062393/f7ibrahi/projects/VisionMamba

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_bilinear.py
PRETRAIN_CKPT=/home/f7ibrahi/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/vim_tiny_bilinear/best_checkpoint.pth

# Create logs directory
mkdir -p logs

# Run training with Slurm launcher
srun python seg/train.py --launcher slurm \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-bilinear --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=bilinear \
             optimizer.lr=0.001 \
             optimizer.weight_decay=0.05
