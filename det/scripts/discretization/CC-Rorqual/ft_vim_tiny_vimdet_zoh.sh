#!/bin/bash
# Zero Order Hold (ZOH) discretization for Vision Mamba detection on MS-COCO
#
# Usage: ./ft_vim_tiny_vimdet_zoh.sh [DATASET_PATH]
#   DATASET_PATH: Path to coco directory (e.g., /path/to/coco)
#                 If not provided, defaults to ./datasets
#   Note: Detectron2 expects COCO at ${DETECTRON2_DATASETS}/coco/

# Get dataset path from command line argument or use default
# Expected structure: ${DETECTRON2_DATASETS}/coco/ (so we use parent of coco directory)
if [ -z "$1" ]; then
    DETECTRON2_DATASETS="./datasets"
    echo "No dataset path provided, using default: ${DETECTRON2_DATASETS}"
else
    DETECTRON2_DATASETS="$(dirname "${1}")"
    echo "Using datasets root: ${DETECTRON2_DATASETS} (COCO at ${DETECTRON2_DATASETS}/coco/)"
fi

export DETECTRON2_DATASETS

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

# Multi-node setup
# Get the number of GPUs per node from SLURM
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-2}
# Get the number of nodes
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
# Get the node rank
NODE_RANK=${SLURM_PROCID:-0}
# Get master address (first node in the allocation)
if [ -z "$MASTER_ADDR" ]; then
    if [ -n "$SLURM_JOB_NODELIST" ]; then
        # Extract first hostname from SLURM_JOB_NODELIST
        MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    else
        MASTER_ADDR="localhost"
    fi
fi
export MASTER_ADDR

echo "Multi-node setup: $NUM_NODES nodes, $GPUS_PER_NODE GPUs per node"
echo "Node rank: $NODE_RANK, Master address: $MASTER_ADDR"

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh
DET_CONFIG=projects/ViTDet/configs/COCO/${DET_CONFIG_NAME}.py
PRETRAIN_CKPT=/home/f7ibrahi/links/projects/def-wangcs/f7ibrahi/projects/VisionMamba/output/classification_logs/vim_tiny_zoh/best_checkpoint.pth

# Set CUDA_VISIBLE_DEVICES to use all GPUs assigned by SLURM
# SLURM automatically sets CUDA_VISIBLE_DEVICES, but we ensure it's set correctly
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # If not set by SLURM, use all GPUs on this node
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
fi
export CUDA_VISIBLE_DEVICES

# For multi-node training, use torch.distributed.run with proper node configuration
if [ "$NUM_NODES" -gt 1 ]; then
    python -m torch.distributed.run \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        det/tools/lazyconfig_train_net.py \
        --config-file ${DET_CONFIG} \
        train.output_dir=output/detection_logs/vim_tiny_vimdet_zoh \
        train.init_checkpoint="" \
        dataloader.train.total_batch_size=32 \
        dataloader.train.num_workers=16 \
        dataloader.test.num_workers=8 \
        model.backbone.net.discretization_method=zoh \
        model.backbone.net.pretrained=${PRETRAIN_CKPT}
else
    # Single node training
    python -m torch.distributed.run \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        det/tools/lazyconfig_train_net.py \
        --config-file ${DET_CONFIG} \
        train.output_dir=output/detection_logs/vim_tiny_vimdet_zoh \
        train.init_checkpoint="" \
        dataloader.train.total_batch_size=32 \
        dataloader.train.num_workers=16 \
        dataloader.test.num_workers=8 \
        model.backbone.net.discretization_method=zoh \
        model.backbone.net.pretrained=${PRETRAIN_CKPT}
fi
