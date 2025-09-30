#!/bin/bash
# Bilinear (Tustin) discretization for Vision Mamba segmentation on ADE20K

# Activate environment (adjust path as needed)
# source /path/to/your/conda/bin/activate vim-seg
# cd /path/to/VisionMamba/seg

SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_60k_bilinear.py
PRETRAIN_CKPT=/path/to/pretrained_ckpts/pretrained-vim-t.pth

python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE:-1} --node_rank=${RANK:-0} --master_addr=${MASTER_ADDR:-localhost} --master_port=10297 \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t-bilinear --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} \
             model.backbone.if_bimamba=False \
             model.backbone.bimamba_type=v2 \
             model.backbone.discretization_method=bilinear \
             optimizer.lr=2e-4 \
             optimizer.weight_decay=0.1
