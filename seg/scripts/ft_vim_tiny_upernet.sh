#!/bin/bash
# bash /client-tools/repair_A100.sh
source /mnt/bn/lianghuidata/miniconda/bin/activate /mnt/bn/lianghuidata/miniconda/envs/vim-seg
cd /mnt/bn/lianghuidata/Vim/seg

SEG_CONFIG=configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k.py
PRETRAIN_CKPT=/mnt/bn/lianghuidata/Vim/pretrained_ckpts/pretrained-vim-t.pth

python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=10295 \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --work-dir work_dirs/vimseg-t --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT} model.backbone.if_bimamba=True model.backbone.bimamba_type=v2 optimizer.lr=1e-5 optimizer.weight_decay=0.01 train_cfg.max_iters=200000 