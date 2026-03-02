#!/bin/bash
# bash /client-tools/repair_A100.sh
source /mnt/bn/lianghuidata/miniconda/bin/activate /mnt/bn/lianghuidata/miniconda/envs/vim-seg
cd /mnt/bn/lianghuidata/Vim/seg

SEG_CONFIG=configs/vim/upernet/upernet_vim_small_24_512_slide_200k.py
TRAINED_CKPT=/mnt/bn/lianghuidata/ckpts/vim/seg/vim-s-upernet-iter-60000.pth

# Init backward Mamba params from forward when loading unidirectional ckpt (default: true). Set INIT_BACKWARD_FROM_FORWARD=false to disable.
INIT_BACKWARD_FROM_FORWARD=${INIT_BACKWARD_FROM_FORWARD:-true}

python test.py ${SEG_CONFIG} ${TRAINED_CKPT} --eval mIoU \
    --options model.backbone.init_backward_from_forward=${INIT_BACKWARD_FROM_FORWARD} model.backbone.if_bimamba=True model.backbone.bimamba_type=v2 optim_wrapper.optimizer.lr=1e-5 model.backbone.use_residual_as_feature=True model.backbone.last_layer_process=add optim_wrapper.paramwise_cfg.layer_decay_rate=0.95