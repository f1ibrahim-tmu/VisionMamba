from functools import partial

from .cascade_mask_rcnn_vimdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vim_lr_decay_rate,
)

train.init_checkpoint = "./output/detection_logs/vim_tiny_vimdet_highorder/checkpoint.pth"
# Stricter gradient clipping for non-ZOH stability (helps prevent Inf/NaN and mask-head collapse)
train.clip_grad = dict(enabled=True, clip_type="norm", clip_value=0.5)

model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "./output/vim_tiny_highorder/best_checkpoint.pth"
model.backbone.net.discretization_method = "highorder"  # Higher-Order Hold discretization
# Tighter dt range for highorder stability (aligned with seg)
model.backbone.net.ssm_cfg = dict(dt_min=0.0003, dt_max=0.015, dt_scale=0.2)
optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)
