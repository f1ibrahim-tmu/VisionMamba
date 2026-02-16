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
# Gradient clipping aligned with ZOH and seg (1.0); 0.5 was starving detection heads
train.clip_grad = dict(enabled=True, clip_type="norm", clip_value=1.0)

# Longer warmup so detection heads can stabilize before backbone dominates
lr_multiplier.warmup_length = 1000 / train.max_iter  # 1000 iters vs default 250

model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "./output/vim_tiny_highorder/best_checkpoint.pth"
model.backbone.net.discretization_method = "highorder"  # Higher-Order Hold discretization
# Tighter dt range for highorder stability (aligned with seg)
model.backbone.net.ssm_cfg = dict(dt_min=0.0003, dt_max=0.015, dt_scale=0.2)
# Optimizer lr and weight_decay overridden in scripts (optimizer.lr=1e-5, optimizer.weight_decay=0.01) to match segmentation.

optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)
