import os
from functools import partial

from .cascade_mask_rcnn_vimdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vim_lr_decay_rate,
)

_out = os.environ.get("OUTPUT_ROOT", ".")
train.init_checkpoint = f"{_out}/detection_logs/vim_tiny_vimdet_foh/checkpoint.pth"
# Gradient clipping aligned with ZOH and seg FOH (1.0); 0.5 was starving detection heads
train.clip_grad = dict(enabled=True, clip_type="norm", clip_value=1.0)

# Longer warmup for FOH so detection heads can stabilize before backbone dominates
lr_multiplier.warmup_length = 1000 / train.max_iter  # 1000 iters vs default 250

# Model configuration
model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = f"{_out}/classification_logs/vim_tiny_foh/best_checkpoint.pth"
model.backbone.net.discretization_method = "foh"  # First Order Hold discretization
# Tighter dt range for FOH stability (aligned with seg; avoids stagnant loss_mask)
model.backbone.net.ssm_cfg = dict(dt_min=0.0005, dt_max=0.03, dt_scale=0.3)
# Optimizer lr and weight_decay overridden in scripts (optimizer.lr=1e-5, optimizer.weight_decay=0.01) to match segmentation.

optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)
