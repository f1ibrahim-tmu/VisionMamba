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
train.init_checkpoint = f"{_out}/detection_logs/vim_tiny_vimdet_zoh/checkpoint.pth"
# Gradient clipping to prevent Inf/NaN divergence (ZOH can be numerically unstable)
train.clip_grad = dict(enabled=True, clip_type="norm", clip_value=1.0)

# Model configuration
model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = f"{_out}/classification_logs/vim_tiny_zoh/best_checkpoint.pth"
model.backbone.net.discretization_method = "zoh"  # Zero Order Hold discretization

# Enable activation checkpointing to reduce memory usage
model.backbone.net.use_act_checkpoint = True

# Reduce batch size to prevent OOM (from 64 to 32 for 8 GPUs = 4 images per GPU)
dataloader.train.total_batch_size = 32

# Ensure AMP is enabled (should already be enabled in base config, but make sure)
train.amp.enabled = True

# ZOH intentionally uses Mamba default ssm_cfg (dt_min=0.001, dt_max=0.1, dt_scale=1.0); no override.
# Optimizer lr and weight_decay overridden in scripts (optimizer.lr=1e-5, optimizer.weight_decay=0.01) to match segmentation.

optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)
