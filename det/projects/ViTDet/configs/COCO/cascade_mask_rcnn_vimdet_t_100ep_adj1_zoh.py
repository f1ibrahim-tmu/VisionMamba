from functools import partial

from .cascade_mask_rcnn_vimdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vim_lr_decay_rate,
)

train.init_checkpoint = "./output/detection_logs/vim_tiny_vimdet_zoh/checkpoint.pth"

# Model configuration
model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "./output/vim_tiny_zoh/best_checkpoint.pth"
model.backbone.net.discretization_method = "zoh"  # Zero Order Hold discretization

# Enable activation checkpointing to reduce memory usage
model.backbone.net.use_act_checkpoint = True

# Reduce batch size to prevent OOM (from 64 to 32 for 8 GPUs = 4 images per GPU)
dataloader.train.total_batch_size = 32

# Ensure AMP is enabled (should already be enabled in base config, but make sure)
train.amp.enabled = True

# Enable gradient clipping to prevent training divergence (NaN/Inf)
train.clip_grad = dict(
    enabled=True,
    clip_type="norm",  # "norm" or "value"
    clip_value=1.0,   # max gradient norm
    norm_type=2.0     # L2 norm
)

optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)
