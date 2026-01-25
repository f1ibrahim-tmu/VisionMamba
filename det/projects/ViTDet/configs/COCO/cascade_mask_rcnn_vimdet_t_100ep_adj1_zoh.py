from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .cascade_mask_rcnn_vimdet_b_100ep import (
    dataloader,
    model,
    train,
    optimizer,
    get_vim_lr_decay_rate,
)

train.init_checkpoint = "./output/detection_logs/vim_tiny_vimdet_zoh/checkpoint.pth"

# Model configuration
model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "./output/classification_logs/vim_tiny_zoh/best_checkpoint.pth"
model.backbone.net.discretization_method = "zoh"  # Zero Order Hold discretization

# Enable activation checkpointing to reduce memory usage
model.backbone.net.use_act_checkpoint = True

# Reduce batch size to prevent OOM (from 64 to 32 for 8 GPUs = 4 images per GPU)
# Note: Script overrides this to 16 (4 per GPU with 4 GPUs)
dataloader.train.total_batch_size = 32

# Ensure AMP is enabled (should already be enabled in base config, but make sure)
train.amp.enabled = True

# Enable gradient clipping to prevent training divergence (NaN/Inf)
# Very aggressive clipping for ZOH discretization stability
train.clip_grad = dict(
    enabled=True,
    clip_type="norm",  # "norm" or "value"
    clip_value=0.1,   # Very aggressive: reduced from 0.5 to 0.1 for ZOH stability
    norm_type=2.0     # L2 norm
)

# Reduce learning rate significantly to prevent divergence
# Default is 1e-4, reducing to 1e-5 (10x reduction) for ZOH stability
# ZOH is more numerically unstable than other discretization methods
optimizer.lr = 1e-5

# Increase warmup period significantly (from 250 to 10000 iterations)
# ZOH needs very long warmup to stabilize before full learning rate kicks in
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=10000 / train.max_iter,  # Increased from 2000 to 10000 iterations for ZOH
    warmup_factor=0.00001,  # Very small: reduced from 0.0001 to 0.00001 (LR starts at 1e-10)
)

optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)
