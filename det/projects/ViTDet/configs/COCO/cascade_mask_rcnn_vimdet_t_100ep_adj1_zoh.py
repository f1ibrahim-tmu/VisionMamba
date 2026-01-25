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
model.backbone.net.pretrained = "./output/classification_logs/vim_tiny_zoh/best_checkpoint.pth"
model.backbone.net.discretization_method = "zoh"  # Zero Order Hold discretization

optimizer.params.lr_factor_func = partial(get_vim_lr_decay_rate, num_layers=24, lr_decay_rate=0.837)
