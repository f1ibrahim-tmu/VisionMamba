# --------------------------------------------------------
# Vision Mamba Segmentation with Zero Order Hold (ZOH) Discretization
# Based on the original Vision Mamba implementation
# --------------------------------------------------------'
_base_ = [
    '../../_base_/models/upernet_vim.py', '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_200k.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='VisionMambaSeg',
        img_size=512, 
        patch_size=16, 
        in_chans=3,
        embed_dim=192, 
        depth=24,
        out_indices=[5, 11, 17, 23],
        pretrained=None,
        rms_norm=True,
        residual_in_fp32=False,
        fused_add_norm=True,
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        final_pool_type='all',
        if_divide_out=True,
        if_cls_token=False,
        discretization_method='zoh',  # Zero Order Hold discretization (uses defaults: dt_min=0.001, dt_max=0.1, dt_scale=1.0)
    ),
    decode_head=dict(
        in_channels=[192, 192, 192, 192],
        num_classes=150,
        channels=192,
    ),
    auxiliary_head=dict(
        in_channels=192,
        num_classes=150
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

# MMSeg 1.x format: constructor and paramwise_cfg go in optim_wrapper
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    constructor='VimLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.92)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=200000,
        by_epoch=False
    )
]

# By default, models are trained on 4 GPUs with 8 images per GPU
# MMEngine format: explicit dataloader configs (overrides base config)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True)
)

# Backward compatibility: keep old format
# MMEngine format: explicit dataloader configs (overrides base config)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True)
)

# Backward compatibility: keep old format
data=dict(samples_per_gpu=8, workers_per_gpu=4)
