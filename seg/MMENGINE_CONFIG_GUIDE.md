# MMEngine Configuration Guide

This guide documents the modern MMEngine ≥ 0.7 configuration format used in this codebase.

## Overview

This codebase has been modernized to use MMEngine ≥ 0.7 and MMCV 2.x, which introduces a new configuration system that replaces the old MMCV 1.x format.

## Key Configuration Changes

### 1. Training Loop Configuration

**Old Format (Deprecated):**
```python
runner = dict(type='IterBasedRunner', max_iters=60000)
```

**New Format (MMEngine ≥ 0.7):**
```python
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

**Important Notes:**
- `IterBasedRunner` no longer exists in MMEngine ≥ 0.7
- Use `IterBasedTrainLoop` for iteration-based training
- `train_cfg`, `val_cfg`, and `test_cfg` are separate configurations
- `val_interval` controls validation frequency

### 2. Dataloader Configuration

**Old Format (Deprecated):**
```python
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(...),
    val=dict(...)
)
```

**New Format (MMEngine):**
```python
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ADE20KDataset',
        data_root='...',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ADE20KDataset',
        data_root='...',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(...)
)
```

**Important Notes:**
- `samples_per_gpu` → `batch_size`
- `workers_per_gpu` → `num_workers`
- Explicit `sampler` configuration is required
- `persistent_workers=True` improves performance
- `InfiniteSampler` for training (iteration-based)
- `DefaultSampler` for validation/testing

### 3. Optimizer Configuration

**Old Format (Deprecated):**
```python
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optimizer_config = dict(
    type="DistOptimizerHook",
    use_fp16=False,
    grad_clip=None
)
```

**New Format (MMEngine):**
```python
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)

optim_wrapper = dict(
    type='OptimWrapper',  # or 'AmpOptimWrapper' for FP16
    optimizer=optimizer,
    clip_grad=dict(max_norm=1.0)  # optional gradient clipping
)
```

**For FP16 Training:**
```python
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Use this for FP16
    optimizer=optimizer
)
```

**Important Notes:**
- `optimizer_config` is deprecated
- Use `optim_wrapper` instead
- `AmpOptimWrapper` replaces apex AMP
- Gradient clipping is configured in `optim_wrapper.clip_grad`

### 4. Learning Rate Scheduler

**Old Format (Deprecated):**
```python
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)
```

**New Format (MMEngine):**
```python
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
        end=60000,
        by_epoch=False
    )
]
```

**Important Notes:**
- `lr_config` → `param_scheduler`
- `param_scheduler` is a list of scheduler configs
- Use `LinearLR` for warmup
- Use `PolyLR` for polynomial decay
- `begin` and `end` specify iteration/epoch ranges

### 5. Hooks Configuration

**Old Format (Deprecated):**
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ]
)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=4)
```

**New Format (MMEngine):**
```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=4
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_hooks = [
    dict(type='ThroughputHook', priority='NORMAL')
]
```

**Important Notes:**
- `log_config` → `default_hooks.logger`
- `checkpoint_config` → `default_hooks.checkpoint`
- Hooks are organized into `default_hooks` and `custom_hooks`
- Custom hooks must be registered with `@HOOKS.register_module()`

### 6. Evaluation Configuration

**Old Format (Deprecated):**
```python
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
```

**New Format (MMEngine):**
```python
evaluation = dict(
    interval=1000,
    metric='mIoU',
    save_best='mIoU',
    rule='greater'  # or 'less' for loss metrics
)
```

**Important Notes:**
- `rule` parameter specifies whether higher or lower is better
- Evaluation is configured in `train_cfg.val_interval`

## Complete Config Example

Here's a complete example of a modern MMEngine config:

```python
_base_ = [
    '../_base_/models/upernet_vim.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_200k.py'
]

# Model configuration
model = dict(...)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=200000, by_epoch=False)
]

# Training configuration
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Dataloaders
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True)
)

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=4),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# Evaluation
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU', rule='greater')
```

## Deprecated APIs

The following APIs are **deprecated** and should not be used:

1. ❌ `IterBasedRunner`, `EpochBasedRunner` → Use `Runner` with `train_cfg`
2. ❌ `samples_per_gpu` → Use `batch_size` in `train_dataloader`
3. ❌ `optimizer_config` → Use `optim_wrapper`
4. ❌ `lr_config` → Use `param_scheduler`
5. ❌ `workflow` → Use `train_dataloader`/`val_dataloader`
6. ❌ `apex` AMP → Use `AmpOptimWrapper`
7. ❌ `mmseg.apis.multi_gpu_test`/`single_gpu_test` → Use `Runner.test()`
8. ❌ `fp16` config key → Use `AmpOptimWrapper` in `optim_wrapper`

## Backward Compatibility

The codebase maintains backward compatibility with legacy configs:
- `samples_per_gpu` is still supported (but deprecated)
- Old `runner` configs are converted automatically
- Legacy optimizer configs are converted to `optim_wrapper`

However, **new configs should use the MMEngine format** for full compatibility and future-proofing.

## Migration Checklist

When updating your configs:

- [ ] Replace `runner` with `train_cfg`/`val_cfg`/`test_cfg`
- [ ] Convert `samples_per_gpu` to explicit `train_dataloader` configs
- [ ] Replace `optimizer_config` with `optim_wrapper`
- [ ] Convert `lr_config` to `param_scheduler` list
- [ ] Update `log_config` and `checkpoint_config` to `default_hooks`
- [ ] Remove any `apex` or `fp16` configs (use `AmpOptimWrapper`)
- [ ] Add `rule` parameter to `evaluation` config

## References

- [MMEngine Documentation](https://mmengine.readthedocs.io/)
- [MMSegmentation Migration Guide](https://mmsegmentation.readthedocs.io/en/latest/migration/)
- [MMCV 2.x Documentation](https://mmcv.readthedocs.io/en/2.x/)
