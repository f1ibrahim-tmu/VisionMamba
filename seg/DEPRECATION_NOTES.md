# Deprecation Notes

This document lists all deprecated APIs and their replacements in the modernized codebase.

## ⚠️ Deprecated APIs

### 1. Runner Classes

**Deprecated:**
```python
from mmengine.runner import IterBasedRunner, EpochBasedRunner

runner = dict(type='IterBasedRunner', max_iters=60000)
```

**Replacement:**
```python
from mmengine.runner import Runner

train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

runner = Runner(
    model=model,
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    val_cfg=val_cfg,
    test_cfg=test_cfg,
    ...
)
```

**Reason:** MMEngine ≥ 0.7 redesigned the runner system. `IterBasedRunner` and `EpochBasedRunner` no longer exist. Use `Runner` with loop configurations.

---

### 2. Dataloader Configuration

**Deprecated:**
```python
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(...),
    val=dict(...)
)
```

**Replacement:**
```python
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(...)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(...)
)
```

**Reason:** MMEngine requires explicit dataloader configurations for proper distributed training support.

---

### 3. Optimizer Configuration

**Deprecated:**
```python
optimizer = dict(type='AdamW', lr=1e-4)
optimizer_config = dict(
    type="DistOptimizerHook",
    use_fp16=False,
    grad_clip=None
)
```

**Replacement:**
```python
optimizer = dict(type='AdamW', lr=1e-4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=1.0)  # optional
)
```

**Reason:** MMEngine uses `optim_wrapper` to encapsulate optimizer and gradient handling.

---

### 4. Learning Rate Configuration

**Deprecated:**
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

**Replacement:**
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

**Reason:** MMEngine uses a list-based scheduler system for more flexible scheduling.

---

### 5. Hook Configuration

**Deprecated:**
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ]
)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=4)
```

**Replacement:**
```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=4
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
)
```

**Reason:** MMEngine organizes hooks into `default_hooks` and `custom_hooks` for better structure.

---

### 6. Apex AMP

**Deprecated:**
```python
import apex
model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
```

**Replacement:**
```python
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer
)
```

**Reason:** Apex is no longer maintained. MMEngine uses native PyTorch AMP via `AmpOptimWrapper`.

---

### 7. Testing APIs

**Deprecated:**
```python
from mmseg.apis import multi_gpu_test, single_gpu_test

if distributed:
    results = multi_gpu_test(model, data_loader, ...)
else:
    results = single_gpu_test(model, data_loader, ...)
```

**Replacement:**
```python
from mmengine.runner import Runner

runner = Runner(
    model=model,
    test_dataloader=test_dataloader,
    test_evaluator=test_evaluator,
    test_cfg=dict(type='TestLoop')
)

runner.test()
```

**Reason:** MMEngine's `Runner.test()` handles both single-GPU and multi-GPU testing automatically.

---

### 8. FP16 Configuration

**Deprecated:**
```python
fp16 = dict(loss_scale=512.0)
# or
optimizer_config = dict(use_fp16=True)
```

**Replacement:**
```python
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    loss_scale='dynamic'  # or a number
)
```

**Reason:** FP16 is now handled through `optim_wrapper`, not a separate config.

---

### 9. Workflow Configuration

**Deprecated:**
```python
workflow = [('train', 1), ('val', 1)]
```

**Replacement:**
```python
# No replacement needed - use train_dataloader and val_dataloader
# Validation is controlled by train_cfg.val_interval
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)
```

**Reason:** MMEngine handles training/validation loops through `train_cfg` and separate dataloaders.

---

### 10. Custom Runner Classes

**Deprecated:**
```python
from mmengine.runner import IterBasedRunner

@RUNNERS.register_module()
class IterBasedRunnerAmp(IterBasedRunner):
    ...
```

**Replacement:**
```python
# Use AmpOptimWrapper instead
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

**Reason:** Custom runners for AMP are no longer needed. Use `AmpOptimWrapper` in `optim_wrapper`.

---

## Migration Timeline

- **MMCV 1.x / MMEngine < 0.7**: Old APIs work
- **MMEngine ≥ 0.7**: Old APIs deprecated, new APIs required
- **MMSegmentation 1.x**: Old APIs removed, only new APIs supported

## Backward Compatibility

This codebase maintains **limited backward compatibility** for:
- `samples_per_gpu` (converted automatically, but deprecated)
- Legacy `runner` configs (converted automatically, but deprecated)
- Old `optimizer_config` (converted automatically, but deprecated)

However, **new configs should use the MMEngine format** for:
- Better performance
- Full feature support
- Future compatibility

## Getting Help

If you encounter deprecated API usage:

1. Check `seg/MMENGINE_CONFIG_GUIDE.md` for examples
2. See `MMCV_2x_MIGRATION_GUIDE.md` for detailed migration steps
3. Refer to [MMEngine Documentation](https://mmengine.readthedocs.io/)
4. Check `CRITICAL_MIGRATION_ISSUES.md` for known issues
