# MMCV 1.7.2 to MMCV 2.x Migration Guide

This guide documents the changes needed to migrate from MMCV 1.7.2 to MMCV 2.x for compatibility with:
- PyTorch 2.1.0
- CUDA 12.2
- H100 GPUs

## Installation

### Step 1: Remove Old MMCV Installation

```bash
python -m pip uninstall -y mmcv mmcv-full mmcv-lite
rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/mmcv*
```

### Step 2: Install MMCV 2.x and MMEngine

```bash
# Install MMEngine (required for MMCV 2.x)
python -m pip install mmengine

# Install MMCV 2.x compatible with PyTorch 2.1.0 and CUDA 12.2
python -m pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu122/torch2.1/index.html
```

### Step 3: Verify Installation

```bash
python << 'EOF'
import mmcv
import mmengine
import torch
from mmcv.ops import DeformConv2d

print(f"MMCV Version: {mmcv.__version__}")
print(f"MMEngine Version: {mmengine.__version__}")
print(f"Torch Version: {torch.__version__}")

# Test CUDA operations
x = torch.randn(1, 3, 10, 10).cuda()
offset = torch.randn(1, 18, 10, 10).cuda()
conv = DeformConv2d(3, 3, 3, padding=1).cuda()
y = conv(x, offset)
print("OK: MMCV ops working")
EOF
```

## Code Changes Summary

### Import Changes

| MMCV 1.x | MMCV 2.x / MMEngine ≥ 0.7 |
|----------|---------------------------|
| `from mmcv.runner import ...` | `from mmengine.runner import ...` |
| `from mmcv.parallel import ...` | `from mmengine.model import ...` |
| `from mmcv.utils import Config` | `from mmengine.config import Config` |
| `from mmcv.utils import DictAction` | `from mmengine.argparse import DictAction` |
| `from mmcv.utils import mkdir_or_exist` | `from mmengine.utils import mkdir_or_exist` |
| `from mmcv.fileio import ...` | `from mmengine.fileio import ...` |
| `from mmcv.runner import get_dist_info` | `from mmengine.dist import get_dist_info` |
| `from mmcv.runner import init_dist` | `from mmengine.dist import init_dist` |
| `from mmcv.runner import load_checkpoint` | `from mmengine.runner import load_checkpoint` |
| `from mmcv.runner import wrap_fp16_model` | `from mmengine.runner import wrap_fp16_model` |
| `from mmcv.parallel import MMDataParallel` | `from mmengine.model import MMDataParallel` |
| `from mmcv.parallel import MMDistributedDataParallel` | `from mmengine.model import MMDistributedDataParallel` |
| `from mmcv.parallel import is_module_wrapper` | `from mmengine.model import is_model_wrapper` |
| `from mmcv.runner import build_optimizer` | `from mmengine.optim import build_optim_wrapper` |
| `from mmcv.runner import IterBasedRunner` | **REMOVED** - Use `Runner` with `train_cfg` |
| `from mmcv.runner import EpochBasedRunner` | **REMOVED** - Use `Runner` with `train_cfg` |
| `from mmseg.apis import multi_gpu_test` | **REMOVED** - Use `Runner.test()` |
| `from mmseg.apis import single_gpu_test` | **REMOVED** - Use `Runner.test()` |
| `mmcv.Config.fromfile()` | `mmengine.Config.fromfile()` |
| `mmcv.mkdir_or_exist()` | `mmengine.utils.mkdir_or_exist()` |
| `mmcv.dump()` | `mmengine.fileio.dump()` |
| `mmcv.is_list_of()` | `mmengine.utils.is_list_of()` |
| `mmcv.imrescale()` | `from mmcv.image import imrescale` |
| `mmcv.imresize()` | `from mmcv.image import imresize` |

### Config Format Changes

| Old Format (MMCV 1.x) | New Format (MMEngine ≥ 0.7) |
|----------------------|----------------------------|
| `runner = dict(type='IterBasedRunner', max_iters=60000)` | `train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)` |
| `data = dict(samples_per_gpu=8, workers_per_gpu=16)` | `train_dataloader = dict(batch_size=8, num_workers=16, sampler=dict(type='InfiniteSampler'))` |
| `optimizer_config = dict(type="DistOptimizerHook", use_fp16=False)` | `optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)` |
| `lr_config = dict(policy='poly', power=0.9, min_lr=1e-4)` | `param_scheduler = [dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=60000)]` |
| `log_config = dict(interval=50, hooks=[...])` | `default_hooks = dict(logger=dict(type='LoggerHook', interval=50))` |
| `checkpoint_config = dict(interval=1000, max_keep_ckpts=4)` | `default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=4))` |
| `workflow = [('train', 1)]` | **REMOVED** - Use `train_dataloader`/`val_dataloader` |
| `fp16 = dict(...)` | `optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)` |

### Key API Changes

1. **Runner API**: MMEngine ≥ 0.7 uses `Runner` with `train_cfg`/`val_cfg`/`test_cfg` instead of `IterBasedRunner`/`EpochBasedRunner`
2. **Training Loops**: `IterBasedRunner` → `IterBasedTrainLoop` in `train_cfg`
3. **Optimizer**: Uses `optim_wrapper` instead of direct `optimizer` + `optimizer_config`
4. **Dataloaders**: `samples_per_gpu` → explicit `train_dataloader`/`val_dataloader` configs
5. **Learning Rate**: `lr_config` → `param_scheduler` (list format)
6. **Hooks**: `log_config`/`checkpoint_config` → `default_hooks` dict
7. **AMP**: Apex AMP → `AmpOptimWrapper` in `optim_wrapper`
8. **Testing**: `mmseg.apis.multi_gpu_test` → `Runner.test()`

## Files Updated

### Segmentation (`seg/`)

1. **seg/test.py**
   - Updated imports from `mmcv.runner` to `mmengine.runner`
   - Updated `mmcv.Config` to `mmengine.Config`
   - Updated utility functions to use `mmengine` equivalents

2. **seg/train.py**
   - Updated imports to use `mmengine`
   - Updated `Config` and `DictAction` imports

3. **seg/mmcv_custom/checkpoint.py**
   - Updated fileio imports
   - Updated `is_module_wrapper` to `is_model_wrapper`
   - Updated utility functions

4. **seg/mmcv_custom/train_api.py**
   - Updated parallel imports
   - Changed optimizer building to use `optim_wrapper`
   - Updated runner initialization to use `Runner` class directly
   - Updated hook registration

5. **seg/mmcv_custom/layer_decay_optimizer_constructor.py**
   - Updated optimizer constructor imports
   - Changed registry from `OPTIMIZER_BUILDERS` to `OPTIM_WRAPPER_CONSTRUCTORS`

6. **seg/mmcv_custom/resize_transform.py**
   - Updated image processing functions
   - Updated utility functions

7. **seg/mmcv_custom/apex_runner/**
   - Updated all imports to use `mmengine`
   - Updated checkpoint and optimizer hooks

8. **seg/tools/analysis_tools/benchmark.py**
   - Updated imports to use `mmengine`

### Detection (`det/`)

1. **det/detectron2/modeling/mmdet_wrapper.py**
   - Updated `ConfigDict` import from `mmengine.config`

## Requirements Files

Update your requirements files to include:

```
mmengine>=0.10.0
mmcv>=2.0.0
```

Note: The exact versions depend on your mmsegmentation and mmdetection versions. Check their compatibility matrices.

## Testing

After migration, test your scripts:

```bash
# Test segmentation training
cd seg
python train.py configs/your_config.py --work-dir work_dirs/test

# Test segmentation inference
python test.py configs/your_config.py checkpoint.pth --eval mIoU
```

## Common Issues and Solutions

### Issue: `ImportError: cannot import name '_ext' from 'mmcv._ext'`

**Solution**: This means you have an incompatible MMCV installation. Follow the installation steps above to install the correct MMCV 2.x version.

### Issue: `AttributeError: module 'mmcv' has no attribute 'Config'`

**Solution**: Use `mmengine.Config` instead of `mmcv.Config`.

### Issue: `ModuleNotFoundError: No module named 'mmcv.runner'`

**Solution**: Import from `mmengine.runner` instead of `mmcv.runner`.

### Issue: `ImportError: cannot import name 'IterBasedRunner' from 'mmengine.runner'`

**Solution**: `IterBasedRunner` no longer exists in MMEngine ≥ 0.7. Use `Runner` with `train_cfg=dict(type='IterBasedTrainLoop', ...)` instead.

### Issue: `ImportError: cannot import name 'multi_gpu_test' from 'mmseg.apis'`

**Solution**: These functions are removed in MMSegmentation 1.x. Use `Runner.test()` instead. See updated `test.py` for example.

### Issue: Runner API errors

**Solution**: MMEngine ≥ 0.7 uses a different runner API. The runner is initialized with:
- `train_cfg`/`val_cfg`/`test_cfg` instead of `runner` config
- `train_dataloader`/`val_dataloader` instead of building manually
- `optim_wrapper` instead of `optimizer` + `optimizer_config`

Check the updated `train_api.py` for the new pattern.

### Issue: Dataloader configuration errors

**Solution**: Use explicit `train_dataloader`/`val_dataloader` configs instead of `samples_per_gpu`. See `MMENGINE_CONFIG_GUIDE.md` for examples.

### Issue: Optimizer configuration errors

**Solution**: Use `optim_wrapper` instead of `optimizer_config`. For FP16, use `AmpOptimWrapper` in `optim_wrapper`, not apex.

### Issue: Learning rate scheduler errors

**Solution**: Convert `lr_config` to `param_scheduler` list format. See `MMENGINE_CONFIG_GUIDE.md` for examples.

## Additional Notes

1. **MMEngine Dependency**: MMCV 2.x requires MMEngine ≥ 0.7 as a separate package. Make sure it's installed.

2. **Backward Compatibility**: Limited backward compatibility is maintained for:
   - `samples_per_gpu` (automatically converted, but deprecated)
   - Legacy `runner` configs (automatically converted, but deprecated)
   - Old `optimizer_config` (automatically converted, but deprecated)
   
   However, **new configs should use the MMEngine format** for full compatibility.

3. **Config Files**: Your existing config files will need updates:
   - Replace `runner` with `train_cfg`/`val_cfg`/`test_cfg`
   - Convert `samples_per_gpu` to explicit `train_dataloader` configs
   - Replace `optimizer_config` with `optim_wrapper`
   - Convert `lr_config` to `param_scheduler`
   - Update hooks to `default_hooks` format

4. **Apex Removal**: Apex AMP is no longer supported. Use `AmpOptimWrapper` in `optim_wrapper` for FP16 training.

5. **Testing API**: `mmseg.apis.multi_gpu_test` and `single_gpu_test` are removed. Use `Runner.test()` instead.

## Documentation

For complete configuration examples and migration details, see:
- `seg/MMENGINE_CONFIG_GUIDE.md` - Complete MMEngine config format guide
- `seg/DEPRECATION_NOTES.md` - All deprecated APIs and replacements
- `seg/CRITICAL_MIGRATION_ISSUES.md` - Known issues and fixes
- [MMEngine Documentation](https://mmengine.readthedocs.io/)
- [MMSegmentation Migration Guide](https://mmsegmentation.readthedocs.io/en/latest/migration/)

4. **Custom Hooks**: If you have custom hooks, they may need to be updated to work with MMEngine's hook system.

## References

- [MMCV 2.x Documentation](https://mmcv.readthedocs.io/en/2.x/)
- [MMEngine Documentation](https://mmengine.readthedocs.io/)
- [MMSegmentation Compatibility](https://github.com/open-mmlab/mmsegmentation)
- [MMDetection Compatibility](https://github.com/open-mmlab/mmdetection)

