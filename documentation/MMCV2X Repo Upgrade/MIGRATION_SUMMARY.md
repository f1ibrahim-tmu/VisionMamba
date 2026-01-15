# MMCV 2.x Migration Summary

## Overview

Your codebase has been updated to be compatible with MMCV 2.x, which is required for:
- PyTorch 2.1.0
- CUDA 12.2
- H100 GPUs

## What Was Changed

### Files Updated

#### Segmentation (`seg/`)
1. ✅ `seg/test.py` - Updated all MMCV imports to MMEngine
2. ✅ `seg/train.py` - Updated Config and utility imports
3. ✅ `seg/mmcv_custom/checkpoint.py` - Updated fileio, parallel, and utility imports
4. ✅ `seg/mmcv_custom/train_api.py` - Updated runner, optimizer, and hook APIs
5. ✅ `seg/mmcv_custom/layer_decay_optimizer_constructor.py` - Updated optimizer constructor
6. ✅ `seg/mmcv_custom/resize_transform.py` - Updated image processing functions
7. ✅ `seg/mmcv_custom/apex_runner/apex_iter_based_runner.py` - Updated runner imports
8. ✅ `seg/mmcv_custom/apex_runner/checkpoint.py` - Updated checkpoint utilities
9. ✅ `seg/mmcv_custom/apex_runner/optimizer.py` - Updated hook imports
10. ✅ `seg/tools/analysis_tools/benchmark.py` - Updated imports

#### Detection (`det/`)
1. ✅ `det/detectron2/modeling/mmdet_wrapper.py` - Updated ConfigDict import

#### Requirements
1. ✅ `seg/seg-requirements.txt` - Updated to use mmengine and mmcv 2.x

## Next Steps

### 1. Install MMCV 2.x and MMEngine

On your HPC cluster, run:

```bash
# Remove old MMCV installation
python -m pip uninstall -y mmcv mmcv-full mmcv-lite
rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/mmcv*

# Install MMEngine (required dependency)
python -m pip install mmengine

# Install MMCV 2.x for CUDA 12.2 and PyTorch 2.1.0
python -m pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu122/torch2.1/index.html
```

### 2. Verify Installation

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
print("✓ MMCV ops working correctly")
EOF
```

### 3. Test Your Scripts

Test your segmentation training script:

```bash
cd seg
python train.py configs/your_config.py --work-dir work_dirs/test
```

### 4. Update Your Scripts (if needed)

If you have custom scripts that directly import MMCV, you may need to update them. See `MMCV_2x_MIGRATION_GUIDE.md` for the complete migration reference.

## Key Changes to Be Aware Of

### Import Changes
- `mmcv.runner` → `mmengine.runner`
- `mmcv.parallel` → `mmengine.model`
- `mmcv.utils.Config` → `mmengine.config.Config`
- `mmcv.utils.DictAction` → `mmengine.argparse.DictAction`
- `mmcv.fileio` → `mmengine.fileio`
- **REMOVED**: `IterBasedRunner`, `EpochBasedRunner` (use `Runner` with `train_cfg`)
- **REMOVED**: `mmseg.apis.multi_gpu_test`, `single_gpu_test` (use `Runner.test()`)

### API Changes
- **Runner API**: Uses `Runner` class with `train_cfg`/`val_cfg`/`test_cfg` instead of `IterBasedRunner`/`EpochBasedRunner`
- **Training Loops**: `IterBasedRunner` → `IterBasedTrainLoop` in `train_cfg`
- **Optimizer**: Uses `optim_wrapper` instead of `optimizer` + `optimizer_config`
- **Dataloaders**: `samples_per_gpu` → explicit `train_dataloader`/`val_dataloader` configs
- **Learning Rate**: `lr_config` → `param_scheduler` (list format)
- **Hooks**: `log_config`/`checkpoint_config` → `default_hooks` dict
- **AMP**: Apex AMP → `AmpOptimWrapper` in `optim_wrapper`
- **Testing**: `multi_gpu_test`/`single_gpu_test` → `Runner.test()`

### Config Format Changes
- `runner` → `train_cfg`/`val_cfg`/`test_cfg`
- `samples_per_gpu` → `train_dataloader.batch_size`
- `optimizer_config` → `optim_wrapper`
- `lr_config` → `param_scheduler`
- `log_config`/`checkpoint_config` → `default_hooks`
- `workflow` → **REMOVED** (use dataloaders directly)
- `fp16` → `AmpOptimWrapper` in `optim_wrapper`

## Potential Issues

### 1. Config File Compatibility
Your existing config files should mostly work, but you may need to:
- Update `runner` configuration sections
- Update `optimizer` to `optim_wrapper` format (if needed)
- Update hook configurations

### 2. Custom Hooks
If you have custom hooks, they must:
- Inherit from `mmengine.hooks.Hook` (not `mmcv.runner.Hook`)
- Be registered with `@HOOKS.register_module()` from `mmengine.registry`
- Use MMEngine hook lifecycle methods

### 3. MMSegmentation/MMDetection Versions
Ensure your `mmsegmentation` and `mmdetection` versions are compatible with MMCV 2.x:
- **MMSegmentation ≥ 1.0.0** supports MMCV 2.x and MMEngine ≥ 0.7
- **MMDetection ≥ 3.0.0** supports MMCV 2.x and MMEngine ≥ 0.7

### 4. Config File Updates Required
All config files should be updated to use:
- `train_cfg`/`val_cfg`/`test_cfg` instead of `runner`
- `train_dataloader`/`val_dataloader` instead of `samples_per_gpu`
- `optim_wrapper` instead of `optimizer_config`
- `param_scheduler` instead of `lr_config`
- `default_hooks` instead of `log_config`/`checkpoint_config`

See `seg/MMENGINE_CONFIG_GUIDE.md` for complete examples.

## Documentation

- See `MMCV_2x_MIGRATION_GUIDE.md` for detailed migration information
- [MMCV 2.x Documentation](https://mmcv.readthedocs.io/en/2.x/)
- [MMEngine Documentation](https://mmengine.readthedocs.io/)

## Support

If you encounter issues:
1. Check the error message - it usually indicates which import needs updating
2. Refer to `MMCV_2x_MIGRATION_GUIDE.md` for the import mapping
3. Check MMCV/MMEngine documentation for API changes

