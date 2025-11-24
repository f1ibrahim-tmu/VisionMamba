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

| MMCV 1.x | MMCV 2.x |
|----------|----------|
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
| `from mmcv.runner import build_runner` | Use `Runner` class directly |
| `mmcv.Config.fromfile()` | `mmengine.Config.fromfile()` |
| `mmcv.mkdir_or_exist()` | `mmengine.utils.mkdir_or_exist()` |
| `mmcv.dump()` | `mmengine.fileio.dump()` |
| `mmcv.is_list_of()` | `mmengine.utils.is_list_of()` |
| `mmcv.imrescale()` | `from mmcv.image import imrescale` |
| `mmcv.imresize()` | `from mmcv.image import imresize` |

### Key API Changes

1. **Runner API**: MMCV 2.x uses `Runner` class directly instead of `build_runner()`
2. **Optimizer**: Uses `optim_wrapper` instead of direct `optimizer`
3. **Hooks**: Hook registration and configuration changed
4. **Workflow**: Replaced with `train_dataloader` and `val_dataloader`

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

### Issue: Runner API errors

**Solution**: MMCV 2.x uses a different runner API. The runner is initialized directly with the model, optim_wrapper, and other parameters. Check the updated `train_api.py` for the new pattern.

## Additional Notes

1. **MMEngine Dependency**: MMCV 2.x requires MMEngine as a separate package. Make sure it's installed.

2. **Backward Compatibility**: Some functions have been aliased for backward compatibility (e.g., `is_module_wrapper` → `is_model_wrapper`), but it's recommended to update to the new names.

3. **Config Files**: Your existing config files should mostly work, but you may need to update some sections to match the new API expectations.

4. **Custom Hooks**: If you have custom hooks, they may need to be updated to work with MMEngine's hook system.

## References

- [MMCV 2.x Documentation](https://mmcv.readthedocs.io/en/2.x/)
- [MMEngine Documentation](https://mmengine.readthedocs.io/)
- [MMSegmentation Compatibility](https://github.com/open-mmlab/mmsegmentation)
- [MMDetection Compatibility](https://github.com/open-mmlab/mmdetection)

