# MMCV 2.x Compatibility Fix

## Issue

You're getting this error:
```
AssertionError: MMCV==2.1.0 is used but incompatible. Please install mmcv>=(1, 3, 13, 0, 0, 0), <(1, 8, 0, 0, 0, 0).
```

## Root Cause

The issue is that `mmsegmentation==0.29.1` only supports MMCV 1.x (versions 1.3.13 to 1.8.0). To use MMCV 2.x, you need to upgrade `mmsegmentation` to version 1.0.0 or higher.

## Solution

### Option 1: Upgrade mmsegmentation (Recommended for MMCV 2.x)

If you're on the `mmcv-2.x` branch and want to use MMCV 2.x:

```bash
# Uninstall old mmsegmentation
pip uninstall mmsegmentation

# Install mmsegmentation 1.0.0+ which supports MMCV 2.x
pip install mmsegmentation>=1.0.0

# Or install from source if needed
cd seg/mmsegmentation
git checkout v1.0.0  # or latest version
pip install -e .
```

**Note:** mmsegmentation 1.0.0+ may have API changes. You may need to update your code accordingly.

### Option 2: Use MMCV 1.7.2 (Keep current setup)

If you want to stay with `mmsegmentation==0.29.1`, you must use MMCV 1.7.2:

```bash
# Switch to main branch (which has MMCV 1.7.2)
git checkout main

# Uninstall MMCV 2.x
pip uninstall mmcv mmengine

# Install MMCV 1.7.2
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

## Version Compatibility Table

| mmsegmentation | MMCV Version | MMEngine | Notes |
|----------------|--------------|----------|-------|
| 0.29.1 | 1.3.13 - 1.8.0 | Not required | Current on main branch |
| 1.0.0+ | 2.0.0+ | Required | Required for MMCV 2.x |

## Recommended Action

Since you're trying to use MMCV 2.x with PyTorch 2.1.0 and CUDA 12.2:

1. **Upgrade mmsegmentation to 1.0.0+**:
   ```bash
   pip install "mmsegmentation>=1.0.0"
   ```

2. **Verify installation**:
   ```bash
   python -c "import mmseg; print(mmseg.__version__)"
   python -c "import mmcv; print(mmcv.__version__)"
   python -c "import mmengine; print(mmengine.__version__)"
   ```

3. **Test your code** - You may need to update some API calls as mmsegmentation 1.0.0+ has some breaking changes from 0.29.1.

## Breaking Changes in mmsegmentation 1.0.0+

- API changes in some modules
- Config file format changes
- Some deprecated functions removed

Refer to the [mmsegmentation migration guide](https://github.com/open-mmlab/mmsegmentation) for details.

