# MMSegmentation & MMDetection Upgrade Analysis

## Summary

**MMSegmentation:** Upgrading from 0.29.1 to >=1.0.0 **WILL BREAK** some code. Several APIs have changed.

**MMDetection:** Yes, you need to upgrade. MMDetection < 3.0.0 is incompatible with MMCV 2.x. You need MMDetection >= 3.0.0.

## MMSegmentation 0.29.1 → 1.0.0+ Breaking Changes

### 1. **Registry System Changes** ⚠️ BREAKING

**Current Code:**
```python
from mmseg.models.builder import BACKBONES
@BACKBONES.register_module()
class VisionMambaSeg(VisionMamba):
    ...
```

**Issue:** In mmsegmentation 1.0.0+, the registry system changed. `BACKBONES` might be moved to `mmseg.registry`.

**Fix Required:**
```python
# Try this first (new location)
from mmseg.registry import MODELS as BACKBONES
# Or
from mmengine.registry import MODELS as BACKBONES
```

**File Affected:** `seg/backbone/vim.py:12`

---

### 2. **Eval Hooks Changes** ⚠️ BREAKING

**Current Code:**
```python
from mmseg.core import DistEvalHook, EvalHook
```

**Issue:** In mmsegmentation 1.0.0+, eval hooks moved to `mmseg.engine`.

**Fix Required:**
```python
from mmseg.engine import DistEvalHook, EvalHook
```

**File Affected:** `seg/mmcv_custom/train_api.py:10`

**Note:** Line 144 already has `from mmseg.engine import SegEvaluator` which suggests this was anticipated, but the import on line 10 is still wrong.

---

### 3. **Pipeline Registry** ⚠️ POTENTIAL BREAKING

**Current Code:**
```python
from mmseg.datasets.builder import PIPELINES
@PIPELINES.register_module()
class SETR_Resize(object):
    ...
```

**Issue:** Pipeline registry might have moved to `mmseg.registry`.

**Fix Required:**
```python
from mmseg.registry import TRANSFORMS as PIPELINES
# Or check if it's still in datasets.builder
```

**File Affected:** `seg/mmcv_custom/resize_transform.py:5`

---

### 4. **Utils Functions** ⚠️ POTENTIAL BREAKING

**Current Code:**
```python
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
from mmseg.utils import collect_env, get_root_logger
```

**Issue:** Some utility functions might have moved to `mmengine` or changed locations.

**Potential Changes:**
- `build_ddp`, `build_dp` - might be in `mmengine.model` or `mmseg.utils`
- `get_device` - might be in `mmengine.device`
- `setup_multi_processes` - might be in `mmengine.utils`
- `collect_env` - might be in `mmengine.utils`
- `get_root_logger` - might be in `mmengine.logging`

**Files Affected:**
- `seg/test.py:21`
- `seg/train.py:20`
- `seg/backbone/vim.py:11`

---

### 5. **APIs Module** ⚠️ POTENTIAL BREAKING

**Current Code:**
```python
from mmseg.apis import multi_gpu_test, single_gpu_test, set_random_seed
```

**Issue:** These functions might have moved or changed signatures.

**Potential Changes:**
- `multi_gpu_test`, `single_gpu_test` - might be in `mmseg.apis` or moved
- `set_random_seed` - might be in `mmengine.utils` or `mmseg.utils`

**Files Affected:**
- `seg/test.py:18`
- `seg/train.py:16`

---

### 6. **Digit Version** ⚠️ POTENTIAL BREAKING

**Current Code:**
```python
from mmseg import digit_version
```

**Issue:** `digit_version` might have moved to `mmengine.utils`.

**Fix Required:**
```python
from mmengine.utils import digit_version
```

**File Affected:** `seg/test.py:17`

---

## MMDetection Compatibility

### Current Status
- **No explicit mmdetection version found in det-requirements.txt**
- **MMDetection is used in:** `det/detectron2/modeling/mmdet_wrapper.py`

### Required Upgrade
- **MMDetection < 3.0.0** is **INCOMPATIBLE** with MMCV 2.x
- **Must upgrade to MMDetection >= 3.0.0**

### Potential Breaking Changes in MMDetection 3.0.0+

**Current Code:**
```python
from mmdet.models import build_backbone, build_neck, build_detector
from mmdet.core import PolygonMasks as mm_PolygonMasks, BitmapMasks as mm_BitMasks
```

**Potential Issues:**
1. **Registry System:** Similar to mmsegmentation, registries might have moved
2. **Core Module:** `mmdet.core` might have been reorganized
3. **Model Building:** `build_*` functions might have moved

**Files Affected:**
- `det/detectron2/modeling/mmdet_wrapper.py:66, 72, 144, 190`

---

## Recommended Action Plan

### Step 1: Update Requirements

**For seg/seg-requirements.txt (already done):**
```
mmsegmentation>=1.0.0
```

**For det/det-requirements.txt (needs update):**
```
mmdetection>=3.0.0
```

### Step 2: Fix Breaking Changes

#### Priority 1: Critical (Will definitely break)
1. ✅ Fix `from mmseg.core import DistEvalHook, EvalHook` → `from mmseg.engine import ...`
2. ✅ Fix `from mmseg.models.builder import BACKBONES` → `from mmseg.registry import MODELS as BACKBONES`
3. ⚠️ Check `from mmseg.datasets.builder import PIPELINES` → might need `from mmseg.registry import TRANSFORMS as PIPELINES`

#### Priority 2: High (Likely to break)
4. ⚠️ Update utility imports (`build_ddp`, `build_dp`, `get_device`, etc.)
5. ⚠️ Update `digit_version` import
6. ⚠️ Check API functions (`multi_gpu_test`, `single_gpu_test`, `set_random_seed`)

#### Priority 3: Medium (May need testing)
7. ⚠️ Test all mmsegmentation API calls
8. ⚠️ Update mmdetection imports and test

### Step 3: Testing Checklist

- [ ] Test segmentation training: `python seg/train.py configs/...`
- [ ] Test segmentation inference: `python seg/test.py configs/...`
- [ ] Test detection wrapper: `det/detectron2/modeling/mmdet_wrapper.py`
- [ ] Test custom backbones: `seg/backbone/vim.py`
- [ ] Test custom transforms: `seg/mmcv_custom/resize_transform.py`
- [ ] Test custom hooks: `seg/mmcv_custom/train_api.py`

---

## Compatibility Matrix

| Component | Current | Required for MMCV 2.x | Status |
|-----------|---------|----------------------|--------|
| MMCV | 1.7.2 | 2.0.0+ | ✅ Updated |
| MMEngine | - | 0.10.0+ | ✅ Added |
| MMSegmentation | 0.29.1 | 1.0.0+ | ⚠️ Needs fixes |
| MMDetection | ? | 3.0.0+ | ⚠️ Needs upgrade |

---

## References

- [MMSegmentation 1.0.0 Release Notes](https://github.com/open-mmlab/mmsegmentation/releases/tag/v1.0.0)
- [MMDetection 3.0.0 Release Notes](https://github.com/open-mmlab/mmdetection/releases/tag/v3.0.0)
- [MMEngine Documentation](https://mmengine.readthedocs.io/)

