# Critical Migration Issues Found

## Summary of High-Risk Gaps

### ✅ FIXED Issues

1. **train.py workflow reference** - FIXED
   - Removed `cfg.workflow` check, now uses `validate` flag directly

### ⚠️ CRITICAL Issues Remaining

#### 1. test.py Legacy APIs (HIGH PRIORITY) ✅ FIXED

**Location:** `seg/test.py:23` (was)

**Issue:** Was importing deprecated `mmseg.apis.multi_gpu_test` and `single_gpu_test`

**Impact:** These functions are removed in MMSegmentation 1.x. Should use `Runner.test()` instead.

**Fix Applied:** Refactored test.py to use MMEngine's Runner API:

- Removed `from mmseg.apis import multi_gpu_test, single_gpu_test`
- Replaced with `from mmengine.runner import Runner`
- Replaced `multi_gpu_test()` and `single_gpu_test()` calls with `runner.test()`
- Updated to use `MMDataParallel` and `MMDistributedDataParallel` from mmengine
- Configured `SegEvaluator` for evaluation
- Maintained backward compatibility for show/format/out options (with warnings)

**Status:** ✅ FIXED - Now uses MMEngine Runner API

---

#### 2. samples_per_gpu in Configs (MEDIUM PRIORITY)

**Location:** Multiple config files

**Issue:** Configs still use legacy `samples_per_gpu` instead of explicit dataloader configs

**Files Affected:**

- `seg/configs/_base_/datasets/ade20k.py:53`
- `seg/configs/_base_/datasets/ade20k_640x640.py:36`
- `seg/configs/_base_/datasets/ade20k_1024x1024.py:36`
- `seg/configs/_base_/datasets/ade20k_ms_eval.py:38`
- All `upernet_vim_*.py` configs

**Current State:**

```python
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(...),
    val=dict(...)
)
```

**MMEngine Format (Recommended):**

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

**Impact:**

- Current implementation works because `train_api.py` manually builds dataloaders from `samples_per_gpu`
- However, this is not the "proper" MMEngine way
- May cause issues with distributed training if not handled correctly

**Status:** ⚠️ PARTIALLY WORKING - Backward compatibility maintained, but not ideal

---

#### 3. fp16 Handling in test.py (LOW PRIORITY)

**Location:** `seg/test.py:228-230`

**Issue:** Uses `cfg.get('fp16', None)` and `wrap_fp16_model()`

**Current Code:**

```python
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
```

**Impact:**

- `fp16` config key is deprecated
- Should use `AmpOptimWrapper` instead
- However, for inference, `wrap_fp16_model()` might still be acceptable

**Status:** ⚠️ ACCEPTABLE - Works but uses deprecated API

---

#### 4. Custom Hooks Registration (VERIFIED ✅)

**Location:** `seg/mmcv_custom/throughput_hook.py:7`

**Status:** ✅ CORRECTLY REGISTERED

```python
@HOOKS.register_module()
class ThroughputHook(Hook):
    ...
```

**Usage:** Hook is registered and used in `train_api.py:139`

---

## Recommendations

### Immediate Actions Required:

1. **Fix test.py** (HIGH PRIORITY)

   - Refactor to use `Runner.test()` instead of `multi_gpu_test`/`single_gpu_test`
   - This is a breaking change that will cause errors in MMSegmentation 1.x+

2. **Consider Converting Dataloaders** (MEDIUM PRIORITY)

   - Convert all configs to use explicit `train_dataloader`/`val_dataloader`/`test_dataloader`
   - Update `train_api.py` to use config dataloaders directly instead of building manually
   - This ensures proper MMEngine compatibility

3. **Update fp16 Handling** (LOW PRIORITY)
   - Consider removing `fp16` config support in test.py
   - Use model's optim_wrapper configuration instead

---

## Verification Commands

Run these to verify fixes:

```bash
# Check for legacy imports
grep -R "mmseg.apis" -n seg/
grep -R "IterBasedRunner" -n seg/
grep -R "mmcv.runner" -n seg/

# Check for legacy config keys
grep -R "workflow" -n seg/configs/
grep -R "samples_per_gpu" -n seg/configs/
grep -R "fp16" -n seg/configs/

# Check for loss_scale (should be zero)
grep -R "loss_scale" -n seg/
```
