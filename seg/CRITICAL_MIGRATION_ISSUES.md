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

#### 2. samples_per_gpu in Configs (MEDIUM PRIORITY) ✅ FIXED

**Location:** Multiple config files

**Issue:** Configs were using legacy `samples_per_gpu` instead of explicit dataloader configs

**Fix Applied:**

- Added explicit `train_dataloader` configs to all config files
- Updated `train_api.py` to support both MMEngine format (explicit dataloaders) and legacy format
- Maintained backward compatibility with `samples_per_gpu` for legacy code

**Files Updated:**

- `seg/configs/_base_/datasets/ade20k.py` - Added train_dataloader/val_dataloader/test_dataloader
- All `upernet_vim_*.py` configs - Added train_dataloader configs
- `seg/mmcv_custom/train_api.py` - Updated to handle both formats

**New Format:**

```python
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True)
)

# Backward compatibility: keep old format
data=dict(samples_per_gpu=8, workers_per_gpu=16)
```

**Status:** ✅ FIXED - Now uses explicit dataloader configs with backward compatibility

---

#### 3. fp16 Handling in test.py (LOW PRIORITY) ✅ FIXED

**Location:** `seg/test.py:229-245`

**Issue:** Was using deprecated `cfg.get('fp16', None)` and `wrap_fp16_model()`

**Fix Applied:**

- Updated to check `optim_wrapper` config first (preferred method)
- Falls back to deprecated `fp16` config for backward compatibility
- Added warning when using deprecated `fp16` config
- For inference, `wrap_fp16_model()` is still acceptable

**New Code:**

```python
# Check optim_wrapper config first (preferred)
use_fp16 = False
if hasattr(cfg, 'optim_wrapper') and cfg.optim_wrapper is not None:
    if isinstance(cfg.optim_wrapper, dict):
        use_fp16 = cfg.optim_wrapper.get('type') == 'AmpOptimWrapper'

# Fallback to deprecated fp16 config (with warning)
if not use_fp16:
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        warnings.warn('Using deprecated `fp16` config...')
        use_fp16 = True

if use_fp16:
    wrap_fp16_model(model)
```

**Status:** ✅ FIXED - Now prefers optim_wrapper config, with backward compatibility

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
