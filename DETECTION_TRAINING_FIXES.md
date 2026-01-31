# Vision Mamba Detection: Training Failure Fixes

**Date:** January 2025  
**Scope:** MS-COCO detection with Cascade Mask R-CNN + Vision Mamba backbone  
**Logs analyzed:** `fir_detection_logs` (CC-Fir HPC runs)

---

## Summary of Failures by Discretization Method

| Method | Failure Type | Root Cause |
|--------|--------------|------------|
| **ZOH** | `FloatingPointError: Predicted boxes or scores contain Inf/NaN` | Training diverged early; numerical instability |
| **FOH** | `JOB CANCELLED DUE TO TIME LIMIT` | SLURM job exceeded allocated time |
| **Bilinear** | `TypeError: 'GeometryCollection' object is not iterable` | Shapely 2.0 + fvcore CropTransform incompatibility |
| **Poly** | `FloatingPointError: Predicted boxes or scores contain Inf/NaN` | Same as ZOH |
| **Highorder** | `TypeError: 'GeometryCollection' object is not iterable` | Same as Bilinear |
| **RK4** | `FloatingPointError: Predicted boxes or scores contain Inf/NaN` | Same as ZOH |

---

## Fixes Applied

### 1. GeometryCollection Fix (Bilinear, Highorder)

**Problem:** fvcore's `CropTransform.apply_polygons` uses Shapely's `polygon.intersection(crop_box)`, which can return a `GeometryCollection` when cropping produces mixed geometry types. In Shapely 2.0+, `GeometryCollection` is not directly iterable—you must use `.geoms`.

**Fix:** Monkey-patched `CropTransform.apply_polygons` in `det/detectron2/data/detection_utils.py` to handle `GeometryCollection` by using `cropped.geoms` when the intersection result is a `GeometryCollection`.

**Files modified:** `det/detectron2/data/detection_utils.py`

---

### 2. Inf/NaN Divergence Fix (ZOH, Poly, RK4)

**Problem:** Predicted RPN boxes/scores contained Inf/NaN very early in training, causing `FloatingPointError`. This indicates numerical instability in the discretization methods.

**Fix:** Added gradient clipping to all discretization configs:
- `train.clip_grad = dict(enabled=True, clip_type="norm", clip_value=1.0)`

**Files modified:**
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_zoh.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_poly.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_rk4.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_bilinear.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_highorder.py`
- `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1_foh.py`

---

### 3. FOH Time Limit (User Action Required)

**Problem:** FOH job was cancelled: `JOB 21348504 ON fc10401 CANCELLED AT 2026-01-30T22:46:54 DUE TO TIME LIMIT`

**Fix:** Increase the SLURM `--time` allocation in your job submission script for FOH detection training. Detection with 100 epochs typically needs 24–48+ hours depending on hardware. Example:

```bash
#SBATCH --time=48:00:00   # or higher for FOH (may be slower than ZOH)
```

---

## Additional Recommendations

1. **If Inf/NaN persists** after gradient clipping: Consider lowering the learning rate in the optimizer config (e.g., scale by 0.5 for ZOH/Poly/RK4).

2. **LR scheduler order warning:** Logs show `lr_scheduler.step()` before `optimizer.step()`. This is a detectron2/trainer ordering issue; consider checking `det/detectron2/engine/train_loop.py` if the first LR value is skipped.

3. **SSM hyperparameters:** Similar to segmentation, detection may benefit from tighter `dt_min`/`dt_max` bounds for non-ZOH methods. See `SEGMENTATION_DISCRETIZATION_REPORT.md` for reference values.
