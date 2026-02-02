# Vision Mamba Segmentation: Discretization Method Training Changes Report

**Date:** January 2025  
**Scope:** ADE20K semantic segmentation with UPerNet + Vision Mamba backbone  
**Branch:** mmcv-2.x

---

## Executive Summary

This report documents changes made to stabilize Vision Mamba segmentation training across six discretization methods (ZOH, Bilinear, FOH, Poly, Highorder, RK4). Initial training runs revealed severe instability for non-ZOH methods: catastrophic collapse (Bilinear), stagnation (FOH, Poly, Highorder), and limited convergence (RK4). A combination of SSM hyperparameter tuning, optimizer settings, and architectural choices was applied to address these issues.

---

## 1. Observed Issues (Pre-Change Baseline)

| Discretization | Observed Behavior | mIoU @ 60K | Issue Severity |
|----------------|------------------|------------|----------------|
| **ZOH** | Stable improvement | 24.02% | None (reference) |
| **Bilinear** | Catastrophic collapse | 14.42% → 1.41% | Critical |
| **FOH** | Stagnation, degradation | — | High |
| **Poly** | Stagnation, degradation | — | High |
| **Highorder** | Stagnation, degradation | — | High |
| **RK4** | Most stable non-ZOH | 9.82% | Moderate |

**Root causes identified:**
- **Dynamic range of `dt`**: Non-ZOH methods are more sensitive to the step size (`dt`) used in state-space discretization. Large or poorly bounded `dt` values cause numerical instability and gradient explosion.
- **Stiffness of state transitions**: Higher-order methods (FOH, Poly, Highorder) introduce stiffer dynamics; RK4 is more robust but still sensitive.
- **Optimizer mismatch**: High weight decay (0.05–0.1) over-regularizes SSM parameters (`dt_proj`, A, B, C), shrinking them into unstable regimes.
- **Learning rate**: 1e-4 was too aggressive for sensitive discretization paths; gradients can scale differently than in ZOH.
- **Architectural mismatch**: Scripts overrode `if_bimamba=False`, diverging from the paper’s bidirectional setup.

---

## 2. Changes Implemented

### 2.1 SSM Hyperparameters (`ssm_cfg`)

Tighter bounds on `dt_min`, `dt_max`, and `dt_scale` constrain the learned step size and reduce instability.

| Method | dt_min | dt_max | dt_scale | Rationale |
|--------|--------|--------|----------|-----------|
| **ZOH** | 0.001 (default) | 0.1 (default) | 1.0 (default) | No change; already stable |
| **Bilinear** | 0.0005 | 0.03 | 0.3 | Strong reduction to avoid collapse; Tustin can blow up with large dt |
| **FOH** | 0.0005 | 0.02 | 0.25 | Stricter than Bilinear; FOH is more sensitive |
| **Poly** | 0.0005 | 0.02 | 0.3 | Similar to FOH; polynomial interpolation sensitive to dt scale |
| **Highorder** | 0.0003 | 0.015 | 0.2 | Most restrictive; higher-order methods most prone to instability |
| **RK4** | 0.0005 | 0.05 | 0.5 | Moderate reduction; RK4 is inherently more stable |

**Config location:** `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_<method>.py`  
**Implementation:** `ssm_cfg=dict(dt_min=..., dt_max=..., dt_scale=...)` in `model.backbone`

---

### 2.2 Bidirectional Mamba (`if_bimamba=True`)

**Change:** Set `model.backbone.if_bimamba=True` in all segmentation scripts.

**Reason:** The Vision Mamba paper uses bidirectional Mamba for dense prediction. Unidirectional Mamba (`if_bimamba=False`) limits context and hurts segmentation performance.

**Impact per method:**
- **All methods:** Improved spatial context and segmentation quality.
- **Non-ZOH:** Bidirectional processing can help stabilize gradients by providing more balanced signal flow.

**Files modified:** All `ft_vim_tiny_upernet_*.sh` in CC-Fir, CC-Rorqual, CVIS; `ft_vim_tiny_upernet.sh`, `ft_vim_small_upernet.sh`, `eval_vim_small_upernet.sh`

---

### 2.3 Training Duration (200K Iterations, No Early Stopping)

**Change:** Explicitly set `train_cfg.max_iters=200000` in all scripts; confirmed no early stopping.

**Reason:** The paper trains for 200K iterations. Shorter runs or early stopping prevent full convergence, especially for slower-converging non-ZOH methods.

**Impact per method:**
- **ZOH:** Ensures full 200K training as in the paper.
- **Non-ZOH:** Allows more time to overcome initial instability and reach better mIoU.

**Base config:** `seg/configs/_base_/schedules/schedule_200k.py` already defines `max_iters=200000`; scripts reinforce this via `--options`.

---

### 2.4 Weight Decay (0.01)

**Change:** Set `optimizer.weight_decay=0.01` in all configs and scripts (was 0.05 or 0.1).

**Reason:** High weight decay over-regularizes SSM parameters. `dt_proj`, A, B, and C need to stay in a usable range; strong L2 penalty pushes them toward zero and causes unstable dynamics.

**Impact per method:**

| Method | Previous WD | New WD | Effect |
|--------|-------------|--------|--------|
| All | 0.05 or 0.1 | 0.01 | Gentler regularization; SSM parameters can evolve without over-shrinking |

**Stability mechanism:** Lower weight decay reduces conflict with gradient updates and keeps `dt` and state matrices in a numerically stable regime.

---

### 2.5 Learning Rate (1e-5)

**Change:** Scripts use `optimizer.lr=1e-5` (10× lower than typical 1e-4).

**Reason:** Non-ZOH discretization paths can produce larger or noisier gradients. A lower learning rate reduces overshooting and gradient-scale mismatch.

**Impact per method:**
- **ZOH:** Slightly slower but stable convergence.
- **Non-ZOH:** Critical for avoiding collapse; allows small, controlled updates to sensitive parameters.

---

## 3. Summary by Discretization Method

### ZOH (Zero Order Hold)
- **Baseline:** Stable, 24.02% mIoU at 60K.
- **Changes:** `if_bimamba=True`, `weight_decay=0.01`, `max_iters=200000`, `lr=1e-5`.
- **SSM:** Default `dt_min=0.001`, `dt_max=0.1`, `dt_scale=1.0` (unchanged).
- **Expected:** Maintained stability; full 200K training; alignment with paper.

### Bilinear (Tustin)
- **Baseline:** Catastrophic collapse (14.42% → 1.41%).
- **Changes:** `ssm_cfg(dt_min=0.0005, dt_max=0.03, dt_scale=0.3)`, `if_bimamba=True`, `weight_decay=0.01`, `max_iters=200000`, `lr=1e-5`.
- **Expected:** Avoid collapse by constraining `dt`; slower but stable training.

### FOH (First Order Hold)
- **Baseline:** Stagnation and degradation.
- **Changes:** `ssm_cfg(dt_min=0.0005, dt_max=0.02, dt_scale=0.25)`, `if_bimamba=True`, `weight_decay=0.01`, `max_iters=200000`, `lr=1e-5`.
- **Expected:** More stable dynamics; potential for meaningful mIoU improvement.

### Poly (Polynomial Interpolation)
- **Baseline:** Stagnation and degradation.
- **Changes:** `ssm_cfg(dt_min=0.0005, dt_max=0.02, dt_scale=0.3)`, `if_bimamba=True`, `weight_decay=0.01`, `max_iters=200000`, `lr=1e-5`.
- **Expected:** Reduced sensitivity to `dt`; improved convergence.

### Highorder (Higher-Order Hold)
- **Baseline:** Stagnation and degradation.
- **Changes:** `ssm_cfg(dt_min=0.0003, dt_max=0.015, dt_scale=0.2)`, `if_bimamba=True`, `weight_decay=0.01`, `max_iters=200000`, `lr=1e-5`.
- **Expected:** Strongest constraints; most conservative tuning for the most sensitive method.

### RK4 (Runge-Kutta 4th Order)
- **Baseline:** Most stable non-ZOH; 9.82% mIoU.
- **Changes:** `ssm_cfg(dt_min=0.0005, dt_max=0.05, dt_scale=0.5)`, `if_bimamba=True`, `weight_decay=0.01`, `max_iters=200000`, `lr=1e-5`.
- **Expected:** Better mIoU with bidirectional context and full 200K training; RK4 remains the most robust non-ZOH option.

---

## 4. Files Modified

### Config Files (8)
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k.py`
- `seg/configs/vim/upernet/upernet_vim_small_24_512_slide_200k.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_zoh.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_bilinear.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_foh.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_poly.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_highorder.py`
- `seg/configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k_rk4.py`

### Shell Scripts (26)
- **CC-Fir:** `ft_vim_tiny_upernet_{zoh,bilinear,foh,poly,highorder,rk4}.sh`, `singlegputest_ft_vim_tiny_upernet_zoh.sh`
- **CC-Rorqual:** `ft_vim_tiny_upernet_{zoh,bilinear,foh,poly,highorder,rk4}.sh`
- **CVIS:** `ft_vim_tiny_upernet_{zoh,foh}.sh`
- **Main:** `ft_vim_tiny_upernet.sh`, `ft_vim_small_upernet.sh`, `eval_vim_small_upernet.sh`

---

## 5. Verification Checklist

- [ ] ZOH: Training completes 200K iters; mIoU ≥ baseline (~24% at 60K, higher at 200K)
- [ ] Bilinear: No collapse; mIoU improves or stays stable over 200K
- [ ] FOH, Poly, Highorder: No stagnation; mIoU improves over training
- [ ] RK4: mIoU improves over 9.82% with full 200K training
- [ ] All: `if_bimamba=True` in both train and eval scripts

---

## 6. Post-failure recommendations (confirmed)

These address NCCL timeouts and HPC job failures observed in Rorqual segmentation runs (see failure assessment). Solutions **#1** (NCCL timeout env) and **#2** (resume from checkpoint) are left to the user; **#3–#5** are checked and confirmed below.

### #3 — Reduce validation frequency ✅ Implemented

**Recommendation:** Use validation every 2000 iterations instead of 1000 to lower the chance of one rank stalling during slide inference and triggering NCCL timeouts.

**Confirmed:** In `seg/configs/_base_/schedules/schedule_200k.py`:
- `train_cfg.val_interval` set to **2000** (was 1000).
- `evaluation.interval` set to **2000** (was 1000).

All 200K segmentation runs that use this schedule now validate half as often; checkpoint interval remains 1000.

### #4 — Stability changes from this report ✅ Already in place

**Recommendation:** Use `ssm_cfg`, `weight_decay=0.01`, `if_bimamba=True`, 200K iters, and lr=1e-5 as in this report.

**Confirmed:** Grep over `seg/` shows:
- **ssm_cfg:** Set in configs for bilinear, foh, poly, highorder, rk4 (ZOH uses defaults).
- **weight_decay=0.01:** In all 8 segmentation configs and all segmentation shell scripts.
- **if_bimamba=True:** In all segmentation train/eval scripts (CC-Fir, CC-Rorqual, CVIS, main scripts).
- **max_iters=200000:** In schedule_200k and script `--options`.
- **lr=1e-5:** In configs and scripts.

No further code changes needed for #4.

### #5 — SLURM / wall time and resume ✅ Documented

**Recommendation:** Ensure job wall time is long enough for 200K iterations; use checkpoint resume if jobs are preempted or hit limits.

**Confirmed:** This repo does not contain SLURM submit scripts (no `#SBATCH` in seg/). Submission is external (e.g. Rorqual). Guidance:
- **Wall time:** 200K iters at ~1.2 s/iter is ~67 hours of training; request at least **72 hours** (e.g. `#SBATCH --time=72:00:00`) so a single job can finish, or use shorter jobs and resume.
- **Resume:** All segmentation scripts already look for `latest.pth` (or latest `.pth`) in the work dir and pass `--resume-from` when found; no code change needed.
- **Reference:** Detection side uses `#SBATCH --time=48:00:00` in `DETECTION_TRAINING_FIXES.md`; segmentation 200K runs should use longer time or explicit resume strategy.

---

## 7. References

- Vision Mamba paper (segmentation setup: bidirectional Mamba, 200K iters, weight_decay=0.01)
- Mamba SSM: `mamba-1p1p1/mamba_ssm/modules/mamba_simple.py` (dt_min, dt_max, dt_scale)
- Baseline logs: `rorqual_segmentation_logs` (pre-change behavior)
