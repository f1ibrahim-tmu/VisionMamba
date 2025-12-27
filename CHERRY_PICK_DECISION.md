# Cherry-Pick Decision Guide: custom_cuda → mmcv-2.x

## Quick Summary
- **Total commits to review:** 40 commits
- **Already in mmcv-2.x (similar):** ~8 commits (need verification)
- **Unique improvements:** ~32 commits

---

## ✅ HIGH PRIORITY - Must Cherry-Pick (CUDA & Core Fixes)

### 1. **ebf2bdc** - Add custom CUDA kernels for all discretization methods
**Status:** ✅ UNIQUE - Not in mmcv-2.x
- **Why:** Core CUDA implementation improvement
- **Impact:** Performance optimization for all discretization methods
- **Action:** Cherry-pick

### 2. **0608edb** - Add option to select CUDA kernel or Python reference
**Status:** ✅ UNIQUE - Not in mmcv-2.x
- **Why:** Enables flexible implementation selection
- **Impact:** Allows fair comparison and debugging
- **Action:** Cherry-pick

### 3. **4774b22** - Fix CUDA kernel template type for complex delta_u_val
**Status:** ✅ UNIQUE - Not in mmcv-2.x
- **Why:** Critical fix for complex number support
- **Impact:** Fixes compilation errors
- **Action:** Cherry-pick

### 4. **644dac1** - Fix Bilinear, Poly, and Higher-Order discretization formulas
**Status:** ⚠️ VERIFY - Similar to c23f2ae in mmcv-2.x
- **Why:** Important formula corrections
- **Action:** Compare with c23f2ae, cherry-pick if more complete

### 5. **1e10a2b** - Fix FOH discretization formula using Taylor series expansion
**Status:** ⚠️ VERIFY - Similar to c23f2ae in mmcv-2.x
- **Why:** Performance improvement (avoids division)
- **Action:** Compare with c23f2ae, cherry-pick if more complete

### 6. **2d50daa** - Fix bilinear discretization method Python reference implementation
**Status:** ⚠️ VERIFY - Similar to 7866632 in mmcv-2.x
- **Why:** Tensor shape fixes
- **Action:** Compare with 7866632, cherry-pick if different/better

### 7. **9a16cee** - Fix RK4 discretization method for variable B case
**Status:** ⚠️ VERIFY - Similar to 944069c, defe3af in mmcv-2.x
- **Why:** Critical RK4 bug fix
- **Action:** Compare with existing fixes, cherry-pick if more complete

### 8. **fb13860** - Implement non-causal bidirectional scan for Polynomial Interpolation
**Status:** ✅ UNIQUE - Not in mmcv-2.x
- **Why:** Corrects theoretical behavior of Poly Interpolation
- **Impact:** Makes Poly non-causal (smooth), HOH causal
- **Action:** Cherry-pick

---

## 🔧 MEDIUM PRIORITY - Bug Fixes & Compatibility

### 9. **cdda7d0** - Fix selective_scan_cuda.fwd() calls to include discretization_method enum
**Status:** ⚠️ VERIFY - Similar to d9fd21a in mmcv-2.x
- **Why:** Critical TypeError fix
- **Action:** Compare with d9fd21a, likely already fixed

### 10. **b12033d** - Fix deprecation warnings: update timm imports
**Status:** ⚠️ VERIFY - Similar to f8e81f3 in mmcv-2.x
- **Why:** Cleanup deprecation warnings
- **Action:** Compare with f8e81f3, cherry-pick if more complete

### 11-13. **01782ee, 5541480, 1d330d7** - PyTorch compatibility fixes for custom_fwd/custom_bwd
**Status:** ⚠️ VERIFY - Similar fixes may exist in mmcv-2.x
- **Why:** Backward compatibility with older PyTorch
- **Action:** Check if mmcv-2.x has these fixes, cherry-pick if missing

### 14. **19fb2bc** - Fix ZeroDivisionError in SmoothedValue
**Status:** ⚠️ VERIFY - Similar to ddd5a1f in mmcv-2.x
- **Why:** Prevents crashes in metric logging
- **Action:** Compare with ddd5a1f, likely already fixed

### 15-17. **6c10a4f, 5a3b82e, 21d5346** - Additional PyTorch compatibility fixes
**Status:** ⚠️ VERIFY - May overlap with existing fixes
- **Why:** Additional compatibility layers
- **Action:** Review and cherry-pick if not redundant

### 18-19. **06230e5, 76c7984** - Fix deprecated PyTorch AMP API
**Status:** ⚠️ VERIFY - Similar to 27e8b21 in mmcv-2.x
- **Why:** Update to new torch.amp API
- **Action:** Compare with 27e8b21, likely already fixed

---

## 📊 MEDIUM PRIORITY - Testing & Verification

### 20. **3d7b19e** - Expand verification script to test all discretization methods
**Status:** ⚠️ VERIFY - Similar to 310facf in mmcv-2.x
- **Why:** Comprehensive testing
- **Action:** Compare with 310facf, cherry-pick if more complete

### 21. **a429ac3** - Add diagnostic script to verify discretization method differences
**Status:** ✅ UNIQUE - Not in mmcv-2.x
- **Why:** Useful diagnostic tool
- **Action:** Cherry-pick

### 22. **e962cb7** - Add benchmarking analysis scripts
**Status:** ⚠️ VERIFY - Similar to 01c76bc in mmcv-2.x
- **Why:** Performance analysis tools
- **Action:** Compare with 01c76bc, cherry-pick if different

---

## 🛠️ LOW-MEDIUM PRIORITY - Infrastructure

### 23. **932dc6d** - Fix hardcoded mamba-ssm path
**Status:** ⚠️ VERIFY - Similar to 6f867f6 in mmcv-2.x
- **Why:** Environment variable support
- **Note:** mmcv-2.x version (6f867f6) also removes old folders - more complete
- **Action:** Skip (mmcv-2.x version is better)

### 24. **51e7c79** - different mamba-ssm sources for build on each hpc
**Status:** ✅ UNIQUE - Not in mmcv-2.x
- **Why:** HPC-specific configuration
- **Action:** Cherry-pick if needed for your HPC setup

### 25-27. **827d4e0, ce979f8, da0a58e** - Distributed training fixes
**Status:** ⚠️ CONFLICT - 827d4e0 and ce979f8 conflict
- **Why:** Port selection for distributed training
- **Note:** 827d4e0 removes --rdzv-endpoint, ce979f8 adds it
- **Action:** Decide which approach you prefer, cherry-pick accordingly

---

## 🔄 LOW PRIORITY - Script Updates & Performance Tuning

### 28-40. Script updates, num_workers changes, SIGBUS mitigations
**Status:** ⚠️ REVIEW - May conflict with mmcv-2.x changes
- **Why:** Various script improvements and workarounds
- **Action:** Review individually, cherry-pick if still relevant
- **Note:** SIGBUS mitigations (31-35) may be important if you encounter those issues

---

## 🎯 Recommended Cherry-Pick Order

### Phase 1: Core CUDA Improvements (Do First)
```bash
git cherry-pick ebf2bdc    # Add custom CUDA kernels
git cherry-pick 0608edb    # Add CUDA/Python selection option
git cherry-pick 4774b22    # Fix complex type template
```

### Phase 2: Formula Fixes (Verify First)
```bash
# Compare these with existing commits in mmcv-2.x first:
# git show 644dac1 vs git show c23f2ae
# git show 1e10a2b vs git show c23f2ae
# git show 2d50daa vs git show 7866632
# git show 9a16cee vs git show 944069c defe3af

git cherry-pick fb13860    # Non-causal bidirectional scan (unique)
```

### Phase 3: Bug Fixes (Verify First)
```bash
# Compare with existing fixes:
# git show cdda7d0 vs git show d9fd21a
# git show b12033d vs git show f8e81f3
# git show 19fb2bc vs git show ddd5a1f

# If unique or better, cherry-pick:
git cherry-pick 01782ee 5541480 1d330d7  # PyTorch compatibility
```

### Phase 4: Testing & Diagnostics
```bash
git cherry-pick a429ac3    # Diagnostic script (unique)
# Compare: git show 3d7b19e vs git show 310facf
# Compare: git show e962cb7 vs git show 01c76bc
```

### Phase 5: Infrastructure (If Needed)
```bash
git cherry-pick 51e7c79    # HPC-specific config (if needed)
# Review distributed training fixes (25-27) based on your needs
```

### Phase 6: Script Updates (Review Individually)
```bash
# Review and cherry-pick SIGBUS mitigations if needed:
git cherry-pick 7cc3fe3 dbdbc62 60f007d 7d64abf 9a43163
# Review num_workers changes if still relevant
```

---

## ⚠️ Important Notes

1. **Test After Each Phase:** Don't cherry-pick all at once. Test after each phase.

2. **Resolve Conflicts:** Some commits may conflict with MMCV 2.x migration changes. Resolve carefully.

3. **Verify Duplicates:** Use `git show <commit>` to compare commits before cherry-picking.

4. **Chronological Order:** Cherry-pick in chronological order (oldest first) to minimize conflicts.

5. **Backup First:** Create a backup branch before starting:
   ```bash
   git checkout mmcv-2.x
   git checkout -b mmcv-2.x-with-custom-cuda-improvements
   ```

---

## 🔍 Quick Verification Commands

```bash
# Compare two commits
git diff 932dc6d 6f867f6 --stat

# Show what files changed in a commit
git show --stat <commit>

# Check if commit is already in mmcv-2.x (by message)
git log mmcv-2.x --grep="<commit message>" --oneline

# See commit details
git show <commit>
```

