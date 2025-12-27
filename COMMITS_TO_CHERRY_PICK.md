# Commits in `custom_cuda` Branch to Consider for Cherry-Picking to `mmcv-2.x`

This document lists all commits in `custom_cuda` that are not in `mmcv-2.x`, organized by category.

## Summary Statistics
- **Total commits in custom_cuda not in mmcv-2.x:** 40 commits
- **Note:** Some commits with similar messages exist in both branches (likely cherry-picked already), but with different hashes

---

## 🎯 Category 1: CUDA Implementation Improvements

### 1. **ebf2bdc** - Add custom CUDA kernels for all discretization methods
**Priority: HIGH** ⭐⭐⭐
- Add DiscretizationMethod enum and support in CUDA kernel infrastructure
- Implement CUDA kernels for ZOH, FOH, BILINEAR, POLY, HIGHORDER, and RK4
- Create discretization_kernels.cuh with compute_discretization() template
- Update selective_scan_interface.py to use CUDA kernels for all methods
- Add profiling script (profile_discretization.py) to measure performance
- **Impact:** Addresses performance discrepancy where BILINEAR was faster than ZOH

### 2. **0608edb** - Add option to select CUDA kernel or Python reference for all discretization methods
**Priority: HIGH** ⭐⭐⭐
- Add use_cuda_kernel parameter to selective_scan_fn and Mamba module
- Support VIM_USE_CUDA_KERNEL environment variable for script-level control
- Allow ZOH to use Python reference implementation (PyTorch operations)
- All methods (ZOH, FOH, BILINEAR, etc.) can now use either implementation path
- **Impact:** Enables fair comparison between CUDA and Python implementations

### 3. **4774b22** - Fix CUDA kernel template type for complex delta_u_val
**Priority: HIGH** ⭐⭐⭐
- Fixes compilation errors when instantiating template with complex weight types
- Uses conditional type for delta_u_val parameter: float for real case, complex_t for complex case
- Updated all complex case implementations to handle delta_u_val as complex value
- **Impact:** Critical fix for complex number support in CUDA kernels

---

## 🐛 Category 2: Python Reference Implementation Fixes

### 4. **644dac1** - Fix Bilinear, Poly, and Higher-Order discretization formulas
**Priority: HIGH** ⭐⭐⭐
- **BILINEAR:** Fixed stability issue by swapping matrix inversion
  - Old (unstable): Ā = (I + ΔA/2)⁻¹(I - ΔA/2)
  - New (correct): Ā = (I - ΔA/2)⁻¹(I + ΔA/2)
- **POLYNOMIAL:** Added missing ½×FOH terms to the formula
- **HIGHER-ORDER:** Implemented correct generalized formula
- Updated both Python reference and CUDA kernel implementations
- **Note:** Similar commit exists in mmcv-2.x (c23f2ae) - verify if this is more complete

### 5. **1e10a2b** - Fix FOH discretization formula using Taylor series expansion
**Priority: HIGH** ⭐⭐⭐
- Updated FOH discretization to use correct formula: B_d = A^(-2) * (exp(A*Δ) - I - A*Δ) * B
- Uses Taylor series expansion to avoid division (performance improvement)
- Eliminates element-wise division which was a performance bottleneck
- Updated both Python reference and CUDA kernel implementations
- **Note:** Similar commit exists in mmcv-2.x (c23f2ae) - verify if this is more complete

### 6. **2d50daa** - Fix bilinear discretization method Python reference implementation
**Priority: HIGH** ⭐⭐⭐
- Fixed tensor shape mismatch: convert half_delta_A from vector to diagonal matrix
- Fixed identity matrix shape: create I with correct dimensions
- Updated scan loop to use matrix-vector multiplication for bilinear method
- Fixed B computation broadcasting for both variable and non-variable B cases
- **Note:** Similar commit exists in mmcv-2.x (7866632) - verify differences

### 7. **9a16cee** - Fix RK4 discretization method for variable B case
**Priority: HIGH** ⭐⭐⭐
- Fixed tensor shape mismatches in RK4 implementation for variable B
- Corrected einsum operation from 'bdlnm,bdl->bdlnm' to proper shape handling
- Fixed RK4 coefficients computation for B.dim() == 3 case
- RK4 now produces different outputs from ZOH (was identical before due to bug)
- **Note:** Similar commits exist in mmcv-2.x (944069c, defe3af) - verify if this is more complete

### 8. **fb13860** - Implement non-causal bidirectional scan for Polynomial Interpolation
**Priority: MEDIUM** ⭐⭐
- Updated Polynomial Interpolation to use bidirectional (non-causal) scan
- Performs forward and backward passes, then averages results
- Forces Python reference path (CUDA kernel doesn't support bidirectional scan)
- Verified and documented HOH as causal method
- **Impact:** Matches theoretical behavior: Poly Interpolation = non-causal smooth

---

## 🔧 Category 3: Critical Bug Fixes

### 9. **cdda7d0** - Fix selective_scan_cuda.fwd() calls to include discretization_method enum parameter
**Priority: CRITICAL** ⭐⭐⭐
- Fixed TypeError where selective_scan_cuda.fwd() was called with 9 arguments instead of 10
- Added disc_method_enum parameter (defaulting to 0 for zoh discretization) to all calls
- Applied to: MambaInnerFnNoOutProj.forward, MambaInnerFn.forward, BiMambaInnerFn.forward
- **Note:** Similar commit exists in mmcv-2.x (d9fd21a) - verify if identical

### 10. **b12033d** - Fix deprecation warnings: update timm imports and suppress pydantic warning
**Priority: MEDIUM** ⭐⭐
- Update timm.models.registry -> timm.models (deprecated import)
- Update timm.models.layers -> timm.layers (deprecated import)
- Applied to all files: vim, seg, and det backbones
- Add warning filter for pydantic protected namespace warning
- **Note:** Similar fix exists in mmcv-2.x (f8e81f3) - verify if this is more complete

### 11. **01782ee** - Fix custom_fwd/custom_bwd wrapper: return decorator directly for old PyTorch
**Priority: MEDIUM** ⭐⭐
- Fix TypeError when device_type is provided in older PyTorch versions
- In old PyTorch, custom_fwd/custom_bwd are direct decorators, not factories
- Return the decorator function directly instead of calling it with kwargs
- Applied to all three mamba-ssm layer_norm.py files

### 12. **5541480** - Fix custom_fwd/custom_bwd device_type compatibility for older PyTorch
**Priority: MEDIUM** ⭐⭐
- Add compatibility wrappers that remove device_type argument for older PyTorch versions
- Fixes TypeError when device_type='cuda' is passed to decorators in PyTorch < 2.0
- Applied to all three mamba-ssm layer_norm.py files

### 13. **1d330d7** - Fix RMSNorm import: Add backward compatibility for custom_fwd/custom_bwd
**Priority: MEDIUM** ⭐⭐
- Add try/except to import from torch.amp (PyTorch 2.0+) or fallback to torch.cuda.amp
- Fixes ImportError when using RMSNorm with older PyTorch versions
- Applied to all three mamba-ssm layer_norm.py files

### 14. **19fb2bc** - Fix ZeroDivisionError in SmoothedValue when count is zero
**Priority: MEDIUM** ⭐⭐
- Added check in global_avg property to return 0.0 when count is 0
- Added checks in median, avg, max, and value properties to handle empty deque
- Prevents crash when metric logger tries to format before any values are added
- **Note:** Similar commit exists in mmcv-2.x (ddd5a1f) - verify if identical

### 15. **6c10a4f** - Fix custom_bwd/custom_fwd wrapper for older PyTorch versions
**Priority: MEDIUM** ⭐⭐
- Fix TypeError: custom_bwd() missing 1 required positional argument
- In PyTorch < 2.1.0, custom_bwd/custom_fwd are decorators, not decorator factories
- Wrapper now correctly returns decorators directly for older PyTorch versions

### 16. **5a3b82e** - Fix PyTorch compatibility: support custom_fwd/custom_bwd for both old and new PyTorch versions
**Priority: MEDIUM** ⭐⭐
- Add compatibility wrapper for custom_fwd/custom_bwd decorators
- Handle device_type argument for PyTorch < 2.1.0 (from torch.cuda.amp)
- Support device_type argument for PyTorch >= 2.1.0 (from torch.amp)
- Fixes ImportError: cannot import name 'custom_bwd' from 'torch.amp'

### 17. **21d5346** - resolved the import of torch.amp (Torch 2.1.0+) and fall back to torch.cuda.amp
**Priority: MEDIUM** ⭐⭐
- Resolves import compatibility between PyTorch 2.1.0+ and older versions
- **Note:** Similar fixes exist in mmcv-2.x - verify if this is redundant

### 18. **06230e5** - Fix deprecated PyTorch AMP API: update custom_fwd/custom_bwd to use torch.amp
**Priority: MEDIUM** ⭐⭐
- Update custom_fwd/custom_bwd to use torch.amp with device_type='cuda'
- **Note:** Similar fix exists in mmcv-2.x (27e8b21) - verify if identical

### 19. **76c7984** - Fix deprecation warning: replace torch.cuda.amp with torch.amp.custom_fwd/bwd
**Priority: MEDIUM** ⭐⭐
- Replace deprecated torch.cuda.amp with torch.amp.custom_fwd/bwd
- **Note:** Similar fix exists in mmcv-2.x (27e8b21) - verify if identical

---

## 📊 Category 4: Verification and Testing Scripts

### 20. **3d7b19e** - Expand verification script to test all discretization methods
**Priority: MEDIUM** ⭐⭐
- Added comprehensive testing for all 6 discretization methods
- Implemented pairwise comparison matrix to check differences between all method pairs
- Added summary statistics and detailed diagnostics
- Improved error handling for CPU vs CUDA execution
- **Note:** Similar commit exists in mmcv-2.x (310facf) - verify differences

### 21. **a429ac3** - Add diagnostic script to verify discretization method differences
**Priority: LOW** ⭐
- Created verify_discretization.py to test if ZOH and Bilinear methods produce different outputs
- Helps identify if CUDA kernel switch for discretization methods is working correctly
- Includes detailed diagnostics and recommendations

### 22. **e962cb7** - Add benchmarking analysis scripts for Vision Mamba models
**Priority: MEDIUM** ⭐⭐
- Add benchmark_latency_flops.py: comprehensive latency and FLOPs benchmarking
- Add run_benchmark_after_training.sh: wrapper script for post-training benchmarks
- Add benchmark_all_methods.sh: batch benchmarking for all discretization methods
- Add compare_benchmarks.py: results comparison and analysis tool
- **Note:** Similar commit exists in mmcv-2.x (01c76bc) - verify differences

---

## 🛠️ Category 5: Infrastructure and Configuration

### 23. **932dc6d** - Fix hardcoded mamba-ssm path and add mamba-1p1p1-fir-h100 folder
**Priority: MEDIUM** ⭐⭐
- Make mamba-ssm path configurable via MAMBA_SSM_PATH environment variable
- Add auto-detection for all mamba folder variants
- Add new mamba-1p1p1-fir-h100 folder to auto-detection list
- **Note:** Similar commit exists in mmcv-2.x (6f867f6) - verify differences

### 24. **51e7c79** - different mamba-ssm sources for build on each hpc
**Priority: LOW** ⭐
- Allows different mamba-ssm sources for different HPC systems
- **Impact:** Environment-specific configuration

### 25. **827d4e0** - Remove --rdzv-endpoint=localhost:0 for multi-node compatibility
**Priority: LOW** ⭐
- Remove --rdzv-endpoint=localhost:0 from Rorqual training scripts
- Fixes failures when using multiple remote nodes
- For multi-node setups, PyTorch will use default rendezvous mechanism

### 26. **ce979f8** - Fix: Add --rdzv-endpoint=localhost:0 to auto-select available port
**Priority: LOW** ⭐
- Add --rdzv-endpoint=localhost:0 to torch.distributed.run
- Automatically selects available port, preventing 'port 29500 already in use' errors
- **Note:** Conflicts with commit 827d4e0 - need to decide which approach to use

### 27. **da0a58e** - Fix: Remove --master_port from Rorqual training scripts
**Priority: LOW** ⭐
- Remove --master_port=0 argument from training scripts
- Fixes 'unrecognized arguments: --master_port=0' error

---

## 📝 Category 6: Script Updates and Performance Tuning

### 28. **051d173** - updated performance analysis scripts
**Priority: LOW** ⭐
- General script updates

### 29. **ca5009c, b5fcaf8, ce80441, 55bf968** - updated num_workers for better GPU parallel data loading
**Priority: LOW** ⭐
- Multiple commits updating num_workers
- **Note:** These may conflict with later commits that reduce num_workers

### 30. **8a80a5b** - updated scripts
**Priority: LOW** ⭐
- General script updates

### 31. **7cc3fe3** - Switch RK4 script to single-GPU training by default due to persistent SIGBUS issues
**Priority: MEDIUM** ⭐⭐
- Change default from distributed (2 GPUs) to single-GPU training
- Single-GPU avoids distributed barrier that triggers SIGBUS errors
- Increase batch size to 16 (from 8) since using 1 GPU instead of 2

### 32. **dbdbc62** - Add single-GPU fallback and documentation for RK4 SIGBUS issues
**Priority: MEDIUM** ⭐⭐
- Add warning about known SIGBUS issues with RK4 distributed training
- Fix incomplete distributed training command
- Add commented single-GPU fallback option

### 33. **60f007d** - Reduce RK4 script to 2 GPUs to mitigate SIGBUS during import
**Priority: MEDIUM** ⭐⭐
- Change from 4 GPUs to 2 GPUs
- Reduces memory pressure during library initialization phase

### 34. **7d64abf** - Add more aggressive SIGBUS mitigations for RK4 script
**Priority: MEDIUM** ⭐⭐
- Disable shared memory entirely (TORCH_SHARED_MEMORY_DISABLE=1)
- Reduce batch size to 8 per GPU (from 16)
- Reduce OMP_NUM_THREADS to 1 (from 2)
- Add more CUDA memory management settings

### 35. **9a43163** - Add SIGBUS mitigations for RK4 training script
**Priority: MEDIUM** ⭐⭐
- Reduce batch size from 32 to 16 per GPU
- Reduce OMP_NUM_THREADS from 4 to 2
- Add CUDA memory management with expandable_segments

### 36. **3159b18** - Set num_workers to 0 in all CVIS scripts to completely avoid shared memory issues
**Priority: MEDIUM** ⭐⭐
- Changed num_workers from 1 to 0 in all CVIS main scripts (6 scripts)
- Changed num_workers from 1 to 0 in all CVIS/cifar100 scripts (6 scripts)
- Eliminates 'No space left on device' and 'Bus error' issues

### 37. **532e40d** - Reduce num_workers to 1 in all CVIS scripts to further reduce shared memory usage
**Priority: MEDIUM** ⭐⭐
- Changed num_workers from 2 to 1 in all CVIS main scripts (6 scripts)
- Changed num_workers from 2 to 1 in all CVIS/cifar100 scripts (6 scripts)

### 38. **f0dc920** - Reduce num_workers to 2 in all training scripts to fix shared memory issues
**Priority: MEDIUM** ⭐⭐
- Changed num_workers from 4 to 2 in all CVIS, CC/Rorqual, and CC/Cedar-Fir scripts
- Changed num_workers from 25 to 2 in ft-vim-s.sh and ft-vim-t.sh
- Fixes 'No space left on device' and 'Bus error' errors

### 39. **d9b6f71** - updated scripts
**Priority: LOW** ⭐
- General script updates

### 40. **91bf524** - updated output directory to global path instead of relative
**Priority: LOW** ⭐
- Changed output directory to use global paths

---

## 🎯 Recommended Cherry-Pick Strategy

### **HIGH PRIORITY (Must Cherry-Pick):**
1. **ebf2bdc** - Add custom CUDA kernels for all discretization methods
2. **0608edb** - Add option to select CUDA kernel or Python reference
3. **4774b22** - Fix CUDA kernel template type for complex delta_u_val
4. **644dac1** - Fix Bilinear, Poly, and Higher-Order discretization formulas (verify vs c23f2ae)
5. **1e10a2b** - Fix FOH discretization formula (verify vs c23f2ae)
6. **2d50daa** - Fix bilinear discretization method (verify vs 7866632)
7. **9a16cee** - Fix RK4 discretization method (verify vs 944069c, defe3af)
8. **fb13860** - Implement non-causal bidirectional scan for Polynomial Interpolation

### **MEDIUM PRIORITY (Should Cherry-Pick):**
9. **cdda7d0** - Fix selective_scan_cuda.fwd() calls (verify vs d9fd21a)
10. **b12033d** - Fix deprecation warnings: update timm imports (verify vs f8e81f3)
11. **01782ee, 5541480, 1d330d7** - PyTorch compatibility fixes (verify vs existing fixes)
12. **19fb2bc** - Fix ZeroDivisionError (verify vs ddd5a1f)
13. **3d7b19e** - Expand verification script (verify vs 310facf)
14. **e962cb7** - Add benchmarking analysis scripts (verify vs 01c76bc)
15. **932dc6d** - Fix hardcoded mamba-ssm path (verify vs 6f867f6)
16. **7cc3fe3, dbdbc62, 60f007d, 7d64abf, 9a43163** - RK4 SIGBUS mitigations
17. **3159b18, 532e40d, f0dc920** - num_workers fixes for shared memory issues

### **LOW PRIORITY (Optional):**
- Script updates and infrastructure changes
- Environment-specific configurations

---

## ⚠️ Important Notes

1. **Verify Duplicates:** Many commits have similar messages in both branches. Before cherry-picking, verify if the changes are already in `mmcv-2.x` with different hashes.

2. **Conflicts:** Some commits may conflict with MMCV 2.x migration changes. Review carefully before cherry-picking.

3. **Order Matters:** Cherry-pick commits in chronological order (oldest first) to minimize conflicts.

4. **Test After Each:** Test after cherry-picking each commit to ensure nothing breaks.

5. **Commit Dependencies:** Some commits depend on others (e.g., CUDA kernel commits depend on each other).

---

## 📋 Next Steps

1. Review this list and decide which commits to cherry-pick
2. For commits marked "verify vs X", check if the changes are already in `mmcv-2.x`
3. Create a cherry-pick script or cherry-pick commits one by one
4. Test after each cherry-pick to ensure compatibility with MMCV 2.x changes

