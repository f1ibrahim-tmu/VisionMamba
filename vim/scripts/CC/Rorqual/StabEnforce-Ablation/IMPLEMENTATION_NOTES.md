# Implementation Notes: Training Pipeline Integration

## What Was Implemented Earlier

In the initial Feature-StabEnforce implementation, we focused on the **core Mamba layer functionality**:

1. ✅ Created `stability_enforcement.py` module with core functions:
   - `compute_spectral_radius()`
   - `apply_spectral_normalization()`
   - `apply_eigenvalue_clamping()`
   - `compute_stability_penalty()`
   - `apply_stability_enforcement()`

2. ✅ Modified `mamba_simple.py` to:
   - Add stability enforcement parameters to `__init__`
   - Apply stabilizers in `_construct_A_matrix()`
   - Add `compute_stability_loss()` method

3. ✅ Created documentation explaining the concepts

## What Was NOT Implemented Earlier

The **training pipeline integration** was not implemented because:

1. **Separation of Concerns**: We focused on the core algorithm implementation first, ensuring the Mamba layer could support stability enforcement before integrating it into the training pipeline.

2. **Testing Strategy**: The layer-level implementation needed to be tested independently before adding training loop complexity.

3. **Incremental Development**: It's a best practice to implement and test components incrementally rather than all at once.

## What Was Implemented Now

### 1. Command-Line Arguments (`vim/main.py`)

Added 5 new arguments to the argument parser:
- `--use-spectral-normalization`: Enable spectral normalization
- `--use-eigenvalue-clamping`: Enable eigenvalue clamping
- `--use-stability-penalty`: Enable stability penalty loss
- `--stability-epsilon`: Threshold for stability (default: 0.01)
- `--stability-penalty-weight`: Weight for penalty in loss (default: 0.1)

### 2. Model Creation Integration (`vim/main.py`)

Modified model creation to:
- Build `ssm_cfg` dictionary with stability parameters
- Pass `ssm_cfg` to `create_model()` which forwards it to `VisionMamba`
- `VisionMamba` passes it to `create_block()` which passes it to `Mamba` via `**ssm_cfg`

### 3. Training Loop Integration (`vim/engine.py`)

Modified `train_one_epoch()` to:
- Compute stability loss from all Mamba layers when `use_stability_penalty=True`
- Add stability loss to the main loss
- Log stability loss in metrics

### 4. Ablation Study Scripts

Created 8 individual scripts + 1 runner script:
- `Rorqual_pt-vim-zoh-stab-baseline.sh` - No stabilizers
- `Rorqual_pt-vim-zoh-stab-sn.sh` - Spectral Normalization only
- `Rorqual_pt-vim-zoh-stab-ec.sh` - Eigenvalue Clamping only
- `Rorqual_pt-vim-zoh-stab-sp.sh` - Stability Penalty only
- `Rorqual_pt-vim-zoh-stab-sn-ec.sh` - SN + EC
- `Rorqual_pt-vim-zoh-stab-sn-sp.sh` - SN + SP
- `Rorqual_pt-vim-zoh-stab-ec-sp.sh` - EC + SP
- `Rorqual_pt-vim-zoh-stab-all.sh` - All stabilizers
- `Rorqual_run-stab-enforce-ablation.sh` - Runner for all configs

## Data Flow

```
Command Line Args (main.py)
    ↓
ssm_cfg dictionary
    ↓
create_model() → VisionMamba(ssm_cfg=ssm_cfg)
    ↓
create_block(ssm_cfg=ssm_cfg)
    ↓
Mamba(**ssm_cfg)  # Unpacks stability parameters
    ↓
_construct_A_matrix() → apply_stability_enforcement()
    ↓
Forward Pass → compute_stability_loss() (if enabled)
    ↓
train_one_epoch() → Adds stability_loss to main loss
```

## Testing Strategy

1. **Quick Test**: Run single configuration with seed 0
   ```bash
   bash Rorqual_pt-vim-zoh-stab-baseline.sh 0
   bash Rorqual_pt-vim-zoh-stab-all.sh 0
   ```

2. **Full Ablation**: Run all 8 configurations with 5 seeds each
   ```bash
   bash Rorqual_run-stab-enforce-ablation.sh
   ```

3. **Extract Results**: Compare configurations statistically
   ```bash
   python ../extract_results.py --base-dir ./output/classification_logs
   ```

## Key Design Decisions

1. **Backward Compatibility**: All stabilizers are disabled by default, so existing scripts continue to work.

2. **Flexible Configuration**: Each stabilizer can be enabled independently, enabling comprehensive ablation studies.

3. **Automatic Application**: Spectral normalization and eigenvalue clamping are applied automatically during `_construct_A_matrix()`, requiring no changes to training code.

4. **Explicit Loss**: Stability penalty must be explicitly enabled and is computed during training, giving full control over when it's used.

## Files Modified

- `vim/main.py`: Added arguments and model creation integration
- `vim/engine.py`: Added stability loss computation
- `vim/scripts/CC/Rorqual/StabEnforce-Ablation/`: Created 9 new scripts + README

## Next Steps

1. Test with single seed to verify integration works
2. Run full ablation study with multiple seeds
3. Analyze results to determine optimal stabilizer configuration
4. Potentially tune hyperparameters (epsilon, penalty weight)
