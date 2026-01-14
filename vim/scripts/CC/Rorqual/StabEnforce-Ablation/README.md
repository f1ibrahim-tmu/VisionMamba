# Stability Enforcement Ablation Study Scripts

This directory contains scripts for running ablation studies on the Feature-StabEnforce stability enforcement implementation.

## Overview

The ablation study tests 8 different configurations of stability enforcement for each discretization method:

1. **baseline**: No stabilizers (baseline)
2. **sn**: Spectral Normalization only
3. **ec**: Eigenvalue Clamping only
4. **sp**: Stability Penalty only
5. **sn-ec**: Spectral Normalization + Eigenvalue Clamping
6. **sn-sp**: Spectral Normalization + Stability Penalty
7. **ec-sp**: Eigenvalue Clamping + Stability Penalty
8. **all**: All stabilizers enabled (SN + EC + SP)

## Available Discretization Methods

### Zero Order Hold (ZOH)
- Model: `vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2`
- Output prefix: `vim_tiny_zoh_stab_*`

### Bilinear (Tustin)
- Model: `vim_tiny_patch16_224_bimambav2_bilinear`
- Output prefix: `vim_tiny_bilinear_stab_*`

## Scripts

### ZOH Discretization Scripts

Each script runs training for a specific configuration with ZOH:

- `Rorqual_pt-vim-zoh-stab-baseline.sh` - Baseline (no stabilizers)
- `Rorqual_pt-vim-zoh-stab-sn.sh` - Spectral Normalization only
- `Rorqual_pt-vim-zoh-stab-ec.sh` - Eigenvalue Clamping only
- `Rorqual_pt-vim-zoh-stab-sp.sh` - Stability Penalty only
- `Rorqual_pt-vim-zoh-stab-sn-ec.sh` - SN + EC
- `Rorqual_pt-vim-zoh-stab-sn-sp.sh` - SN + SP
- `Rorqual_pt-vim-zoh-stab-ec-sp.sh` - EC + SP
- `Rorqual_pt-vim-zoh-stab-all.sh` - All stabilizers

### Bilinear Discretization Scripts

Each script runs training for a specific configuration with Bilinear:

- `Rorqual_pt-vim-bilinear-stab-baseline.sh` - Baseline (no stabilizers)
- `Rorqual_pt-vim-bilinear-stab-sn.sh` - Spectral Normalization only
- `Rorqual_pt-vim-bilinear-stab-ec.sh` - Eigenvalue Clamping only
- `Rorqual_pt-vim-bilinear-stab-sp.sh` - Stability Penalty only
- `Rorqual_pt-vim-bilinear-stab-sn-ec.sh` - SN + EC
- `Rorqual_pt-vim-bilinear-stab-sn-sp.sh` - SN + SP
- `Rorqual_pt-vim-bilinear-stab-ec-sp.sh` - EC + SP
- `Rorqual_pt-vim-bilinear-stab-all.sh` - All stabilizers

### Runner Scripts

- `Rorqual_run-stab-enforce-ablation.sh` - Runs all ZOH configurations with multiple seeds
- `Rorqual_run-bilinear-stab-enforce-ablation.sh` - Runs all Bilinear configurations with multiple seeds

## Usage

### Run Single Configuration with Single Seed

**ZOH Discretization:**
```bash
# Run baseline with seed 0
bash Rorqual_pt-vim-zoh-stab-baseline.sh 0

# Run with all stabilizers and seed 1
bash Rorqual_pt-vim-zoh-stab-all.sh 1
```

**Bilinear Discretization:**
```bash
# Run baseline with seed 0
bash Rorqual_pt-vim-bilinear-stab-baseline.sh 0

# Run with all stabilizers and seed 1
bash Rorqual_pt-vim-bilinear-stab-all.sh 1
```

### Run Single Configuration with Multiple Seeds

**ZOH Discretization:**
```bash
# Run baseline with seeds 0-4
for seed in 0 1 2 3 4; do
    bash Rorqual_pt-vim-zoh-stab-baseline.sh $seed
done
```

**Bilinear Discretization:**
```bash
# Run baseline with seeds 0-4
for seed in 0 1 2 3 4; do
    bash Rorqual_pt-vim-bilinear-stab-baseline.sh $seed
done
```

### Run Full Ablation Study

**ZOH Discretization:**
```bash
# Run all 8 configurations with 5 seeds each (40 total runs)
bash Rorqual_run-stab-enforce-ablation.sh
```

**Bilinear Discretization:**
```bash
# Run all 8 configurations with 5 seeds each (40 total runs)
bash Rorqual_run-bilinear-stab-enforce-ablation.sh
```

## Output Structure

After running, output directories will be:

**ZOH Discretization:**
```
./output/classification_logs/
├── vim_tiny_zoh_stab_baseline_seed0/
├── vim_tiny_zoh_stab_baseline_seed1/
├── ...
├── vim_tiny_zoh_stab_sn_seed0/
├── ...
├── vim_tiny_zoh_stab_all_seed0/
└── ...
```

**Bilinear Discretization:**
```
./output/classification_logs/
├── vim_tiny_bilinear_stab_baseline_seed0/
├── vim_tiny_bilinear_stab_baseline_seed1/
├── ...
├── vim_tiny_bilinear_stab_sn_seed0/
├── ...
├── vim_tiny_bilinear_stab_all_seed0/
└── ...
```

## Extract Results

After all runs complete, extract statistics:

```bash
# From parent directory
cd ../..
python vim/scripts/CC/Rorqual/extract_results.py --base-dir ./output/classification_logs
```

Or extract for specific configurations:

**ZOH Discretization:**
```bash
# Extract results for baseline
python vim/scripts/CC/Rorqual/extract_results.py --base-dir ./output/classification_logs --method vim_tiny_zoh_stab_baseline

# Extract results for all stabilizers
python vim/scripts/CC/Rorqual/extract_results.py --base-dir ./output/classification_logs --method vim_tiny_zoh_stab_all
```

**Bilinear Discretization:**
```bash
# Extract results for baseline
python vim/scripts/CC/Rorqual/extract_results.py --base-dir ./output/classification_logs --method vim_tiny_bilinear_stab_baseline

# Extract results for all stabilizers
python vim/scripts/CC/Rorqual/extract_results.py --base-dir ./output/classification_logs --method vim_tiny_bilinear_stab_all
```

## Configuration Details

All scripts use the following default settings:

- **ZOH Model**: `vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2`
- **Bilinear Model**: `vim_tiny_patch16_224_bimambav2_bilinear`
- **Dataset**: CIFAR-100
- **Batch size**: 128
- **Learning rate**: 0.001
- **Stability epsilon (ε)**: 0.01
- **Stability penalty weight (λ_stab)**: 0.1

## Notes

- Each script accepts a seed as the first argument (defaults to 0)
- Scripts automatically resume from checkpoint if available
- For fresh training, remove or comment out the `--resume` line
- All scripts use 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)
- Port conflicts are handled automatically via SLURM job ID or process ID

## Troubleshooting

**Issue**: Script fails with "checkpoint not found"

- **Solution**: Remove or comment out the `--resume` line for fresh training

**Issue**: Port conflicts

- **Solution**: Scripts automatically handle port assignment, but if issues persist, modify MASTER_PORT calculation

**Issue**: Want to test with different hyperparameters

- **Solution**: Edit the individual script files to modify `--stability-epsilon` or `--stability-penalty-weight`
