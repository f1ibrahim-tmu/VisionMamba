# Running CIFAR-100 Experiments with Multiple Random Seeds

This directory contains scripts modified to support running experiments with multiple random seeds for statistical significance.

## Overview

To ensure statistical significance, experiments should be run with **3-5 different random seeds** and report **Mean ± Standard Deviation**. If error bars overlap, the improvement is not statistically significant.

## Modified Scripts

All individual training scripts have been updated to accept a seed parameter:

- `pt-cifar100-vim-zoh.sh`
- `pt-cifar100-vim-foh.sh`
- `pt-cifar100-vim-bilinear.sh`
- `pt-cifar100-vim-poly.sh`
- `pt-cifar100-vim-highorder.sh`
- `pt-cifar100-vim-rk4.sh`

### Changes Made

1. **Seed Parameter**: Each script now accepts a seed as the first argument (defaults to 0)
2. **Output Directory**: Includes seed in path: `vim_tiny_{method}_seed{SEED}`
3. **Resume Path**: Updated to match the new output directory structure

### Usage

**Run with default seed (0):**
```bash
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-zoh.sh
```

**Run with specific seed:**
```bash
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-zoh.sh 1
bash ./scripts/CVIS/cifar100/pt-cifar100-vim-zoh.sh 2
```

## Helper Scripts

### 1. Run Single Method with Multiple Seeds

**Script**: `run-single-method-seeds.sh`

Run a single discretization method with multiple seeds:

```bash
# IMPORTANT: Use 'bash' not 'sh' - the script uses bash-specific features
# Run ZOH with default seeds (0,1,2,3,4)
bash ./scripts/CVIS/cifar100/run-single-method-seeds.sh zoh

# Run HighOrder with custom seeds
bash ./scripts/CVIS/cifar100/run-single-method-seeds.sh highorder 0,1,2,3,4

# Available methods: zoh, foh, bilinear, poly, highorder, rk4
```

### 2. Run All Methods with Multiple Seeds

**Script**: `run-multiple-seeds.sh`

Run all discretization methods with multiple seeds sequentially:

```bash
# IMPORTANT: Use 'bash' not 'sh' - the script uses bash-specific features
bash ./scripts/CVIS/cifar100/run-multiple-seeds.sh

# Or make it executable and run directly:
chmod +x ./scripts/CVIS/cifar100/run-multiple-seeds.sh
./scripts/CVIS/cifar100/run-multiple-seeds.sh
```

**Note**: Do NOT use `sh` - use `bash` instead. The script uses bash arrays which are not supported in plain `sh`.

This will run each method (ZOH, FOH, Bilinear, Poly, HighOrder, RK4) with seeds 0, 1, 2, 3, 4.

### 3. Extract Results and Compute Statistics

**Script**: `extract_results.py`

Extract maximum accuracy from each seed run and compute Mean ± Std:

```bash
# Extract results for all methods
python ./scripts/CVIS/cifar100/extract_results.py

# Extract results for a specific method
python ./scripts/CVIS/cifar100/extract_results.py --method zoh
python ./scripts/CVIS/cifar100/extract_results.py --method highorder

# Use custom seeds
python ./scripts/CVIS/cifar100/extract_results.py --method zoh --seeds 0,1,2,3,4

# Custom base directory
python ./scripts/CVIS/cifar100/extract_results.py --base-dir ./output/cifar100
```

**Output:**
- Prints results for each seed
- Computes Mean ± Standard Deviation
- Shows Min/Max values
- Saves summary to `./output/cifar100/results_summary.json`

## Workflow Example

### Step 1: Run Experiments

```bash
# Option A: Run all methods with all seeds (takes a long time)
bash ./scripts/CVIS/cifar100/run-multiple-seeds.sh

# Option B: Run one method at a time
bash ./scripts/CVIS/cifar100/run-single-method-seeds.sh zoh
bash ./scripts/CVIS/cifar100/run-single-method-seeds.sh foh
# ... etc
```

### Step 2: Extract Results

```bash
# After all runs complete, extract statistics
python ./scripts/CVIS/cifar100/extract_results.py
```

### Step 3: Report Results

The output will show:
```
Method: vim_tiny_zoh
  Seed 0: 85.23%
  Seed 1: 85.45%
  Seed 2: 85.12%
  Seed 3: 85.67%
  Seed 4: 85.34%

  Results: 5/5 seeds completed
  Mean ± Std: 85.36 ± 0.20%
  Min: 85.12%, Max: 85.67%
```

## Output Structure

After running with multiple seeds, your output directory will look like:

```
./output/cifar100/
├── vim_tiny_zoh_seed0/
│   ├── log.txt
│   ├── checkpoint.pth
│   └── best_checkpoint.pth
├── vim_tiny_zoh_seed1/
│   └── ...
├── vim_tiny_zoh_seed2/
│   └── ...
├── vim_tiny_zoh_seed3/
│   └── ...
├── vim_tiny_zoh_seed4/
│   └── ...
└── results_summary.json  (created by extract_results.py)
```

## Notes

- **Resume Training**: If a run fails, you can resume by running the same script with the same seed (it will automatically resume from checkpoint)
- **Parallel Execution**: You can run different seeds in parallel on different GPUs by modifying the `CUDA_VISIBLE_DEVICES` in each script
- **Statistical Significance**: Use at least 3 seeds (preferably 5) for reliable statistics
- **Error Bars**: If error bars (Mean ± Std) overlap between methods, the difference is not statistically significant

## Troubleshooting

**Issue**: `Syntax error: "(" unexpected` when running with `sh`
- **Solution**: Use `bash` instead of `sh`. The scripts use bash-specific features (arrays). Run: `bash ./scripts/CVIS/cifar100/run-multiple-seeds.sh`

**Issue**: Script says checkpoint not found
- **Solution**: Remove the `--resume` line or set it to empty string for fresh training

**Issue**: Results extraction shows "Not found"
- **Solution**: Check that `log.txt` exists in the output directory and contains valid JSON logs

**Issue**: Want to use different seeds
- **Solution**: Modify the `SEEDS` array in `run-multiple-seeds.sh` or pass custom seeds to `run-single-method-seeds.sh`

