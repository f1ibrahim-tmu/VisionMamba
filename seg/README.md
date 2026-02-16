# VimSeg - Vision Mamba Segmentation

## Overview

This codebase has been modernized to use **MMEngine ≥ 0.7** and **MMCV 2.x** for compatibility with:
- PyTorch 2.1.0+
- CUDA 12.2+
- H100 GPUs

## Environment Setup

### Python 3.10+ Recommended

```bash
conda create -n vimseg python=3.10
conda activate vimseg
```

### Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio

# Install MMEngine (required for MMCV 2.x)
pip install mmengine>=0.10.0

# Install MMCV 2.x
pip install mmcv>=2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu122/torch2.1/index.html

# Install MMSegmentation 1.x (compatible with MMCV 2.x)
pip install mmsegmentation>=1.0.0

# Install other dependencies
pip install -r seg-requirements.txt
```

**Note:** For specific CUDA/PyTorch versions, check [MMCV installation guide](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html).

### Configure Dataset Path

Edit `seg/configs/_base_/datasets/ade20k.py` and set your `data_root`:

```python
data_root = '/path/to/your/datasets'
```

## Modern Configuration Format

This codebase uses **MMEngine ≥ 0.7** configuration format. See `MMENGINE_CONFIG_GUIDE.md` for details.

### Key Changes from Old Format:

- ✅ `train_cfg`/`val_cfg`/`test_cfg` instead of `runner`
- ✅ `train_dataloader`/`val_dataloader` instead of `samples_per_gpu`
- ✅ `optim_wrapper` instead of `optimizer_config`
- ✅ `param_scheduler` instead of `lr_config`
- ✅ `default_hooks` instead of `log_config`/`checkpoint_config`
- ✅ `AmpOptimWrapper` instead of apex AMP

### Example Config Structure:

```python
# Training loop configuration
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Dataloader configuration
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True)
)

# Optimizer configuration
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=60000, by_epoch=False)
]

# Hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=4)
)
```

## Training

### Single GPU Training

```bash
cd seg
python train.py configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k.py \
    --work-dir work_dirs/vim_tiny
```

### Multi-GPU Training

```bash
cd seg
torchrun --nproc_per_node=4 train.py \
    configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k.py \
    --work-dir work_dirs/vim_tiny \
    --launcher pytorch
```

### Using Training Scripts

```bash
bash scripts/ft_vim_tiny_upernet.sh
```

## Testing/Evaluation

### Single GPU Testing

```bash
cd seg
python test.py configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k.py \
    checkpoint.pth --eval mIoU
```

### Multi-GPU Testing

```bash
cd seg
torchrun --nproc_per_node=4 test.py \
    configs/vim/upernet/upernet_vim_tiny_24_512_slide_200k.py \
    checkpoint.pth --eval mIoU \
    --launcher pytorch
```

### Using Evaluation Scripts

```bash
bash scripts/eval_vim_tiny_upernet.sh
```

## Migration from Old Format

If you have configs using the old MMCV 1.x format, see:
- `MMENGINE_CONFIG_GUIDE.md` - Complete guide to new config format
- `MMCV_2x_MIGRATION_GUIDE.md` - Detailed migration instructions
- `CRITICAL_MIGRATION_ISSUES.md` - Known issues and fixes

## Deprecated Features

The following are **deprecated** and will not work:

- ❌ `IterBasedRunner` / `EpochBasedRunner` → Use `Runner` with `train_cfg`
- ❌ `samples_per_gpu` → Use `train_dataloader.batch_size`
- ❌ `apex` AMP → Use `AmpOptimWrapper`
- ❌ `mmseg.apis.multi_gpu_test` → Use `Runner.test()`
- ❌ `workflow` config → Use `train_dataloader`/`val_dataloader`

## Requirements

See `seg-requirements.txt` for complete dependency list.

**Key Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- MMEngine ≥ 0.10.0
- MMCV ≥ 2.0.0
- MMSegmentation ≥ 1.0.0

## Troubleshooting

### Common Issues

1. **ImportError: cannot import name 'IterBasedRunner'**
   - Solution: Your configs need to use `train_cfg` instead of `runner`. See `MMENGINE_CONFIG_GUIDE.md`.

2. **Dataloader errors**
   - Solution: Ensure you have explicit `train_dataloader` configs. See migration guide.

3. **Optimizer errors**
   - Solution: Use `optim_wrapper` instead of `optimizer_config`.

4. **Hook registration errors**
   - Solution: Custom hooks must use `@HOOKS.register_module()` from `mmengine.registry`.

## Documentation

- `MMENGINE_CONFIG_GUIDE.md` - Complete MMEngine config format guide
- `MMCV_2x_MIGRATION_GUIDE.md` - Migration from MMCV 1.x to 2.x
- `CRITICAL_MIGRATION_ISSUES.md` - Known issues and solutions
- `MIGRATION_SUMMARY.md` - Overview of changes

## Acknowledgement

Vim semantic segmentation is built with:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (v1.x)
- [MMEngine](https://github.com/open-mmlab/mmengine) (≥ 0.7)
- [MMCV](https://github.com/open-mmlab/mmcv) (v2.x)
- [EVA-02](https://github.com/baaivision/EVA/tree/master/EVA-02)
- [BEiT](https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation)
